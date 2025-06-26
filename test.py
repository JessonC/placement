import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


# ====== Actor-Critic 网络定义 ======
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        state_dim: 环境状态向量的维度
        action_dim: 动作空间大小（离散动作数）
        hidden_dim: 网络隐藏层大小
        """
        super(ActorCritic, self).__init__()
        # 共享底座网络：用于提取状态特征
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 全连接层：状态 -> 隐藏
            nn.ReLU()                          # 激活函数
        )
        # Actor 分支：输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # 隐藏 -> 隐藏
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), # 隐藏 -> 动作数 logits
            nn.Softmax(dim=-1)                 # Softmax 得到各动作概率
        )
        # Critic 分支：输出状态值估计 V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # 隐藏 -> 隐藏
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)           # 隐藏 -> 单一标量值
        )

    def forward(self, x):
        """
        x: 输入状态，shape=(batch_size, state_dim)
        返回:
          action_probs:  shape=(batch_size, action_dim)
          state_value:   shape=(batch_size, 1)
        """
        base_out = self.base(x)               # 共享特征提取
        action_probs = self.actor(base_out)   # Actor 分支
        state_value  = self.critic(base_out)  # Critic 分支
        return action_probs, state_value


# ====== PPO 智能体定义 ======
class PPOAgent:
    def __init__(self,
                 env_name='CartPole-v1',
                 gamma=0.99,
                 clip_eps=0.2,
                 lr=3e-4,
                 k_epochs=4,
                 ent_coef=0.01,
                 gae_lambda=0.95,
                 batch_size=64):
        """
        env_name: Gym 环境名称
        gamma: 折扣因子 γ
        clip_eps: PPO 剪切范围 ε
        lr: 学习率
        k_epochs: 每次 update 的内部迭代次数
        ent_coef: 熵正则化系数
        gae_lambda: GAE 参数 λ
        batch_size: 批次大小（这里只是预留，示例中并未做严格 batch 切分）
        """
        # 创建环境
        self.env = gym.make(env_name)
        # 存超参数
        self.gamma      = gamma
        self.clip_eps   = clip_eps
        self.k_epochs   = k_epochs
        self.ent_coef   = ent_coef
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size

        # 从环境中获取状态和动作维度
        state_dim  = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        # 初始化 Actor-Critic 网络与优化器
        self.policy    = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        """
        根据当前策略（policy）选动作，并返回：
          action:    int，实际执行的动作
          log_prob:  torch.Tensor，动作对数概率（后续用于 ratio 计算）
          value:     torch.Tensor，当前状态的值函数估计 V(s)
          entropy:   torch.Tensor，策略熵（用于增加探索）
        """
        # 将 numpy 状态转为 torch Tensor，添加 batch 维度
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        # 前向传播，得到动作分布和状态值
        probs, value = self.policy(state_tensor)
        # 构造离散分布
        dist   = Categorical(probs)
        action = dist.sample()               # 按概率采样动作
        # 使用detach截断计算图，避免梯度重复反向传播
        return action.item(), dist.log_prob(action).detach(), value.detach().squeeze(0), dist.entropy().detach()

    def compute_gae(self, rewards, values, dones):
        """
        使用广义优势估计（GAE）计算优势函数 A_t
        rewards: list[float]，回报序列
        values:  list[float]，对应的 V(s_t) 估计
        dones:   list[bool]，是否终止
        返回: advantages: list[float]
        """
        advantages = []
        gae = 0
        # 在 values 末尾补 0，便于计算最后一步 δ
        values = values + [0]
        # 逆序计算
        for step in reversed(range(len(rewards))):
            # δ_t = r_t + γ V(s_{t+1}) (1−done) − V(s_t)
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            # GAE: A_t = δ_t + γ λ (1−done) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, memory):
        """
        根据 memory 中收集的数据执行一次 PPO 更新
        memory: dict 包含 keys:
          'states', 'actions', 'log_probs', 'rewards', 'dones', 'values', 'returns', 'advantages'
        """
        # 把列表转为 Tensor，方便并行计算
        states     = torch.from_numpy(np.array(memory['states'])).float()
        actions    = torch.LongTensor(memory['actions']).unsqueeze(-1)
        old_logps  = torch.stack(memory['log_probs']).unsqueeze(-1)
        returns    = torch.FloatTensor(memory['returns']).unsqueeze(-1)
        advantages = torch.FloatTensor(memory['advantages']).unsqueeze(-1)

        # 多轮 epoch 更新
        for _ in range(self.k_epochs):
            # 1) 重新计算动作概率与状态值
            probs, values = self.policy(states)        # 新策略
            dist           = Categorical(probs)
            new_logps      = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
            entropy        = dist.entropy().mean()     # 平均熵

            # 2) 计算 ratio 与剪切目标
            ratios = torch.exp(new_logps - old_logps)  # r_t(θ)
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1-self.clip_eps, 1+self.clip_eps) * advantages
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            total_loss  = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy

            # 3) 梯度下降一步
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # 清空 memory，为下次采样做准备
        for key in memory:
            memory[key].clear()

    def train(self, max_episodes=1000, update_timestep=2000):
        """
        主训练循环：
          - 持续采样环境交互数据到 memory
          - 达到 update_timestep 或 episode 结束时，计算 returns & advantages
          - 调用 update() 更新策略
        """
        timestep = 0
        # 用 dict 存储交互数据
        memory = {k: [] for k in ['states','actions','log_probs','rewards','dones','values']}
        episode_rewards = []

        for episode in range(1, max_episodes+1):
            # reset 返回两个值：obs, info
            state = self.env.reset()
            if isinstance(state, tuple):  # gym>=0.26
                state = state[0]
            done     = False
            ep_reward = 0

            # 一个完整 episode 交互
            while not done:
                # 选择动作
                action, logp, value, _ = self.select_action(state)
                # step 返回 (obs, reward, terminated, truncated, info)
                # terminated=True：任务因失败而终止，truncated=True:超时或到达最大步数而终止
                next_step = self.env.step(action)
                if len(next_step) == 5:  # gym>=0.26
                    next_state, reward, terminated, truncated, _ = next_step
                    done = terminated or truncated
                else:  # gym<0.26
                    next_state, reward, done, _ = next_step

                # 存储数据到 memory
                memory['states'].append(state)
                memory['actions'].append(action)
                memory['log_probs'].append(logp)
                memory['values'].append(value.item())
                memory['rewards'].append(reward)
                memory['dones'].append(done)

                state     = next_state
                ep_reward += reward
                timestep  += 1

                # 达到 update 条件则进行一次策略更新
                if timestep % update_timestep == 0 or done:
                    # 计算折扣回报 returns
                    returns   = []
                    discounted = 0
                    for r, d in zip(reversed(memory['rewards']), reversed(memory['dones'])):
                        if d:
                            discounted = 0
                        discounted = r + self.gamma * discounted
                        returns.insert(0, discounted)

                    # 计算 GAE 优势
                    advantages = self.compute_gae(memory['rewards'],
                                                  memory['values'],
                                                  memory['dones'])
                    # 存回 memory
                    memory['returns']    = returns
                    memory['advantages'] = advantages

                    # 执行更新，并重置 timestep
                    self.update(memory)
                    timestep = 0

            episode_rewards.append(ep_reward)
            # 每 10 集输出一次平均回报
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}\tAverage Reward (last 10): {avg_reward:.2f}")


if __name__ == '__main__':
    # 指定环境与最大迭代集数
    agent = PPOAgent(env_name='CartPole-v1')
    agent.train(max_episodes=10000)
