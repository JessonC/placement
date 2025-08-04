# random_greedy_save.py

import json
import random
from copy import deepcopy
from pcb_rl_env_final import PCBRLEnv

def evaluate_sequence(env, actions):
    """
    给定已重置的 env 和一系列 actions，
    按序执行并返回最终reward 和 完整的 env.data
    """
    env = deepcopy(env)
    reward = 0.0
    done = False
    for a in actions:
        _, reward, done, _ = env.step(a)
        if done:
            break
    return reward, env

def random_rollout(env, rel=128, step=32):
    """
    随机 rollout 一条完整的布局序列，直到 done，
    返回 action 列表 和 最终 reward。
    """
    actions = []
    while True:
        mask = env.get_action_mask(step=step)
        if not mask:
            # 没有合法动作，提前结束
            break
        a = random.choice(mask)
        actions.append(a)
        _, reward, done, _ = env.step(a)
        if done:
            return actions, reward
    return actions, 0.0

def greedy_improve(env, base_actions, rel=128, step=32, nc=10):
    """
    在 base_actions 基础上做贪婪局部优化：
    对每个位置尝试 nc 次随机替换，若提升则保留。
    """
    best_actions = base_actions[:]
    best_reward, _ = evaluate_sequence(env, best_actions)

    for i in range(len(best_actions)):
        for _ in range(nc):
            candidate = best_actions[:]
            # 前 i 步执行到新的 env 副本
            env_copy = deepcopy(env)
            for a in candidate[:i]:
                env_copy.step(a)
            # 在第 i 步尝试新的随机 mask 动作
            mask = env_copy.get_action_mask(step=step)
            if not mask:
                continue
            a_new = random.choice(mask)
            candidate[i] = a_new
            # 评估新序列
            r, _ = evaluate_sequence(env, candidate)
            if r > best_reward:
                best_reward = r
                best_actions = candidate[:]
    return best_actions, best_reward

if __name__ == '__main__':
    pcb_json = 'pcb_pre_jsons/pcb_cells_1.json'
    gamma = 1.0
    rel   = 128
    step  = 32

    # 初始化环境
    base_env = PCBRLEnv(pcb_json, gamma)
    base_env.reset()

    # 1) 多次随机 rollouts，取最优
    best_actions, best_r = None, -1.0
    for _ in range(100):
        env = deepcopy(base_env)
        actions, r = random_rollout(env, rel=rel, step=step)
        if r > best_r:
            best_r = r
            best_actions = actions[:]

    print(f"Random best reward: {best_r:.6f}")

    # 2) 贪婪改进
    env = deepcopy(base_env)
    imp_actions, imp_r = greedy_improve(env, best_actions, rel=rel, step=step, nc=10)
    print(f"Greedy improved reward: {imp_r:.6f}")

    # 3) 保存最优布局到 JSON
    env = deepcopy(base_env)
    for a in imp_actions:
        env.step(a)
    with open('best_layout.json', 'w', encoding='utf-8') as f:
        json.dump({'cellList': env.cells, 'netList': env.nets}, f, indent=2)

    print("Saved best_layout.json")
