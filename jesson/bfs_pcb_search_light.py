# bfs_pcb_search_light.py

import json
import numpy as np
from collections import deque

from pcb_env_mask_reward import PCBRLEnv

def state_hash(state: np.ndarray) -> tuple:
    """将 state(n_cells×3) 转成 hashable tuple"""
    return tuple(state.flatten().tolist())

def bfs_search(env: PCBRLEnv, max_depth: int = 2, step: int = 64):
    """
    广度优先搜索最优动作序列（在 max_depth 步内）
    - 不 deepcopy 环境；而是存动作序列，重播来生成新状态
    - step 控制动作掩码 rel_x,rel_y 的离散精度
    """
    best_reward = -1e9
    best_seq    = []

    # 初始队列只包含空动作序列
    queue = deque([[]])
    visited = set()

    while queue:
        seq = queue.popleft()

        # 重播 seq 来获得当前 env 状态
        env.reset()
        for act in seq:
            env.step(act)
        st = env._get_state()
        h  = state_hash(st)

        # 如果已经访问过相同布局，跳过
        if h in visited:
            continue
        visited.add(h)

        # 评估 reward
        reward = env._compute_reward()
        if reward > best_reward:
            best_reward = reward
            best_seq    = list(seq)

        # 如果已达最大深度，不展开
        if len(seq) >= max_depth:
            continue

        # 否则扩展：对每个非主芯片都尝试动作
        for dev_choice in range(len(env.non_main)):
            mask = env.get_action_mask(dev_choice, step=step)
            for act in mask:
                queue.append(seq + [act])

    return best_reward, best_seq

if __name__ == "__main__":
    # 示例
    env = PCBRLEnv("pcb_pre_jsons/pcb_cells_1.json", max_steps=10)

    print("Running BFS...")
    best_reward, best_actions = bfs_search(env, max_depth=2, step=64)
    print("Best reward:", best_reward)
    print("Best action sequence:")
    for a in best_actions:
        print(" ", a)

    # 将最佳动作序列 replay 并可视化
    env.reset()
    for a in best_actions:
        env.step(a)

    # 简单可视化
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6,6))
    for cell in env.cell_list:
        pts = np.array(eval(cell["contour"]),dtype=int)
        xs = np.append(pts[:,0], pts[0,0])
        ys = np.append(pts[:,1], pts[0,1])
        ax.plot(xs, ys, linewidth=2, label=cell["cellName"])
    ax.set_xlim(0,255); ax.set_ylim(0,255)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("BFS Optimized Layout")
    plt.show()
