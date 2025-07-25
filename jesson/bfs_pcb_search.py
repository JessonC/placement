# bfs_pcb_search.py

import json
import numpy as np
from copy import deepcopy
from collections import deque

from pcb_env_mask_reward import PCBRLEnv


def state_hash(state: np.ndarray) -> tuple:
    """
    将 state (n_cells×3 数组) 转为可哈希的 tuple。
    """
    return tuple(state.flatten().tolist())


def bfs_search(env: PCBRLEnv, max_depth: int = 3, step: int = 32):
    """
    使用广度优先搜索（BFS）在 max_depth 步内寻找最优布局。

    每一步，对于所有非主芯片设备，都尝试它的 action mask，
    并扩展新的环境状态。记录最佳 reward 和对应 action 序列。

    注意：分支因子可能很大，请将 max_depth 和 step 调小。
    """
    # 初始状态
    init_state = env.reset()
    init_hash = state_hash(init_state)

    Node = tuple  # (env, action_sequence)
    queue = deque()
    queue.append((deepcopy(env), []))

    best_reward = -float('inf')
    best_seq = None

    visited = set([init_hash])

    while queue:
        curr_env, seq = queue.popleft()
        curr_state = curr_env._get_state()
        curr_hash = state_hash(curr_state)
        curr_reward = curr_env._compute_reward()

        # 更新最佳
        if curr_reward > best_reward:
            best_reward = curr_reward
            best_seq = seq

        # 深度限制
        if len(seq) >= max_depth:
            continue

        # 对每个非主芯片设备都尝试动作
        for dev_idx in range(len(curr_env.non_main)):
            # 获取该设备合法 action 掩码
            mask = curr_env.get_action_mask(dev_idx, step=step)
            for action in mask:
                # 克隆环境并执行 action
                new_env = deepcopy(curr_env)
                next_state, reward, done, info = new_env.step(action)
                h = state_hash(next_state)
                if h in visited:
                    continue
                visited.add(h)

                # 新的序列
                new_seq = seq + [action]
                # 入队
                queue.append((new_env, new_seq))

    return best_reward, best_seq


if __name__ == "__main__":
    # 示例用法
    pcb_file = "pcb_pre_jsons/pcb_cells_1.json"
    env = PCBRLEnv(pcb_file, max_steps=10)

    print("Starting BFS search...")
    best_reward, best_actions = bfs_search(env, max_depth=2, step=64)
    print(f"Best reward found: {best_reward}")
    print("Best action sequence:")
    for act in best_actions:
        print("  ", act)

    # 将最佳序列应用到全新环境并可视化
    final_env = PCBRLEnv(pcb_file, max_steps=10)
    final_env.reset()
    for action in best_actions:
        state, _, _, _ = final_env.step(action)

    # 可视化最终布局
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 8))
    for cell in final_env.cell_list:
        contour = np.array(eval(cell["contour"]), dtype=int)
        xs = np.append(contour[:, 0], contour[0, 0])
        ys = np.append(contour[:, 1], contour[0, 1])
        ax.plot(xs, ys, linewidth=2, label=cell["cellName"])
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("BFS Optimized PCB Layout")
    plt.show()
