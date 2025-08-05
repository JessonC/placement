# greedy_search_test.py
import numpy as np
from copy import deepcopy
from pcb_rl_env_two_layers import PCBRLEnv  # 你的两层环境实现文件

def greedy_search(env: PCBRLEnv):
    state = env.reset()
    done = False
    step = 0

    # 每一步，在 [-128,128] 的网格上，以 env.step_size 为步长枚举 dx,dy
    grid = list(range(-128, 129, env.step_size))

    print("=== Start Greedy Search ===")
    while not done:
        best_action = None
        best_cost   = float('inf')

        # 枚举所有候选偏移
        for dx in grid:
            for dy in grid:
                action = (dx, dy)
                # 复制当前环境，尝试一步
                env2 = deepcopy(env)
                idx2, r2, done2, _ = env2.step(action)

                # 计算放置后 cost：wirelength + nc_weight*net_crossing + density_weight*density
                wl  = env2._compute_wirelength()
                nc  = env2._compute_net_crossing()
                den = env2._compute_density(env2.density_grid_N)
                cost = wl + env2.nc_weight * nc + env2.density_weight * den

                if cost < best_cost:
                    best_cost   = cost
                    best_action = action

        # 在原环境里执行最优动作
        idx, reward, done, _ = env.step(best_action)
        print(f"Step {step:02d}: place idx={idx}, action={best_action}, cost={best_cost:.3f}, reward={reward:.6f}, done={done}")
        step += 1

    print("=== Greedy Search Finished ===")
    print(f"Final reward = {reward:.6f}")
    return env

if __name__ == '__main__':
    # 修改下面路径为你的 PCB JSON
    JSON_PATH = 'pcb_pre_jsons/pcb_cells_1.json'

    # 初始化环境
    env = PCBRLEnv(pcb_json_path=JSON_PATH,
                   gamma=1.0,
                   grid_size=256,
                   nc_weight=1.0,
                   density_weight=1.0,
                   density_grid_N=16,
                   step_size=32)

    # 运行贪婪搜索
    final_env = greedy_search(env)

    # 可视化最终布局
    final_env.visualize()
