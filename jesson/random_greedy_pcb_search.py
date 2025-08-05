# random_play.py

import random
from pcb_rl_env_two_layers import PCBRLEnv

if __name__ == '__main__':
    env = PCBRLEnv('pcb_pre_jsons/pcb_cells_1.json',
                   gamma=1.0,
                   nc_weight=1.0,
                   density_weight=1.0,
                   density_grid_N=16,
                   step_size=32)
    env.reset()
    print("Start random play...")
    while True:
        dx = random.choice(range(-128, 129, env.step_size))
        dy = random.choice(range(-128, 129, env.step_size))
        idx, reward, done, _ = env.step((dx, dy))
        print(f"Placed device {idx}, action=({dx},{dy}), reward={reward:.4f}, done={done}")
        if done:
            print("Final reward:", reward)
            env.visualize()
            break
