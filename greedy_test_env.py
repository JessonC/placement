import random
import copy
import numpy as np
from placement_game import PCBRLEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def random_action(env):
    module_choice = random.randrange(env.n_cells - 1)
    move_dir = random.randrange(5)      # 0=无, 1=上,2=下,3=左,4=右
    rot_idx   = random.randrange(4)      # 0,1,2,3 对应 0°,90°,180°,270°
    return (module_choice, move_dir, rot_idx)

def greedy_test(env, iterations=50, trials_per_iter=2000):
    reward_history = []
    action_history = []
    env.reset()
    for it in range(iterations):
        best_r, best_act = -1e9, None
        # 在副本上随机试多次
        for _ in range(trials_per_iter):
            act = random_action(env)
            env_copy = copy.deepcopy(env)
            _, r, _, _ = env_copy.step(act)
            if r > best_r:
                best_r, best_act = r, act
        # 应用到真实环境
        _, real_r, _, _ = env.step(best_act)
        reward_history.append(real_r)
        action_history.append(best_act)
        print(f"Iter {it+1}/{iterations} — best_random_r={best_r:.2f}, applied_r={real_r:.2f}, action={best_act}")
    return reward_history, action_history

def plot_final_layout(env, figsize=(8,8)):
    """
    可视化 env 中所有 cell 的 contour、pin 点和 net 连线
    """
    fig, ax = plt.subplots(figsize=figsize)
    # 画器件轮廓和引脚
    for cell in env.cell_list:
        contour = np.array(eval(cell["contour"]))
        # close the loop
        xs = np.append(contour[:,0], contour[0,0])
        ys = np.append(contour[:,1], contour[0,1])
        ax.plot(xs, ys, '-', linewidth=2)
        # label
        cx, cy = np.array(eval(cell["center"]))
        ax.text(cx, cy, cell["cellName"], ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='black', pad=0.2, alpha=0.7))
        # pin 点
        for pin in cell["pinList"]:
            px, py = np.array(eval(pin["center"]))
            ax.plot(px, py, 'ro', markersize=3)

    # 画 net 连线
    for net in env.net_list:
        coords = []
        for pin_ref in net["pinList"]:
            # 在 cell_list 里找到对应 pin
            cell = next(c for c in env.cell_list if c["cellName"] == pin_ref["cellName"])
            pin  = next(p for p in cell["pinList"]   if p["pinName"] == pin_ref["pinName"])
            coords.append(np.array(eval(pin["center"])))
        # 顺序连线
        for i in range(len(coords)-1):
            x1,y1 = coords[i]; x2,y2 = coords[i+1]
            ax.plot([x1,x2],[y1,y2], color='gray', linewidth=1, alpha=0.7)

    ax.set_xlim(0, env.grid_size-1)
    ax.set_ylim(0, env.grid_size-1)
    ax.set_aspect('equal')
    ax.set_title("Final PCB Layout")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. 初始化环境
    env = PCBRLEnv('pcb_pre_jsons/pcb_cells_1.json')

    # 2. 运行贪婪测试
    rewards, actions = greedy_test(env, iterations=20, trials_per_iter=1000
                                   )

    # 3. 打印结果
    print("\nGreedy Test Done")
    print("Rewards:", rewards)
    print("Actions:", actions)

    # 4. 可视化最终布局
    plot_final_layout(env)