# random_test_env.py

import numpy as np
import matplotlib.pyplot as plt
from pcb_env_sequential import PCBRLEnv
import matplotlib
matplotlib.use('TkAgg')
def visualize_layout(env, title="PCB Layout"):
    """Matplotlib 可视化 env 当前布局"""
    fig, ax = plt.subplots(figsize=(6,6))
    for cell in env.cells:
        pts = np.array(eval(cell['contour']), dtype=int)
        xs = np.append(pts[:,0], pts[0,0])
        ys = np.append(pts[:,1], pts[0,1])
        ax.plot(xs, ys, linewidth=2, label=cell['cellName'])
    ax.set_xlim(0,255)
    ax.set_ylim(0,255)
    ax.set_aspect('equal')
    ax.legend(fontsize='small', loc='upper right')
    ax.set_title(title)
    plt.show()

def random_test_env(env: PCBRLEnv, max_steps=100, step=32):
    """
    随机测试顺序环境：每步针对当前器件随机选一个合法动作执行，
    直到 done 或 max_steps，打印每步信息并可视化最终布局。
    """
    state = env.reset()
    layout, cur = state['layout'], state['current_device']
    print(f"Start placement. first device idx={cur}, total devices={env.n_dev}")

    for t in range(max_steps):
        mask = env.get_action_mask(step=step)
        if not mask:
            print(f"Step {t+1}: device idx={cur} no valid actions, skipping")
            # advance even if no actions
            _, reward, done, info = env.step((0,0,0))
        else:
            act = mask[np.random.randint(len(mask))]
            _, reward, done, info = env.step(act)
            print(f"Step {t+1}: device idx={cur} "
                  f"action={act} reward={reward:.4f} "
                  f"done={done} invalid={info.get('invalid',False)}")

        if done:
            print(f"--> All devices placed at step {t+1}, final reward={reward:.4f}")
            break

        # 更新 for next step
        state = env._get_state()  # although layout updated internally
        cur = env.step_idx < env.n_dev and env.device_order[env.step_idx] or -1

    else:
        print(f"Reached max_steps={max_steps}, last reward={reward:.4f}, done={done}")

    visualize_layout(env, title="Random Sequential Placement Result")


if __name__ == "__main__":
    # 调整为你的 JSON 路径
    pcb_json = "pcb_pre_jsons/pcb_cells_1.json"
    env = PCBRLEnv(pcb_json)
    random_test_env(env, max_steps=50, step=32)
