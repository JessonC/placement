# random_greedy_with_wires.py

import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pcb_env_sequential import PCBRLEnv

def replay_sequence(env, seq):
    """重放动作序列，不返回中间状态，只更新 env."""
    env.reset()
    for act in seq:
        env.step(act)

def random_rollout(pcb_json, step=32):
    env = PCBRLEnv(pcb_json)
    env.reset()
    seq, reward = [], 0.0
    while True:
        state = env._get_state()
        cur = state['current_device']
        if cur < 0:
            # 全部放置完成
            _, reward, done, _ = env.step((0,0,0))  # 触发 final reward
            break
        mask = env.get_action_mask(step=step)
        act = random.choice(mask) if mask else (0,0,0)
        _, reward, done, _ = env.step(act)
        seq.append(act)
        if done:
            break
    return seq, reward

def greedy_improve(pcb_json, base_seq, step=32):
    best = list(base_seq)
    best_r = 0.0
    # 初次评估
    env = PCBRLEnv(pcb_json)
    replay_sequence(env, best)
    _, best_r, _, _ = env.step((0,0,0))
    # 逐位置优化
    for i in range(len(best)):
        env = PCBRLEnv(pcb_json)
        # prefix replay
        replay_sequence(env, best[:i])
        mask = env.get_action_mask(step=step)
        if not mask: continue
        local_best_act = best[i]
        local_best_r = best_r
        for act in mask:
            cand = best.copy()
            cand[i] = act
            env2 = PCBRLEnv(pcb_json)
            replay_sequence(env2, cand)
            _, r2, _, _ = env2.step((0,0,0))
            if r2 > local_best_r:
                local_best_r = r2
                local_best_act = act
        best[i] = local_best_act
        best_r = local_best_r
    return best, best_r

def random_greedy_search(pcb_json, n_rollouts=20, step=32):
    best_seq, best_r = [], -1.0
    for _ in range(n_rollouts):
        seq, r = random_rollout(pcb_json, step)
        if r > best_r:
            best_r, best_seq = r, seq
    print(f"[Random] best reward = {best_r:.4f}")
    imp_seq, imp_r = greedy_improve(pcb_json, best_seq, step)
    print(f"[Greedy] improved reward = {imp_r:.4f}")
    return best_seq, best_r, imp_seq, imp_r

def visualize_with_wires(pcb_json, seq):
    """
    重放 seq 并可视化器件轮廓 + 引脚连线
    """
    env = PCBRLEnv(pcb_json)
    replay_sequence(env, seq)

    fig, ax = plt.subplots(figsize=(6,6))
    # 画器件轮廓
    for idx, cell in enumerate(env.cells):
        pts = np.array(eval(cell['contour']), dtype=int)
        xs = np.append(pts[:,0], pts[0,0])
        ys = np.append(pts[:,1], pts[0,1])
        ax.plot(xs, ys, linewidth=2, label=cell['cellName'])

    # 画连线
    for net in env.nets:
        # 收集每个 pin 的坐标
        coords = []
        for p in net['pinList']:
            # 在 cellList 中找到对应 pin
            for cell in env.cells:
                if cell['cellName'] == p['cellName']:
                    for pin in cell['pinList']:
                        if pin['pinName'] == p['pinName']:
                            coords.append(np.array(eval(pin['center']), dtype=int))
                            break
                    break
        # 按顺序连线
        if len(coords) >= 2:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlim(0,255)
    ax.set_ylim(0,255)
    ax.set_aspect('equal')
    ax.set_title("PCB Layout with Wires")
    ax.legend(fontsize='small', loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pcb_json = "pcb_pre_jsons/pcb_cells_1.json"
    # 运行随机+贪婪搜索
    best_seq, best_r, imp_seq, imp_r = random_greedy_search(
        pcb_json, n_rollouts=50, step=32
    )
    print("Improved sequence:", imp_seq)
    # 可视化最终布局及连线
    visualize_with_wires(pcb_json, imp_seq)
