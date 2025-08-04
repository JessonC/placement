from pcb_rl_env_final import PCBRLEnv, evaluate_sequence, random_greedy_search

pcb_json = "pcb_pre_jsons/pcb_cells_1.json"
gamma    = 1.0

# 1. 计算 baseline reward —— 全部动作都设为 (0,0,0)
env = PCBRLEnv(pcb_json, gamma)
env.reset()
no_op_seq = [(0,0,0)] * len(env.device_order)
baseline_r = evaluate_sequence(pcb_json, no_op_seq, gamma)
print(f"Baseline (no-op) reward: {baseline_r:.6f}")

# 2. 随机+贪婪搜索得到的最佳 reward
best_seq, best_r, imp_seq, imp_r = random_greedy_search(
    pcb_json, n_rollouts=20, rel_range=128, step=32, gamma=gamma
)
print(f"Optimized reward: {imp_r:.6f}")

# 3. 对比
print(f"提升: {imp_r - baseline_r:.6f} ({(imp_r/baseline_r-1)*100:.1f}% increase)")
