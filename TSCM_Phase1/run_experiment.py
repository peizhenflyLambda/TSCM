import numpy as np
import matplotlib.pyplot as plt
from markov_chain import MarkovChainSimulator
from taboo_manager import TabooManager
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')

# 初始化
d = 32
simulator = MarkovChainSimulator(d=d)
n_rounds = 1000          # 扩大至1000轮
n_sequences = 500        # 每轮500条序列

# 实验组（带禁忌）
taboo = TabooManager(d=d, window=20)   # 最优参数
crash_rates_exp = []
active_dims_exp = []

# 对照组（无禁忌）
crash_rates_ctrl = []

# 预生成测试用例（两组共享）
test_indices = np.random.default_rng(1234).integers(0, 1000, size=(n_rounds, n_sequences))

for r in range(n_rounds):
    mask = taboo.generate_mask()
    active_dims_exp.append(np.sum(mask))
    
    crashes = 0
    for seq_idx in test_indices[r]:
        crashed, traj = simulator.run_sequence(seq_idx, mask)
        if crashed:
            fp = simulator.extract_crash_fingerprint(traj)
            taboo.add_fingerprint(fp, round_num=r)
            crashes += 1
    crash_rates_exp.append(crashes / n_sequences)
    taboo.step_round()
    
    crashes_ctrl = 0
    for seq_idx in test_indices[r]:
        crashed_ctrl, _ = simulator.run_sequence(seq_idx, np.ones(d, dtype=int))
        if crashed_ctrl:
            crashes_ctrl += 1
    crash_rates_ctrl.append(crashes_ctrl / n_sequences)
    
    if (r+1) % 100 == 0:
        print(f"已完成 {r+1}/{n_rounds} 轮...")

# 分析
half = n_rounds // 2
first_half_exp = np.mean(crash_rates_exp[:half])
second_half_exp = np.mean(crash_rates_exp[half:])
change = (second_half_exp - first_half_exp) / first_half_exp * 100

first_half_ctrl = np.mean(crash_rates_ctrl[:half])
second_half_ctrl = np.mean(crash_rates_ctrl[half:])
change_ctrl = (second_half_ctrl - first_half_ctrl) / first_half_ctrl * 100

tau, p_mk = kendalltau(np.arange(n_rounds), crash_rates_exp)

dim_first = np.mean(active_dims_exp[:half])
dim_second = np.mean(active_dims_exp[half:])

print("=" * 60)
print("TSCM Phase 1 实验结果 (扩大规模: 1000轮 x 500序列)")
print("=" * 60)
print(f"对照组前500轮平均崩溃率: {first_half_ctrl:.4f}")
print(f"对照组后500轮平均崩溃率: {second_half_ctrl:.4f}")
print(f"对照组变化: {change_ctrl:+.1f}%")
print()
print(f"实验组前500轮平均崩溃率: {first_half_exp:.4f}")
print(f"实验组后500轮平均崩溃率: {second_half_exp:.4f}")
print(f"实验组变化: {change:+.1f}%")
print()
print(f"实验组崩溃率 Mann-Kendall 趋势检验: tau={tau:.4f}, p={p_mk:.4f}")
print(f"有效维度: 前500轮平均 {dim_first:.1f} → 后500轮平均 {dim_second:.1f}")
print()
if change < -5 and p_mk < 0.05:
    print("✅ 假说得到支持：崩溃率显著下降，禁忌标记有效")
elif change < -5:
    print("⚠️ 崩溃率下降但统计不显著：效应较小或噪声较大")
elif abs(change) <= 5:
    print("❓ 崩溃率无实质变化：禁忌标记无明显效果")
else:
    print("❌ 崩溃率上升：禁忌标记可能有害")
print("=" * 60)

# 生成图表
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
window = 100
exp_smooth = np.convolve(crash_rates_exp, np.ones(window)/window, mode='valid')
ctrl_smooth = np.convolve(crash_rates_ctrl, np.ones(window)/window, mode='valid')
x_smooth = np.arange(len(exp_smooth)) + window//2
plt.plot(x_smooth, exp_smooth, 'b-', linewidth=2, label='With Taboo')
plt.plot(x_smooth, ctrl_smooth, 'gray', linewidth=2, label='Without Taboo')
plt.xlabel('Round')
plt.ylabel(f'Crash Rate ({window}-round avg)')
plt.title('Crash Rate Comparison (1000 Rounds)')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(active_dims_exp, 'r-', linewidth=2)
plt.xlabel('Round')
plt.ylabel('Active Dimensions')
plt.title('State Space Contraction (d=32)')
plt.ylim(0, 32)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tscm_phase1_1000rounds.png', dpi=150)
plt.show()