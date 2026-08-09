"""统计分析：Mann‑Kendall 趋势检验、效应量、绘图。"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

def load_data():
    import os
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    path = os.path.join(data_dir, 'raw_results.csv')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]

def main():
    exp, ctrl, active = load_data()
    n_rounds = len(exp)
    half = n_rounds // 2

    first_exp, second_exp = np.mean(exp[:half]), np.mean(exp[half:])
    first_ctrl, second_ctrl = np.mean(ctrl[:half]), np.mean(ctrl[half:])
    tau, p = kendalltau(np.arange(n_rounds), exp)

    print("=" * 60)
    print("TSCM Phase 1 实验结果 (1000轮 × 500序列)")
    print(f"对照组: {first_ctrl:.4f} → {second_ctrl:.4f} ({(second_ctrl-first_ctrl)/first_ctrl*100:+.1f}%)")
    print(f"实验组: {first_exp:.4f} → {second_exp:.4f} ({(second_exp-first_exp)/first_exp*100:+.1f}%)")
    print(f"Mann-Kendall: tau={tau:.4f}, p={p:.4f}")
    print(f"有效维度: {np.mean(active[:half]):.1f} → {np.mean(active[half:]):.1f}")
    print("=" * 60)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    window = 100
    exp_smooth = np.convolve(exp, np.ones(window)/window, mode='valid')
    ctrl_smooth = np.convolve(ctrl, np.ones(window)/window, mode='valid')
    x_smooth = np.arange(len(exp_smooth)) + window // 2
    plt.plot(x_smooth, exp_smooth, 'b', label='With Taboo')
    plt.plot(x_smooth, ctrl_smooth, 'gray', label='Without Taboo')
    plt.legend(); plt.ylabel(f'Crash Rate ({window}-round avg)')
    plt.subplot(2, 1, 2)
    plt.plot(active, 'r'); plt.ylabel('Active Dimensions')
    plt.xlabel('Round')
    plt.tight_layout()
    plt.savefig('result.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    main()