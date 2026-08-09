"""主实验：1000 轮 × 500 序列对照测试。"""
import numpy as np
from phase1.core.markov_chain import MarkovChainSimulator
from phase1.core.taboo_manager import TabooManager

def main():
    d = 32
    sim = MarkovChainSimulator(d=d)
    n_rounds = 1000
    n_sequences = 500

    taboo = TabooManager(d=d, window=20, k=1)
    crash_rates_exp = []
    active_dims = []
    crash_rates_ctrl = []

    # 预生成测试用例索引，保证两组完全相同
    test_indices = np.random.default_rng(1234).integers(0, 1000, size=(n_rounds, n_sequences))

    for r in range(n_rounds):
        mask = taboo.generate_mask()
        active_dims.append(np.sum(mask))

        crashes = 0
        for idx in test_indices[r]:
            crashed, traj = sim.run_sequence(idx, mask)
            if crashed:
                fp = sim.extract_crash_fingerprint(traj)
                taboo.add_fingerprint(fp, round_num=r)
                crashes += 1
        crash_rates_exp.append(crashes / n_sequences)
        taboo.step_round()

        crashes_ctrl = sum(sim.run_sequence(idx, np.ones(d, dtype=int))[0]
                          for idx in test_indices[r])
        crash_rates_ctrl.append(crashes_ctrl / n_sequences)

        if (r + 1) % 200 == 0:
            print(f"已完成 {r + 1}/{n_rounds} 轮...")

    # 保存原始数据到 phase1/data/ 目录
    import os
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'raw_results.csv')
    np.savetxt(out_path,
            np.column_stack([crash_rates_exp, crash_rates_ctrl, active_dims]),
            delimiter=',', header='crash_rate_exp,crash_rate_ctrl,active_dims', comments='')
    print(f"实验完成，数据已保存至 {out_path}")

if __name__ == '__main__':
    main()