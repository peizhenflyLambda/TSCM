"""马尔可夫链状态转移与崩溃检测。"""
import numpy as np
from .matrix_families import MatrixFamily

class MarkovChainSimulator:
    def __init__(self, d=32, max_steps=20, threshold=2.0):
        self.d = d
        self.max_steps = max_steps
        self.threshold = threshold
        # 预生成1000条固定的随机矩阵序列（每条长度 max_steps）
        self.W_sequences = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            fam = rng.choice([MatrixFamily.normal, MatrixFamily.sparse, MatrixFamily.low_rank])
            seq = [fam(d, seed=rng.integers(0, 1e6)) for _ in range(max_steps)]
            self.W_sequences.append(seq)

    def run_sequence(self, seq_idx, mask):
        """执行一条序列，返回 (是否崩溃, 状态轨迹)。"""
        seq = self.W_sequences[seq_idx % len(self.W_sequences)]
        h = np.random.default_rng().normal(0, 0.1, self.d)
        trajectory = [h.copy()]
        for W in seq:
            W_masked = W * mask[:, np.newaxis]   # 掩码作用于行
            h = np.tanh(W_masked @ h)
            trajectory.append(h.copy())
            if np.linalg.norm(h) > self.threshold:
                return True, trajectory[:-1]
        return False, trajectory[:-1]

    def extract_crash_fingerprint(self, trajectory, n_components=8):
        """从崩溃前轨迹提取整数指纹。"""
        steps = min(3, len(trajectory))
        x = np.concatenate(trajectory[-steps:])
        if steps < 3:
            x = np.pad(x, (0, (3 - steps) * self.d))
        rng = np.random.default_rng(hash(tuple(np.round(x, 3))) % (2**31))
        proj = rng.normal(0, 1, (self.d * 3, n_components))
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        reduced = x @ proj
        bits = []
        for val in reduced:
            bits.append('1' if val > 0 else '0')
            bits.append('1' if abs(val) > 0.5 else '0')
        return int(''.join(bits), 2)