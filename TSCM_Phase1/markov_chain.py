#状态转移模拟与崩溃检测
import numpy as np
from typing import Tuple, List

class MatrixFamily:
    """三类随机矩阵生成器，模拟不同崩溃类型"""
    @staticmethod
    def normal(d: int, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1/np.sqrt(d), (d, d))
    
    @staticmethod
    def sparse(d: int, sparsity: float = 0.7, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1/np.sqrt(d), (d, d))
        mask = rng.random((d, d)) > sparsity
        W *= mask
        return W
    
    @staticmethod
    def low_rank(d: int, rank: int = 8, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        U = rng.normal(0, 1, (d, rank))
        V = rng.normal(0, 1, (rank, d))
        return (U @ V) / np.sqrt(rank)

def simple_pca(X: np.ndarray, n_components: int = 8) -> np.ndarray:
    """极简 PCA：直接对数据矩阵 X 进行中心化后 SVD 降维"""
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    # 使用 scipy 的 svd
    from scipy.linalg import svd
    U, s, Vt = svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components, :].T

class MarkovChainSimulator:
    """模拟带崩溃的状态转移"""
    def __init__(self, d: int = 32, max_steps: int = 20, threshold: float = 2.0):
        self.d = d
        self.max_steps = max_steps
        self.threshold = threshold
        # 预生成 1000 条固定的随机矩阵序列（每条长度 max_steps）
        self.W_sequences = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            seq = []
            # 随机选择矩阵类型
            fam = rng.choice([MatrixFamily.normal, MatrixFamily.sparse, MatrixFamily.low_rank])
            for _ in range(max_steps):
                W = fam(d, seed=rng.integers(0, 1e6))
                seq.append(W)
            self.W_sequences.append(seq)
    
    def run_sequence(self, seq_idx: int, mask: np.ndarray) -> Tuple[bool, List[np.ndarray]]:
        """执行一条序列，返回 (是否崩溃, 状态轨迹)"""
        seq = self.W_sequences[seq_idx % len(self.W_sequences)]
        h = np.random.default_rng().normal(0, 0.1, self.d)  # 初始状态
        trajectory = [h.copy()]
        for t, W in enumerate(seq):
            W_masked = W * mask[:, np.newaxis]  # 将掩码应用到行（输出维度）
            h = np.tanh(W_masked @ h)
            trajectory.append(h.copy())
            if np.linalg.norm(h) > self.threshold:
                return True, trajectory[:-1]  # 崩溃前轨迹（不含超阈值点）
        return False, trajectory[:-1]  # 未崩溃，返回全轨迹
    
    def extract_crash_fingerprint(self, trajectory: List[np.ndarray], n_components: int = 8) -> int:
        """从崩溃前轨迹提取整数指纹"""
        # 取最后3步，拼接成 3*d 维向量
        steps = min(3, len(trajectory))
        x = np.concatenate(trajectory[-steps:])
        # 如果轨迹数量不足，用零填充
        if steps < 3:
            x = np.pad(x, (0, (3-steps)*self.d))
        # PCA 降维到 n_components（需要多轨迹数据，这里用单样本绕开：直接使用随机投影）
        # 实际我们直接使用一组随机投影向量，避免需要拟合 PCA
        rng = np.random.default_rng(hash(tuple(np.round(x, 3))) % (2**31))
        proj = rng.normal(0, 1, (self.d * 3, n_components))
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        reduced = x @ proj
        # 量化：取每个分量的符号和大小级别组合成整数指纹
        bits = []
        for i in range(n_components):
            val = reduced[i]
            # 符号位
            bits.append('1' if val > 0 else '0')
            # 是否大于 0.5 标准差（粗略幅度）
            bits.append('1' if abs(val) > 0.5 else '0')
        fingerprint = int(''.join(bits), 2)
        return fingerprint