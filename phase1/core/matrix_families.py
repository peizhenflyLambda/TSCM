"""三类随机矩阵生成器，用于模拟不同崩溃类型。"""
import numpy as np

class MatrixFamily:
    @staticmethod
    def normal(d, seed=None):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1 / np.sqrt(d), (d, d))

    @staticmethod
    def sparse(d, sparsity=0.7, seed=None):
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1 / np.sqrt(d), (d, d))
        W[rng.random((d, d)) > sparsity] = 0
        return W

    @staticmethod
    def low_rank(d, rank=8, seed=None):
        rng = np.random.default_rng(seed)
        U = rng.normal(0, 1, (d, rank))
        V = rng.normal(0, 1, (rank, d))
        return (U @ V) / np.sqrt(rank)