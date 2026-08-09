#禁忌标记管理与掩码生成

import numpy as np
from collections import defaultdict

class TabooManager:
    """管理禁忌标记集合与掩码生成，实现时间衰减布隆过滤器效果"""
    def __init__(self, d: int, window: int = 20):
        self.d = d
        self.window = window
        self.fingerprint_age = {}  # fingerprint -> last_seen_round
        self.current_round = 0
    
    def add_fingerprint(self, fp: int, round_num: int = None):
        if round_num is None:
            round_num = self.current_round
        self.fingerprint_age[fp] = round_num
    
    def step_round(self):
        self.current_round += 1
        # 删除超过窗口期的指纹
        to_delete = [fp for fp, age in self.fingerprint_age.items() 
                     if self.current_round - age > self.window]
        for fp in to_delete:
            del self.fingerprint_age[fp]
    
    def generate_mask(self) -> np.ndarray:
        """根据当前有效指纹生成 d 维二进制掩码 (1 表示可用)"""
        mask = np.ones(self.d, dtype=bool)
        # 每个指纹映射到 k 个维度位置
        k = 1  # 每个指纹屏蔽 1 个维度
        for fp in self.fingerprint_age.keys():
            rng = np.random.default_rng(fp)
            indices = rng.integers(0, self.d, size=k)
            mask[indices] = False  # 将这些维度设为不可用
        return mask.astype(int)
    
    @property
    def active_fingerprints(self):
        return len(self.fingerprint_age)