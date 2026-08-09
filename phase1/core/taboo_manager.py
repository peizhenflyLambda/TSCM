"""禁忌标记管理器（时间衰减窗口）。"""
import numpy as np

class TabooManager:
    def __init__(self, d, window=20, k=1):
        self.d = d
        self.window = window
        self.k = k
        self.fingerprint_age = {}
        self.current_round = 0

    def add_fingerprint(self, fp, round_num=None):
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

    def generate_mask(self):
        """根据当前有效指纹生成 d 维二进制掩码（1 表示可用）。"""
        mask = np.ones(self.d, dtype=bool)
        for fp in self.fingerprint_age.keys():
            rng = np.random.default_rng(fp)
            mask[rng.integers(0, self.d, size=self.k)] = False
        return mask.astype(int)