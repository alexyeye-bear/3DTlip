import os, re, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_SPLIT = re.compile(r"[,\s;|]+")

def _parse_idxs(cell):
    if cell is None: return []
    s = str(cell).strip()
    if not s or s.lower() == "nan": return []
    return [int(x) for x in _SPLIT.split(s) if x.isdigit()]

def _make_path(subdir, subject_id, run, t, suffix=".npy"):
    name = f"sub-{subject_id}_{run}_t{int(t):04d}{suffix}"
    return os.path.join(subdir, name)

class RandomSliceFromCSV(Dataset):
    """
    支持单列或多列条件：
    - condition_cols='a'
    - condition_cols=['b','c']
    每次 __getitem__ 会从 (cond × t) 里均匀随机抽一个 (cond, t)。
    """
    def __init__(self, csv_path: str, condition_cols, row_idx: int = None,
                 suffix: str = ".npy", seed: int = 0):
        df = pd.read_csv(csv_path)

        # 统一成列表
        if isinstance(condition_cols, str):
            condition_cols = [condition_cols]
        for c in condition_cols:
            if c not in df.columns:
                raise ValueError(f"条件列不存在: {c}")

        self.suffix = suffix
        self.rng = random.Random(seed)

        rows = [df.iloc[row_idx]] if row_idx is not None else [r for _, r in df.iterrows()]
        # 每项： (sid, sdir, run, pool) 其中 pool 是 [(cond, t), ...]
        self.items = []
        for r in rows:
            sid  = str(r["subject_id"]).strip()
            sdir = str(r["subdir"]).strip().rstrip("/")
            run  = str(r["run"]).strip()
            if not (sid and sdir and run):
                continue

            pool = []
            for cond in condition_cols:
                idxs = _parse_idxs(r.get(cond))
                pool.extend([(cond, t) for t in idxs])

            if pool:
                self.items.append((sid, sdir, run, pool))

        if not self.items:
            raise RuntimeError("没有可用行或所有条件下均无 t 索引。")

    def __len__(self):
        return len(self.items) * 100

    def __getitem__(self, i):
        base_i = i % len(self.items)
        sid, sdir, run, pool = self.items[base_i]
        cond, t = self.rng.choice(pool)     # 从 (cond × t) 池里随机抽一个
        p = _make_path(sdir, sid, run, t, self.suffix)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"未找到文件: {p}")
        vol = np.load(p)                    # 期望 (D,H,W)
        
        vol=vol[2:58,5:69,0:56]
        vol_t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0)
        meta = {"subject_id": sid, "run": run, "t": t, "condition": cond, "path": p}
        return vol_t, meta


# ----------------- 用法示例 -----------------
if __name__ == "__main__":
    csv_path = "/train_sub_slices_wide.csv"

    # 多个条件：每次从两个条件的所有 t 里随机抽一个 (cond, t)
    ds = RandomSliceFromCSV(csv_path,
                            condition_cols=["a","b"],
                            row_idx=None,            # None 表示对整张表，每行各采一个
                            suffix=".npy",           # 或 "_indice.npy"
                            seed=1234)

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    for x, meta in dl:
        print(x.shape, meta)   # x: [B, D, H, W]；meta 里带 condition 和 t
        break
