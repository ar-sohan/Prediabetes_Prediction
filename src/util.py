from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
def pmap(fn, items, workers=8, desc=""):
    from .config import CFG
    workers = min(workers, int(CFG["llm"].get("max_workers", workers)))   # free tiers: use 1-2 workers
    with ThreadPoolExecutor(workers) as ex: return list(tqdm(ex.map(fn, items), total=len(items), desc=desc))
def mean_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    b = [r.choice(x, len(x)).mean() for _ in range(n_boot)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
