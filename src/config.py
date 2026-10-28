import yaml, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
def P(key): return ROOT / CFG["paths"][key]
def models_dir():
    d = ROOT / CFG["paths"]["models"]; d.mkdir(exist_ok=True, parents=True); return d
def results_dir():
    d = ROOT / CFG["paths"]["results"]; d.mkdir(exist_ok=True, parents=True); return d
