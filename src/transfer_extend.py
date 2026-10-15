"""То же правило, что в фазе A: если лучший lr на краю сетки переноса — добавить шаг ×1/3 (×3), 2 seed; при смене lr — новый финал на 5 seed."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer as TR


def main():
    f = pd.read_csv(TR.RES / "F_transfer.csv")
    t = f[f.stage == "tune"]
    jobs = []
    for o, g in t.groupby("opt"):
        m = g.groupby("lr").val_f1.mean(); lrs = sorted(m.index)
        if m.idxmax() == lrs[0]:
            jobs += [{"opt": o, "lr": lrs[0] / 3, "seed": s, "stage": "tune_ext"} for s in (100, 101)]
        elif m.idxmax() == lrs[-1]:
            jobs += [{"opt": o, "lr": lrs[-1] * 3, "seed": s, "stage": "tune_ext"} for s in (100, 101)]
    with ProcessPoolExecutor(10) as ex:
        ext = pd.DataFrame(list(ex.map(TR.fit, jobs)))
    allt = pd.concat([t, ext])
    best = allt.groupby(["opt", "lr"]).val_f1.mean().reset_index().sort_values("val_f1").groupby("opt").tail(1)
    old = f[f.stage == "final"].groupby("opt").lr.first()
    redo = [r for r in best.itertuples() if abs(old.get(r.opt, -1) - r.lr) > 1e-12]
    print(ext.groupby(["opt", "lr"]).val_f1.mean().round(3).to_string()); print("redo:", [(r.opt, r.lr) for r in redo])
    fin = [{"opt": r.opt, "lr": r.lr, "seed": s, "stage": "final"} for r in redo for s in range(5)]
    with ProcessPoolExecutor(10) as ex:
        new = pd.DataFrame(list(ex.map(TR.fit, fin))) if fin else pd.DataFrame()
    keep = f[~((f.stage == "final") & f.opt.isin([r.opt for r in redo]))]
    pd.concat([keep, ext, new]).to_csv(TR.RES / "F_transfer.csv", index=False)


if __name__ == "__main__":
    main()
