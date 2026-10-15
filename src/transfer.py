"""Фаза F: перенос оптимизаторов в мультимодальную модель ЛР2 (GatedFusion: текст MLP + CLIP MLP + BiGRU по ряду).

Цикл обучения как в ЛР2 (до 80 эпох, early stopping по val Macro-F1, patience 12, batch 128, class weights),
меняется только оптимизатор. lr: сетка {1/3, 1, 3} × лучший lr из ЛР4, 2 seed, выбор по val; затем 5 seed.
Исходная настройка ЛР2 (AdamW, lr 1e-3, wd 1e-3) добавлена как референс.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
LAB2 = ROOT.parents[1] / "2" / "lab2_multimodal_vocal"
sys.path.insert(0, str(LAB2 / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RES = ROOT / "results"
_D = {}


def data():
    if _D:
        return _D
    import importlib.util
    spec = importlib.util.spec_from_file_location("lab2_experiment", LAB2 / "src" / "experiment.py")
    e2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2)
    df, seq, emb, _ = e2.load("mtat")
    sp = {s: (df.split == s).values for s in ["train", "val", "test"]}
    enc = e2.Encoders().fit(df[sp["train"]], seq[sp["train"]], emb[sp["train"]])
    F = {s: enc.transform(df[sp[s]], seq[sp[s]], emb[sp[s]]) for s in sp}
    sc = (F["train"]["ts_seq"].mean((0, 1)), F["train"]["ts_seq"].std((0, 1)) + 1e-6)
    for s in sp:
        _D[s] = (e2.tensors(F[s], sc), torch.tensor(df.y.values[sp[s]]))
    _D["cls"] = e2.GatedFusion
    return _D


def fit(kw):
    import train as T
    torch.set_num_threads(1)
    D = data()
    (Tt, yt), (Tv, yv), (Te, ye) = D["train"], D["val"], D["test"]
    torch.manual_seed(kw["seed"])
    m = D["cls"]()
    if kw["opt"] == "AdamW (ЛР2: 1e-3, wd 1e-3)":
        o = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    else:
        o = T.OPTS[kw["opt"]](m.parameters(), kw["lr"], 0.0)
    w = torch.tensor(len(yt) / (4 * np.bincount(yt.numpy(), minlength=4)), dtype=torch.float32)
    best, state, pat, t0, ep_best = -1, None, 0, time.perf_counter(), 0
    g = torch.Generator().manual_seed(1000 + kw["seed"])
    for ep in range(80):
        m.train()
        perm = torch.randperm(len(yt), generator=g)
        for b in range(0, len(yt), 128):
            ix = perm[b:b + 128]
            out, _ = m(Tt[0][ix], Tt[1][ix], Tt[2][ix])
            loss = nn.functional.cross_entropy(out, yt[ix], weight=w)
            o.zero_grad(); loss.backward(); o.step()
        m.eval()
        with torch.no_grad():
            f = f1_score(yv, m(*Tv)[0].argmax(1).numpy(), average="macro")
        if f > best:
            best, state, pat, ep_best = f, {k: v.clone() for k, v in m.state_dict().items()}, 0, ep + 1
        else:
            pat += 1
            if pat >= 12:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        ft = f1_score(ye, m(*Te)[0].argmax(1).numpy(), average="macro")
    return dict(kw, val_f1=best, test_f1=ft, best_epoch=ep_best, epochs_run=ep + 1, train_s=time.perf_counter() - t0)


def main(opts):
    best_lr = json.load(open(RES / "best_lr.json"))
    tune = [{"opt": o, "lr": best_lr[o] * mlt, "seed": s, "stage": "tune"} for o in opts for mlt in (1 / 3, 1, 3) for s in (100, 101)]
    with ProcessPoolExecutor(10) as ex:
        t = pd.DataFrame(list(ex.map(fit, tune)))
    chosen = t.groupby(["opt", "lr"]).val_f1.mean().reset_index().sort_values("val_f1").groupby("opt").tail(1)
    final = [{"opt": r.opt, "lr": r.lr, "seed": s, "stage": "final"} for r in chosen.itertuples() for s in range(5)]
    final += [{"opt": "AdamW (ЛР2: 1e-3, wd 1e-3)", "lr": 1e-3, "seed": s, "stage": "final"} for s in range(5)]
    with ProcessPoolExecutor(10) as ex:
        f = pd.DataFrame(list(ex.map(fit, final)))
    pd.concat([t, f]).to_csv(RES / "F_transfer.csv", index=False)
    print(f.groupby("opt")[["lr", "val_f1", "test_f1", "best_epoch", "train_s"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1].split(","))
