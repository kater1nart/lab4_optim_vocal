"""Фазы эксперимента ЛР4 (параллельно, 10 процессов, 1 поток на процесс).

A. Подбор lr: одинаковая процедура для всех — сетка {1/10,1/3,1,3,10,30} × базовый lr, 3 seed (100-102), критерий — средний val Macro-F1.
B. Финал: лучший lr, 10 seed (0-9); кривые, шум градиента, тест, стохастические сценарии.
C. Чувствительность: lr ×{1/3,3}, batch {32,128,256}, weight decay {1e-4,1e-3,1e-2}, 5 seed, для 3 оптимизаторов.
D. Разложение дисперсии: варьируется только init / только порядок батчей / только dropout (10 значений).
E. Случайность данных: bootstrap исполнителей обучающей выборки (10 повторов).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as S  # noqa: E402
import train as T  # noqa: E402

RES = T.ROOT / "results"
BASE_LR = {"SGD": 1e-2, "SGD+Momentum": 1e-2, "Nesterov": 1e-2, "AdaGrad": 1e-2, "RMSprop": 1e-3,
           "Adam": 1e-3, "AdamW": 1e-3, "RAdam": 1e-3}
MULT = [0.1, 1 / 3, 1, 3, 10, 30]
TUNE_SEEDS, FINAL_SEEDS = [100, 101, 102], list(range(10))


def job(kw):
    kw = dict(kw)
    phase = kw.pop("phase")
    if kw.pop("with_scenarios", False):
        norm = T.load_data()["norm"]
        kw["scenarios"] = {"OU": S.Scn("ou"), "GBM": S.Scn("gbm", norm=norm), "Markov": S.Scn("markov")}
    if "boot" in kw:
        b = kw.pop("boot")
        art = T.load_data()["train"][2]
        ua = np.unique(art)
        rng = np.random.RandomState(500 + b)
        pick = rng.choice(ua, len(ua))
        kw["train_idx"] = np.concatenate([np.where(art == a)[0] for a in pick])
        kw["boot"] = b
    boot = kw.pop("boot", None)
    r = T.run(**kw)
    curve, noise, pred = r.pop("_curve"), r.pop("_noise"), r.pop("_test_pred")
    r.pop("_model")
    r["phase"], r["boot"] = phase, boot
    return r, curve, noise, pred


def pmap(jobs):
    with ProcessPoolExecutor(10) as ex:
        return list(ex.map(job, jobs, chunksize=1))


def save(out, name):
    rows, curves, noises = [], [], []
    preds = {}
    for r, c, n, p in out:
        key = f"{r['opt']}|lr={r['lr']:.2e}|wd={r['wd']}|b={r['batch']}|s={r['seed']}|i={r['init_seed']}|o={r['order_seed']}|d={r['dropout_seed']}|boot={r['boot']}"
        rows.append(r)
        curves += [dict(x, key=key, opt=r["opt"], seed=r["seed"], phase=r["phase"]) for x in c]
        noises += [dict(x, key=key, opt=r["opt"], seed=r["seed"], phase=r["phase"]) for x in n]
        preds[key] = p
    pd.DataFrame(rows).to_csv(RES / f"{name}.csv", index=False)
    if curves:
        pd.DataFrame(curves).to_csv(RES / "logs" / f"{name}_curves.csv", index=False)
    if noises:
        pd.DataFrame(noises).to_csv(RES / "logs" / f"{name}_noise.csv", index=False)
    np.savez_compressed(RES / "logs" / f"{name}_testpred.npz", **{k.replace("|", "__"): v for k, v in preds.items()})
    return pd.DataFrame(rows)


def main(phases):
    RES.mkdir(exist_ok=True); (RES / "logs").mkdir(exist_ok=True)
    if "A" in phases:
        jobs = [{"phase": "A", "opt": o, "lr": BASE_LR[o] * m, "seed": s, "noise_probe": False}
                for o in T.OPTS for m in MULT for s in TUNE_SEEDS]
        a = save(pmap(jobs), "A_lr_tuning")
        best = a.groupby(["opt", "lr"]).val_f1.mean().reset_index().sort_values("val_f1").groupby("opt").tail(1)
        json.dump(dict(zip(best.opt, best.lr)), open(RES / "best_lr.json", "w"), indent=2)
        print(best.to_string(index=False), flush=True)
    best_lr = json.load(open(RES / "best_lr.json"))
    if "B" in phases:
        jobs = [{"phase": "B", "opt": o, "lr": best_lr[o], "seed": s, "with_scenarios": True} for o in T.OPTS for s in FINAL_SEEDS]
        b = save(pmap(jobs), "B_final")
        print(b.groupby("opt")[["val_f1", "test_f1", "test_ece", "train_s", "unstable"]].mean().round(3).to_string(), flush=True)
    b = pd.read_csv(RES / "B_final.csv")
    rank = b.groupby("opt").val_f1.mean().sort_values(ascending=False)
    adaptive = [o for o in rank.index if o in ("AdaGrad", "RMSprop", "Adam", "AdamW", "RAdam")]
    sgdfam = [o for o in rank.index if o in ("SGD", "SGD+Momentum", "Nesterov")]
    sens_opts = list(dict.fromkeys([adaptive[0], adaptive[1], sgdfam[0]]))
    if "C" in phases:
        jobs = []
        for o in sens_opts:
            for s in range(5):
                for m in [1 / 3, 3]:
                    jobs.append({"phase": "C_lr", "opt": o, "lr": best_lr[o] * m, "seed": s, "noise_probe": False, "log_curve": False})
                for bs in [32, 128, 256]:
                    jobs.append({"phase": "C_batch", "opt": o, "lr": best_lr[o], "batch": bs, "seed": s, "noise_probe": False, "log_curve": False})
                for wd in [1e-4, 1e-3, 1e-2]:
                    jobs.append({"phase": "C_wd", "opt": o, "lr": best_lr[o], "wd": wd, "seed": s, "noise_probe": False, "log_curve": False})
        save(pmap(jobs), "C_sensitivity")
        print("C done", flush=True)
    if "D" in phases:
        jobs = []
        for o in [adaptive[0], sgdfam[0]]:
            for v in range(10):
                jobs.append({"phase": "D_init", "opt": o, "lr": best_lr[o], "seed": 0, "init_seed": v, "order_seed": 0, "dropout_seed": 0, "noise_probe": False, "log_curve": False})
                jobs.append({"phase": "D_order", "opt": o, "lr": best_lr[o], "seed": 0, "init_seed": 0, "order_seed": v, "dropout_seed": 0, "noise_probe": False, "log_curve": False})
                jobs.append({"phase": "D_dropout", "opt": o, "lr": best_lr[o], "seed": 0, "init_seed": 0, "order_seed": 0, "dropout_seed": v, "noise_probe": False, "log_curve": False})
                jobs.append({"phase": "E_data", "opt": o, "lr": best_lr[o], "seed": 0, "boot": v, "noise_probe": False, "log_curve": False})
        save(pmap(jobs), "D_variance")
        print("D done", flush=True)
    json.dump({"budget": T.BUDGET, "base_lr": BASE_LR, "lr_multipliers": MULT, "tune_seeds": TUNE_SEEDS,
               "final_seeds": FINAL_SEEDS, "sensitivity_optimizers": sens_opts, "model": "BiGRU(20->64x2) + dropout 0.2 + Linear(256->4)"},
              open(RES / "protocol.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ABCD")
