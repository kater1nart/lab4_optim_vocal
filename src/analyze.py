"""Анализ ЛР4: таблицы с ДИ, статистический протокол, графики, журнал нестабильных запусков."""
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RES, LOGS, FIG = ROOT / "results", ROOT / "results" / "logs", ROOT / "report" / "figs"
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
ORDER = ["SGD", "SGD+Momentum", "Nesterov", "AdaGrad", "RMSprop", "Adam", "AdamW", "RAdam"]


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p)
    run = 0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1, run)
    return adj


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    B = pd.read_csv(RES / "B_final.csv")
    best_lr = json.load(open(RES / "best_lr.json"))
    out = []

    # ---------- сводная таблица
    rows = []
    for o in ORDER:
        g = B[B.opt == o]
        r = {"opt": o, "lr": best_lr[o]}
        for c in ["val_f1", "test_f1", "test_ece"]:
            r[c], r[c + "_lo"], r[c + "_hi"] = ci(g[c])
        r["test_f1_std"] = g.test_f1.std(ddof=1)
        r["test_f1_min"] = g.test_f1.min()
        r["test_f1_q10"] = g.test_f1.quantile(0.1)
        r["steps_to_target_median"] = g.steps_to_target.median()
        r["reached_target"] = int(g.steps_to_target.notna().sum())
        r["train_s"] = g.train_s.mean()
        r["memory_kb"] = (g.n_params.iloc[0] + g.opt_state_floats.iloc[0]) * 4 / 1024
        r["unstable"] = int(g.unstable.sum())
        r["gap_val_test"] = (g.test_f1 - g.val_f1).mean()
        r["final_train_loss"] = g.final_train_loss.mean()
        rows.append(r)
    S = pd.DataFrame(rows)

    # ---------- сводный балл (по val, не по тесту): 40% val F1, 20% (−std), 15% (−ECE val), 15% (−шаги до цели), 10% (−память)
    V = B.groupby("opt").agg(val=("val_f1", "mean"), vstd=("val_f1", "std"), vece=("val_ece", "mean")).loc[ORDER]
    steps = S.set_index("opt").steps_to_target_median.fillna(B.steps.max())
    mem = S.set_index("opt").memory_kb
    nz = lambda x, hi=True: (x - x.min()) / (x.max() - x.min() + 1e-12) if hi else (x.max() - x) / (x.max() - x.min() + 1e-12)
    score = 0.40 * nz(V.val) + 0.20 * nz(V.vstd, False) + 0.15 * nz(V.vece, False) + 0.15 * nz(steps, False) + 0.10 * nz(mem, False)
    S["score"] = S.opt.map(score)
    S.to_csv(RES / "summary_B.csv", index=False)

    # ---------- статистический протокол (спаренность по seed: одинаковая инициализация и порядок батчей)
    P = B.pivot(index="seed", columns="opt", values="test_f1")[ORDER]
    fr = stats.friedmanchisquare(*[P[o] for o in ORDER])
    ranks = P.rank(axis=1, ascending=False).mean().sort_values()
    best_val = V.val.idxmax()
    pw = []
    for o in ORDER:
        if o == best_val:
            continue
        d = P[best_val] - P[o]
        w = stats.wilcoxon(d) if np.any(d != 0) else None
        pw.append({"vs": o, "mean_diff": d.mean(), "p": w.pvalue if w else 1.0, "P(best>o)": (d > 0).mean()})
    pw = pd.DataFrame(pw); pw["p_holm"] = holm(pw.p)
    pw.to_csv(RES / "pairwise_vs_best.csv", index=False)
    pd.Series(ranks).to_csv(RES / "mean_ranks.csv")

    # блочный bootstrap по исполнителям (тест) для лучшего и SGD: ДИ Macro-F1, усреднённый по 10 seed
    import train as T
    D = T.load_data()
    yte, art = D["test"][1].numpy(), D["test"][2]
    tp = np.load(LOGS / "B_final_testpred.npz")
    def preds(o):
        return [tp[k] for k in tp.files if k.startswith(o + "__")]
    rng = np.random.RandomState(0)
    ua = np.unique(art); idx = {a: np.where(art == a)[0] for a in ua}
    boot = {}
    for o in ORDER:
        ps = preds(o)
        vals = []
        for _ in range(500):
            s = np.concatenate([idx[a] for a in rng.choice(ua, len(ua))])
            vals.append(np.mean([f1_score(yte[s], p[s], average="macro") for p in ps]))
        boot[o] = np.percentile(vals, [2.5, 97.5])
    S["test_f1_blockboot_lo"] = S.opt.map(lambda o: boot[o][0]); S["test_f1_blockboot_hi"] = S.opt.map(lambda o: boot[o][1])
    S.to_csv(RES / "summary_B.csv", index=False)

    # ---------- чувствительность
    C = pd.read_csv(RES / "C_sensitivity.csv")
    C["factor"] = C.phase.str.replace("C_", "")
    C["value"] = np.select([C.factor == "lr", C.factor == "batch"],
                           [(C.lr / C.opt.map(best_lr)).round(3).astype(str) + "×lr*", C.batch.astype(str)], C.wd.astype(str))
    base = B[B.opt.isin(C.opt.unique()) & (B.seed < 5)].assign(factor="база", value="lr*, batch 64, wd 0")
    sens = pd.concat([base, C]).groupby(["opt", "factor", "value"]).agg(test=("test_f1", "mean"), test_std=("test_f1", "std"),
                                                                       val=("val_f1", "mean"), unstable=("unstable", "sum")).reset_index()
    sens.to_csv(RES / "sensitivity.csv", index=False)

    # ---------- разложение дисперсии
    Dv = pd.read_csv(RES / "D_variance.csv")
    var = Dv.groupby(["opt", "phase"]).test_f1.agg(["mean", "std"]).reset_index()
    allv = B[B.opt.isin(Dv.opt.unique())].groupby("opt").test_f1.agg(["mean", "std"]).reset_index().assign(phase="все источники (B)")
    var = pd.concat([var, allv])
    var.to_csv(RES / "variance_decomposition.csv", index=False)

    # ---------- сценарии
    scn = B.groupby("opt")[["test_f1", "scn_OU", "scn_GBM", "scn_Markov"]].mean().loc[ORDER]
    scn.to_csv(RES / "scenarios.csv")

    # ---------- шум градиента
    N = pd.read_csv(LOGS / "B_final_noise.csv")
    noise = N.groupby(["opt", "epoch"]).noise_ratio.agg(["mean", "std"]).reset_index()
    noise.to_csv(RES / "gradient_noise.csv", index=False)

    # ---------- журнал нестабильных запусков (все фазы)
    allruns = pd.concat([pd.read_csv(RES / f) for f in ["A_lr_tuning_all.csv", "B_final.csv", "C_sensitivity.csv", "D_variance.csv"]])
    uns = allruns[allruns.unstable == True]  # noqa: E712
    uns[["phase", "opt", "lr", "wd", "batch", "seed", "final_train_loss", "val_f1"]].to_csv(RES / "unstable_runs.csv", index=False)
    n_runs = len(allruns)

    # ---------- графики
    Cv = pd.read_csv(LOGS / "B_final_curves.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for o in ORDER:
        g = Cv[Cv.opt == o].groupby("step")
        axs[0].plot(g.train_loss.mean().index, g.train_loss.mean().values, label=o)
        m, s = g.val_f1.mean(), g.val_f1.std()
        axs[1].plot(m.index, m.values, label=o); axs[1].fill_between(m.index, m - s, m + s, alpha=0.08)
    axs[0].set_xlabel("шаг"); axs[0].set_ylabel("train loss (среднее по 10 seed)")
    axs[1].set_xlabel("шаг"); axs[1].set_ylabel("val Macro-F1 (среднее ± std)")
    axs[1].axhline(0.60, color="black", ls="--", lw=0.8)
    axs[1].set_ylim(0.3, 0.72); axs[0].legend(fontsize=8)
    for a in axs: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "convergence.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.boxplot([P[o] for o in ORDER], tick_labels=ORDER, showmeans=True)
    for i, o in enumerate(ORDER, 1):
        ax.scatter(np.full(len(P), i) + np.random.RandomState(i).uniform(-0.12, 0.12, len(P)), P[o], s=10, color="#4472C4", alpha=0.7)
    ax.set_ylabel("test Macro-F1 (10 seed)"); plt.xticks(rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(ranks.index[::-1], ranks.values[::-1], color="#A5A5A5")
    ax.set_xlabel("средний ранг по test Macro-F1 (1 — лучший), 10 seed")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "ranks.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for o in ORDER:
        g = noise[noise.opt == o]
        ax.plot(g.epoch, g["mean"], marker="o", label=o)
    ax.set_yscale("log"); ax.set_xlabel("эпоха"); ax.set_ylabel("E‖g_b − G‖² / ‖G‖² (batch 64)")
    ax.legend(fontsize=8, ncol=2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "gradient_noise.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 250)
    print(S[["opt", "lr", "val_f1", "test_f1", "test_f1_lo", "test_f1_hi", "test_f1_std", "test_f1_min", "test_ece",
             "steps_to_target_median", "reached_target", "train_s", "memory_kb", "unstable", "gap_val_test", "final_train_loss", "score",
             "test_f1_blockboot_lo", "test_f1_blockboot_hi"]].round(3).to_string(index=False))
    print("Friedman", fr)
    print(ranks.round(2).to_string())
    print("best by val:", best_val); print(pw.round(4).to_string(index=False))
    print(sens.round(3).to_string(index=False))
    print(var.round(4).to_string(index=False))
    print(scn.round(3).to_string())
    print(noise.round(3).to_string(index=False))
    print("unstable:", len(uns), "of", n_runs)
    print(uns[["phase", "opt", "lr", "batch", "wd", "seed", "final_train_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
