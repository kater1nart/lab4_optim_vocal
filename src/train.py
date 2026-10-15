"""Контролируемый эксперимент ЛР4: фиксированная модель (BiGRU по временному ряду звука), меняется только оптимизатор.

Данные: MTAT (метки и split по исполнителям — ЛР1), ряд 20 шагов × 20 признаков (ЛР2, src/features.py).
Источники случайности управляются раздельно: init_seed (инициализация весов), order_seed (порядок mini-batch),
dropout_seed (маски dropout). По умолчанию все три равны seed.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
LAB1 = ROOT.parents[1] / "1" / "lab1_vocal_llm"
LAB2 = ROOT.parents[1] / "2" / "lab2_multimodal_vocal"
CLASSES = ["male", "female", "group", "instrumental"]

BUDGET = {"epochs": 30, "batch": 64, "scheduler": "cosine", "warmup_steps": 0, "target_val_f1": 0.60}
OPTS = {
    "SGD": lambda p, lr, wd: torch.optim.SGD(p, lr=lr, weight_decay=wd),
    "SGD+Momentum": lambda p, lr, wd: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd),
    "Nesterov": lambda p, lr, wd: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd),
    "AdaGrad": lambda p, lr, wd: torch.optim.Adagrad(p, lr=lr, weight_decay=wd),
    "RMSprop": lambda p, lr, wd: torch.optim.RMSprop(p, lr=lr, alpha=0.99, weight_decay=wd),
    "Adam": lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd),
    "AdamW": lambda p, lr, wd: torch.optim.AdamW(p, lr=lr, weight_decay=wd if wd > 0 else 1e-2),
    "RAdam": lambda p, lr, wd: torch.optim.RAdam(p, lr=lr, weight_decay=wd),
}

_DATA = {}


def load_data():
    if _DATA:
        return _DATA
    df = pd.read_csv(LAB1 / "data" / "processed" / "dataset_final.csv")
    z = np.load(LAB2 / "data" / "processed" / "mtat_ts.npz")
    seq = dict(zip(z["id"], z["seq"]))
    X = np.stack([seq[i] for i in df.clip_id]).astype(np.float32)
    y = df.label.map(CLASSES.index).values
    tr = (df.split == "train").values
    mu, sd = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-6       # нормировка только по train
    X = (X - mu) / sd
    for s in ["train", "val", "test"]:
        m = (df.split == s).values
        _DATA[s] = (torch.tensor(X[m]), torch.tensor(y[m]), df.artist.values[m])
    _DATA["norm"] = (mu, sd)
    return _DATA


class BiGRU(nn.Module):
    def __init__(self, d_in=20, h=64, p_drop=0.2):
        super().__init__()
        self.gru = nn.GRU(d_in, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Linear(4 * h, 4)  # [последнее скрытое состояние (2h), среднее по времени (2h)]

    def forward(self, x):
        out, hN = self.gru(x)
        z = torch.cat([hN[0], hN[1], out.mean(1)], -1)
        return self.head(self.drop(z))


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for lo in np.linspace(0, 0.9, bins):
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return e


@torch.no_grad()
def predict(model, X):
    model.eval()
    return F.softmax(model(X), -1).numpy()


def grad_noise(model, X, y, w, n_batches=16, batch=64, g=None):
    """Оценка шума градиента: E||g_b − G||² / ||G||² (≈ gradient noise scale B_simple / batch)."""
    model.train()
    def grad(ix):
        model.zero_grad()
        F.cross_entropy(model(X[ix]), y[ix], weight=w).backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
    G = grad(torch.arange(len(y)))
    dev = [((grad(torch.randint(0, len(y), (batch,), generator=g)) - G) ** 2).sum().item() for _ in range(n_batches)]
    return float(np.mean(dev) / (G ** 2).sum().item()), float(G.norm())


def run(opt="Adam", lr=1e-3, wd=0.0, seed=0, init_seed=None, order_seed=None, dropout_seed=None,
        epochs=None, batch=None, X_transform=None, log_curve=True, noise_probe=True, train_idx=None, scenarios=None):
    torch.set_num_threads(1)
    D = load_data()
    epochs, batch = epochs or BUDGET["epochs"], batch or BUDGET["batch"]
    init_seed = seed if init_seed is None else init_seed
    order_seed = seed if order_seed is None else order_seed
    dropout_seed = seed if dropout_seed is None else dropout_seed
    Xtr, ytr, _ = D["train"]
    if train_idx is not None:
        Xtr, ytr = Xtr[train_idx], ytr[train_idx]
    Xva, yva, _ = D["val"]
    torch.manual_seed(init_seed)
    model = BiGRU()
    o = OPTS[opt](model.parameters(), lr, wd)
    steps_per_epoch = int(np.ceil(len(ytr) / batch))
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=total)
    w = torch.tensor(len(ytr) / (4 * np.bincount(ytr.numpy(), minlength=4)), dtype=torch.float32)
    g_order = torch.Generator().manual_seed(10_000 + order_seed)
    torch.manual_seed(20_000 + dropout_seed)  # дальше глобальный RNG используется только dropout
    g_probe = torch.Generator().manual_seed(30_000 + seed)
    curve, noise, step, t_target = [], [], 0, None
    loss0 = F.cross_entropy(model(Xtr), ytr, weight=w).item()
    unstable, t0 = False, time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr), generator=g_order)
        ep_loss, gn = 0.0, []
        for b in range(0, len(ytr), batch):
            ix = perm[b:b + batch]
            loss = F.cross_entropy(model(Xtr[ix]), ytr[ix], weight=w)
            o.zero_grad()
            loss.backward()
            gn.append(torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item())
            o.step(); sched.step(); step += 1
            ep_loss += loss.item() * len(ix)
            if not np.isfinite(loss.item()):
                unstable = True
                break
        if unstable:
            break
        pv = predict(model, Xva)
        f1v = f1_score(yva.numpy(), pv.argmax(1), average="macro")
        if t_target is None and f1v >= BUDGET["target_val_f1"]:
            t_target = step
        if log_curve:
            curve.append({"epoch": ep + 1, "step": step, "train_loss": ep_loss / len(ytr), "val_f1": f1v,
                          "grad_norm_mean": float(np.mean(gn)), "lr": sched.get_last_lr()[0]})
        if noise_probe and ep + 1 in (1, epochs // 2, epochs):
            r, gnorm = grad_noise(model, Xtr, ytr, w, g=g_probe)
            noise.append({"epoch": ep + 1, "noise_ratio": r, "full_grad_norm": gnorm})
    train_s = time.perf_counter() - t0
    final_loss = F.cross_entropy(model(Xtr), ytr, weight=w).item() if not unstable else float("nan")
    unstable = unstable or not np.isfinite(final_loss) or final_loss > loss0
    res = {"opt": opt, "lr": lr, "wd": wd, "seed": seed, "init_seed": init_seed, "order_seed": order_seed,
           "dropout_seed": dropout_seed, "epochs": epochs, "batch": batch, "steps": step, "train_s": train_s,
           "unstable": unstable, "final_train_loss": final_loss, "steps_to_target": t_target,
           "opt_state_floats": sum(v.numel() for st in o.state.values() for v in st.values() if torch.is_tensor(v) and v.dim() > 0),
           "n_params": sum(p.numel() for p in model.parameters())}
    for s in ["val", "test"]:
        X, y, _ = D[s]
        if X_transform is not None and s == "test":
            X = X_transform(X)
        p = predict(model, X) if not unstable else np.full((len(y), 4), 0.25)
        res[f"{s}_f1"] = f1_score(y.numpy(), p.argmax(1), average="macro")
        res[f"{s}_acc"] = float((p.argmax(1) == y.numpy()).mean())
        res[f"{s}_ece"] = ece(p, y.numpy())
        if s == "test":
            res["_test_pred"] = p.argmax(1)
    if scenarios:
        Xte, yte, _ = D["test"]
        for name, fn in scenarios.items():
            vals = []
            for r in range(5):  # 5 реализаций случайного сценария
                Xs = fn(Xte, torch.Generator().manual_seed(40_000 + r))
                ps = predict(model, Xs) if not unstable else np.full((len(yte), 4), 0.25)
                vals.append(f1_score(yte.numpy(), ps.argmax(1), average="macro"))
            res[f"scn_{name}"] = float(np.mean(vals))
    res["_curve"], res["_noise"] = curve, noise
    res["_model"] = model
    return res


if __name__ == "__main__":
    for o in ["SGD", "Adam"]:
        r = run(o, 1e-2 if o == "SGD" else 1e-3, seed=0)
        print(o, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")})
