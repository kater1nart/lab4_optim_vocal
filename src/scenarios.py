"""Стохастические сценарии искажения временного ряда звука на тесте (модели обучены на чистых данных).

Ряд: 20 шагов по 0,5 с × 20 признаков (0 rms_db, 1 centroid, 2 bandwidth, 3 rolloff, 4 flatness, 5 zcr, 6 f0, 7-19 MFCC1-13).
Работаем в нормированном пространстве признаков (z-score по train), кроме GBM, который действует на громкость в дБ.

Допущения (явно):
- OU: аддитивная коррелированная во времени помеха (фон, шум записи), стационарная, независимая по признакам;
  e_t = e_{t-1} - θ e_{t-1} Δ + σ √Δ ξ_t,  θ = 0.5, σ = 0.5 (в единицах std признака), Δ = 1 шаг.
- GBM: мультипликативный дрейф усиления (автоматическая регулировка громкости, разные мастеринги);
  G_t = exp((μ - s²/2) t + s W_t), μ = 0, s = 0.15 на шаг; влияет только на rms_db (+20·lg G_t);
  спектральные признаки и MFCC1-13 считаются инвариантными к усилению.
- Markov-switching: два режима на каждом шаге — «чистый» и «деградированный» (клиппинг/потери кодека);
  P(чистый→чистый)=0.9, P(дегр.→дегр.)=0.8 (стационарная доля деградации 1/3); в деградированном режиме
  добавляется независимый гауссов шум σ = 1.0 ко всем признакам.
"""
import numpy as np
import torch

RMS_IDX = 0


def ou(X, g, theta=0.5, sigma=0.5):
    N, T, D = X.shape
    e = torch.zeros(N, D)
    out = X.clone()
    for t in range(T):
        e = e - theta * e + sigma * torch.randn(N, D, generator=g)
        out[:, t] += e
    return out


def gbm(X, g, s=0.15, mu=0.0, norm=None):
    N, T, _ = X.shape
    W = torch.cumsum(torch.randn(N, T, generator=g), 1)
    t = torch.arange(1, T + 1).float()
    logG = (mu - s ** 2 / 2) * t + s * W
    out = X.clone()
    sd_rms = norm[1][RMS_IDX] if norm is not None else 1.0
    out[:, :, RMS_IDX] += 20 * logG / np.log(10) / sd_rms   # сдвиг в дБ -> в нормированные единицы
    return out


def markov(X, g, p_cc=0.9, p_dd=0.8, sigma=1.0):
    N, T, D = X.shape
    state = (torch.rand(N, generator=g) < 1 / 3)
    out = X.clone()
    for t in range(T):
        u = torch.rand(N, generator=g)
        state = torch.where(state, u < p_dd, u >= p_cc)
        out[:, t] += state[:, None].float() * sigma * torch.randn(N, D, generator=g)
    return out


class Scn:
    """Picklable обёртка для передачи в процессы."""
    def __init__(self, kind, norm=None, **kw):
        self.kind, self.norm, self.kw = kind, norm, kw

    def __call__(self, X, g):
        if self.kind == "ou":
            return ou(X, g, **self.kw)
        if self.kind == "gbm":
            return gbm(X, g, norm=self.norm, **self.kw)
        return markov(X, g, **self.kw)
