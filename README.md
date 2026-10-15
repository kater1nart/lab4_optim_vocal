# Лабораторная работа 4. Оптимизаторы и стохастика в задаче определения типа фронт-вокала

Тема ВКР: *Разработка системы автоматической разметки музыкального каталога по типу фронт-вокала для задач поиска и рекомендаций.*
Продолжение: [ЛР1](https://github.com/kater1nart/lab1_vocal_llm), [ЛР2](https://github.com/kater1nart/lab2_multimodal_vocal), [ЛР3](https://github.com/kater1nart/lab3_marl_vocal).

Фиксированная модель: BiGRU по временному ряду звука (20 шагов × 20 признаков, MTAT, split по исполнителям).
Бюджет: 30 эпох × 80 шагов, batch 64, cosine. Подбор lr: сетка ×{1/10…30}, 3 seed, правило расширения на краю. Финал: 10 seed.

| Оптимизатор | Test Macro-F1 [95% ДИ] | ECE | Шагов до val F1 ≥ 0.60 | Средний ранг |
|---|---|---|---|---|
| Nesterov | **0.682** [0.676, 0.688] | 0.230 | 640 | **2.4** |
| SGD+Momentum | 0.678 [0.670, 0.685] | 0.207 | 560 | 3.2 |
| SGD | 0.674 [0.671, 0.678] | **0.059** | 1000 | 3.8 |
| AdaGrad | 0.672 [0.668, 0.676] | 0.149 | **160** | 4.7 |
| RAdam | 0.671 [0.665, 0.676] | 0.195 | 240 | 4.8 |
| Adam | 0.669 [0.656, 0.681] | 0.216 | 200 | 4.9 |
| AdamW | 0.668 [0.655, 0.680] | 0.213 | 200 | 5.1 |
| RMSprop | 0.659 [0.651, 0.667] | 0.265 | 400 | 7.1 |

Friedman p = 0.0015; попарно с лидером — незначимо (Holm). Перенос в Gated-MLP из ЛР2: Adam 0.648 > AdamW(ЛР2) 0.641 > … > Nesterov 0.602.
Отчёт: `report/Отчет_ЛР4.docx`.

## Воспроизведение
```bash
uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt
# нужны соседние репозитории ЛР1 и ЛР2 (метки, split, временные ряды и признаки)
.venv/bin/python src/experiment.py A && .venv/bin/python src/extend_grid.py && .venv/bin/python src/experiment.py BCD
.venv/bin/python src/transfer.py "Nesterov,SGD+Momentum,RMSprop,Adam,AdamW,RAdam" && .venv/bin/python src/transfer_extend.py
.venv/bin/python src/analyze.py && .venv/bin/python src/build_report.py
```
