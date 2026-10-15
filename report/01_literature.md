## 1. Исследовательский обзор

**Тема ВКР:** разработка системы автоматической разметки музыкального каталога по типу фронт-вокала для задач поиска и рекомендаций.

**Постановка ЛР4.** Контролируемый эксперимент: архитектура классификатора типа вокала фиксирована (BiGRU по временному ряду звука из ЛР2), данные и бюджет тоже фиксированы, меняется только оптимизатор с одинаково настроенным learning rate. Вопросы:
1. какой оптимизатор даёт лучшее воспроизводимое качество с учётом скорости, стабильности и ресурсов;
2. насколько выводы ВКР зависят от случайности — инициализации, порядка батчей, dropout, выборки данных и «режима» входного ряда;
3. сохраняется ли выигрыш при переносе в мультимодальную модель из ЛР2.

**Поисковый запрос** (Google Scholar, arXiv, JMLR, ICLR / ICML / NeurIPS):
`("stochastic gradient" OR "adaptive optimizer" OR Adam) AND (generalization OR "high-probability" OR "Markovian noise" OR "heavy-tailed" OR benchmark)`

### Аннотированный обзор

| № | Источник | Группа | Суть | Связь с экспериментом |
|---|---|---|---|---|
| 1 | Robbins H., Monro S. *A Stochastic Approximation Method.* Ann. Math. Statist., 1951 | основы | стохастическая аппроксимация; условия Σηₜ = ∞, Σηₜ² < ∞ | SGD и затухающий шаг (у нас cosine schedule) |
| 2 | Sutskever I. et al. *On the importance of initialization and momentum in deep learning.* ICML 2013 | momentum | импульс и метод Нестерова в глубоком обучении; роль инициализации | SGD+Momentum, Nesterov; источник «инициализация» в карте случайности |
| 3 | Duchi J., Hazan E., Singer Y. *Adaptive Subgradient Methods (AdaGrad).* JMLR 2011 | адаптивные | покоординатный шаг по накопленной сумме квадратов градиентов | AdaGrad; ожидаемо затухающий шаг на длинном обучении |
| 4 | Kingma D., Ba J. *Adam.* ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) | адаптивные | оценки первого и второго моментов; RMSprop — частный случай без импульса | Adam, RMSprop |
| 5 | Loshchilov I., Hutter F. *Decoupled Weight Decay Regularization (AdamW).* ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | регуляризация | weight decay отдельно от адаптивного шага | AdamW; sensitivity по weight decay |
| 6 | Liu L. et al. *On the Variance of the Adaptive Learning Rate and Beyond (RAdam).* ICLR 2020. [arXiv:1908.03265](https://arxiv.org/abs/1908.03265) | адаптивные | высокая дисперсия адаптивного шага на первых итерациях, «выпрямление» | RAdam без warmup |
| 7 | Zhuang J. et al. *AdaBelief.* NeurIPS 2020. [arXiv:2010.07468](https://arxiv.org/abs/2010.07468); Chen X. et al. *Symbolic Discovery of Optimization Algorithms (Lion).* NeurIPS 2023. [arXiv:2302.06675](https://arxiv.org/abs/2302.06675) | новые правила | шаг по «вере» в градиент; правило, найденное поиском | не входят в набор 8 (нет в torch.optim) — ограничение |
| 8 | Wilson A. et al. *The Marginal Value of Adaptive Gradient Methods in Machine Learning.* NeurIPS 2017. [arXiv:1705.08292](https://arxiv.org/abs/1705.08292) | обобщение | адаптивные методы быстрее снижают train loss, но могут обобщать хуже SGD | гипотеза H2: разделять сходимость и обобщение |
| 9 | Keskar N. et al. *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.* ICLR 2017. [arXiv:1609.04836](https://arxiv.org/abs/1609.04836); Smith S., Le Q. *A Bayesian Perspective on Generalization and SGD.* ICLR 2018. [arXiv:1710.06451](https://arxiv.org/abs/1710.06451) | шум SGD и обобщение | большой батч → меньше шума → острые минимумы; оптимальная «температура» шума ∝ lr / batch | sensitivity по batch size и lr |
| 10 | McCandlish S. et al. *An Empirical Model of Large-Batch Training.* 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162) | шум градиента | gradient noise scale B = tr(Σ) / ‖G‖² определяет полезный размер батча | измеряем отношение шума ‖g_b − G‖² / ‖G‖² |
| 11 | Hardt M., Recht B., Singer Y. *Train faster, generalize better: Stability of SGD.* ICML 2016. [arXiv:1509.01240](https://arxiv.org/abs/1509.01240) | обобщение, теория | алгоритмическая устойчивость SGD ограничивает ошибку обобщения | разделение train loss и тестового качества |
| 12 | Gorbunov E., Danilova M., Gasnikov A. *Stochastic Optimization with Heavy-Tailed Noise via Accelerated Gradient Clipping.* NeurIPS 2020. [arXiv:2005.10785](https://arxiv.org/abs/2005.10785) — **российская школа (МФТИ)** | высоковероятностные оценки | оценки сложности с высокой вероятностью при тяжёлых хвостах шума; клиппинг градиента | высоковероятностный взгляд → смотрим на хвосты распределения метрик по seed, а не только на среднее |
| 13 | Beznosikov A. et al. *First Order Methods with Markovian Noise: from Acceleration to Variational Inequalities.* NeurIPS 2023. [arXiv:2305.15938](https://arxiv.org/abs/2305.15938) — **российская школа (МФТИ)** | марковский шум | сходимость при зависимых (марковских) выборках | сценарий Markov-switching; зависимость клипов одного исполнителя → блочный bootstrap |
| 14 | Schmidt R., Schneider F., Hennig P. *Descending through a Crowded Valley — Benchmarking Deep Learning Optimizers.* ICML 2021. [arXiv:2007.01547](https://arxiv.org/abs/2007.01547) | бенчмарки | ~15 оптимизаторов × 8 задач: нет однозначно лучшего, важен бюджет подбора lr | одинаковая процедура подбора lr для всех |
| 15 | Bouthillier X. et al. *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098) | воспроизводимость | дисперсия от инициализации, порядка данных и выборки сравнима с эффектами методов | разложение дисперсии по источникам случайности |

### Сравнительная матрица

| Работа | Метод / объект | Шум и данные (предпосылки) | Задачи | Метрики | Вывод для нас | Код |
|---|---|---|---|---|---|---|
| Robbins–Monro | SA / SGD | несмещённый градиент, ограниченная дисперсия, i.i.d. | теория | сходимость | затухающий шаг | — |
| AdaGrad | адаптивный шаг | выпуклость, разреженные градиенты | онлайн-обучение | regret | шаг падает со временем | torch |
| Adam / AdamW / RAdam | моменты градиента | ограниченные градиенты | CV, NLP | loss, accuracy | быстрый спуск, чувствительны к weight decay / warmup | torch |
| Wilson et al. | адаптивные vs SGD | переобучаемые задачи | CIFAR, PTB | test error | адаптивность ≠ обобщение | есть |
| Keskar; Smith & Le | шум SGD | i.i.d. mini-batch | CV | gap train/test | lr / batch как «температура» | есть |
| McCandlish et al. | gradient noise scale | i.i.d. | CV, RL, LM | B_noise | метрика шума | — |
| Gorbunov et al. (РФ) | клиппинг, ускорение | тяжёлые хвосты | выпуклые / нейросети | high-prob. bounds | смотреть хвосты, а не среднее | есть |
| Beznosikov et al. (РФ) | ускорение, VI | марковская зависимость | теория + эксперименты | сходимость | i.i.d.-допущение нарушается | — |
| Schmidt et al. | бенчмарк 15 оптимизаторов | разные задачи | DeepOBS | test acc, бюджет | процедура подбора lr решает | есть |
| Bouthillier et al. | дисперсия бенчмарков | разные источники | CV, NLP | std, вероятность превосходства | 5+ seed и разложение дисперсии | есть |
| **Эта работа** | **8 оптимизаторов torch.optim** | **mini-batch, init, dropout, порядок, данные (по исполнителям), сценарии OU / GBM / Markov** | **тип вокала по ряду звука** | **Macro-F1, шаги до цели, ECE, время, память, нестабильность** | | **да** |

### Исследовательский разрыв

Бенчмарки оптимизаторов (14) построены на CV и NLP с i.i.d. данными. Теоретические гарантии (1, 3, 12, 13) опираются на допущения о шуме: ограниченная дисперсия, i.i.d. или марковость. Эти допущения не проверяются на малых аудио-задачах с зависимыми объектами (клипы одного исполнителя) и со сдвигами входного ряда. Для задачи типа вокала нет сравнения оптимизаторов, которое одновременно:
- разделяет скорость сходимости, валидационное и тестовое качество;
- раскладывает дисперсию по источникам случайности;
- проверяет перенос выигрыша в мультимодальную модель.

### Гипотезы

- **H1.** При одинаковой процедуре подбора lr адаптивные методы (Adam, AdamW, RAdam) достигают целевого качества на val быстрее, чем SGD-семейство.
- **H2.** Разница итогового тестового Macro-F1 между лучшими методами меньше разброса по seed (вывод «X лучше Y» без интервалов ненадёжен).
- **H3.** Вклад источников случайности сопоставим с разницей между оптимизаторами; наибольший вклад даёт выборка данных.
- **H4.** Выигрыш лучшего оптимизатора сохраняется в мультимодальной модели ЛР2.
