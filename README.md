📁 Структура проекта
text


# Constrained Online Convex Optimization: PFS vs Baselines

Воспроизводимый экспериментальный стенд для сравнения алгоритмов онлайн-оптимизации
с ограничениями (constrained OCO).

## Алгоритмы

1. **PFS** — Online Gradient Descent + Polyak Feasibility Steps
2. **DPP** — Drift-Plus-Penalty (Yu et al., 2017)
3. **DPP-T** — DPP с tightened constraint
4. **POGD** — Projected Online Gradient Descent

## Установка


pip install -r requirements.txt
Запуск экспериментов
Benchmark A: Toy Quadratic (d=2)
Bash

cd src
python -m oco.run_experiment --config ../configs/toy.yaml
Benchmark B: Online Logistic Regression (d=20)
Bash

cd src
python -m oco.run_experiment --config ../configs/logreg.yaml
Построение графиков из сохранённых данных
Bash

python -m oco.plot_results --input ../results/toy/<run_id>/
python -m oco.plot_results --input ../results/logreg/<run_id>/
Результаты
После запуска создаётся папка results/<benchmark>/<timestamp>/ содержащая:

metrics_agg.csv — агрегированные метрики
metrics_step.csv — пошаговые метрики
config_resolved.yaml — использованные параметры
*.png — графики
Структура проекта
configs/ — YAML-конфигурации экспериментов
src/oco/problems/ — определения задач
src/oco/algorithms/ — реализации алгоритмов
src/oco/utils/ — вспомогательные функции
results/ — результаты экспериментов