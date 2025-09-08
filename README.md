# text-autocomplete  
Нейросеть для автодополнения текстов на основе RNN/LSTM и DistilGPT2

---

## 📦 Установка окружения

```bash
python -m venv .venv
.venv\Scripts\activate
```


```bash
pip install -r requirements.txt
```

Либо можно использовать фиксированный список:

```bash
pip install -r requirements.txt
```

---

## Быстрый старт

1. Скачайте датасет с короткими постами (например, `sentiment140`) и сохраните его как `data/raw_dataset.csv`.
2. Отредактируйте параметры в `configs/default.yaml`, если нужно.
3. Запустите `solution.ipynb` он уже содержит пошаговый пайплайн.

---

## Основные скрипты

- `src/lstm_train.py` обучение LSTM-модели (использует `configs/default.yaml`).
- `src/eval_lstm.py`  оценка качества модели с помощью ROUGE-1 / ROUGE-2.
- `src/eval_transformer_pipeline.py`  генерация и оценка с DistilGPT2 (через `transformers.pipeline`).
- `src/metrics.py`  вычисление метрик ROUGE (используется `evaluate`).

---

## Что под капотом

- **Предобработка данных**: lowercase → удаление ссылок, эмодзи, упоминаний → токенизация → генерация пар `X->Y` (со сдвигом на 1) → деление на train/val/test (80/10/10).
- **Модель:** LSTM с двумя режимами — `forward` (обучение) и генерация по одному токену до `<eos>` или достижения лимита.
- **Метрики:** ROUGE-1 и ROUGE-2.
- **Бейзлайн:** DistilGPT2 как референс по качеству и скорости.

---

## Как запускать

В PowerShell (Windows):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_sprint_2_project.txt
pip install ipywidgets
```

Быстрый прогон на срезе из 1000 строк:

```bash
$env:MAX_ROWS="1000"
python -m src.prepare_data
python -m src.lstm_train
python -m src.eval_lstm
python -m src.eval_transformer_pipeline
```

---

## Работа через Jupyter

```bash
jupyter lab
```

Открыть `solution.ipynb`, нажать "Run All". Там уже выставлены:
- `MAX_ROWS=1000`
- Автодетект `cuda`/`cpu`
- Прогресс-бары

---

## Где настраивать

В файле `configs/default.yaml`:
- `data.max_rows` — сколько строк брать при запуске (по умолчанию 1000)
- `training.device` — `cuda` или `cpu`
- `tokenization.sequence_length` — длина входной последовательности
- `training.num_epochs` — сколько эпох учить LSTM

---

## 

- При первом запуске DistilGPT2 скачается автоматически с HuggingFace.
- Если не видно прогресс-баров в Jupyter — поставьте `ipywidgets`.
- Почему ограничение в 1000 строк: у меня ограниченные ресурсы (например, RTX 3050), а на полном датасете всё долго. Но 1000 хватает для теста пайплайна и сравнения моделей.

---
