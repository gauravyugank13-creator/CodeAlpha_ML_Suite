# Task 2: Emotion Recognition from Speech

This sub-module implements a machine learning system to classify acoustic speech waveforms into emotional categories (e.g. happy, sad, angry, neutral, fearful, surprised).

## Project Directory Map

```text
task_2_emotion_recognition/
├── README.md                 ← Task documentation
├── requirements_task2.txt    ← Speech-specific python dependencies
├── data/
│   ├── raw/                  ← Raw audio recordings (.wav)
│   └── processed/            ← Extracted acoustic feature files
├── notebooks/                ← Jupyter exploration and EDA files
├── src/
│   ├── __init__.py           ← Package indicator
│   ├── config.py             ← Paths and audio configuration parameters
│   ├── data_loader.py        ← Audio file cataloging and loader
│   ├── audio_preprocessing.py ← Silence removal, trim, normalise
│   ├── feature_extraction.py  ← MFCCs, chroma, and spectral contrast features
│   ├── model_builder.py      ← Scikit-Learn classification pipelines
│   ├── trainer.py            ← Fitting loops and serialization
│   ├── evaluator.py          ← Validation scoring and confusion plotting
│   └── predictor.py          ← Serving and prediction wrappers
├── models/                   ← Pickled model checkpoints
├── results/
│   ├── plots/                ← Metric visualisations
│   ├── metrics/              ← Classification metrics reports
│   └── predictions/          ← Serving history logging registry
└── app/                      ← Flask app serving endpoints
```

## Setup Instructions

### 1. Requirements Installation
Ensure that the task-specific packages are installed in the virtual environment:
```bash
pip install -r task_2_emotion_recognition/requirements_task2.txt
```

### 2. Run Verification
A standalone verification suite checks that folder directories and python skeleton codes load cleanly without runtime overhead:
```bash
python task_2_emotion_recognition/validate_initialization.py
```
