# Hurricane-Induced-Economic-Loss-Prediction

## Install Environment

We use [uv](https://github.com/astral-sh/uv) to manage this project.

```shell
uv sync
```

Packages we use
```
hurricane v0.1.0
├── geopandas v1.0.1
├── matplotlib v3.9.4
├── pandas v2.2.3 (*)
├── scikit-learn v1.6.1
├── scipy v1.13.1 (*)
├── shapely v2.0.7 (*)
└── xgboost v2.1.4
```

## Run Experiments

Predict economic loss. 

`model_name`: Name of the model to run (e.g., RF, XGB, NN, GBM, Stacked)
```shell
python main.py --model_name XGB
```

## Visualization

plot importance
```shell
python visualization.py
```

plot heatmap
```shell
python visualization.py
```
