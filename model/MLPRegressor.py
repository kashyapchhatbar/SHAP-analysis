from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from concurrent.futures import ProcessPoolExecutor as ThreadPoolExecutor
import pandas as pd
import optuna
import sys


def objective(trial):
    # Suggest values for hyperparameters
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    hidden_layer_sizes_1 = trial.suggest_int("hidden_layer_sizes_1", 8, 32)
    hidden_layer_sizes_2 = trial.suggest_int("hidden_layer_sizes_2", 8, 32)
    activation = trial.suggest_categorical("activation", ["tanh", "relu", "logistic"])
    max_iter = trial.suggest_int("max_iter", 200, 400, log=True)

    model = MLPRegressor(
        alpha=alpha,
        activation=activation,
        hidden_layer_sizes=(hidden_layer_sizes_1, hidden_layer_sizes_2),
        max_iter=max_iter,
    )

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_val)
    r2 = r2_score(y_pred.mean(axis=1), y_val.mean(axis=1))

    return r2

# Specify the dataset
try:
    dataset = sys.argv[1]
    max_workers = int(sys.argv[2])
    print(f"Dataset: {dataset}")
    print(f"Tuning journal: {dataset}/tuning_journal.log")
except IndexError:
    print("Please specify the dataset...")
    print("I will exit now...")
    sys.exit(1)


# Load the data
X_train = pd.read_csv(f"{dataset}/X_train.tsv", sep="\t", index_col=0)
X_val = pd.read_csv(f"{dataset}/X_val.tsv", sep="\t", index_col=0)
y_train = pd.read_csv(f"{dataset}/y_train.tsv", sep="\t", index_col=0)
y_val = pd.read_csv(f"{dataset}/y_val.tsv", sep="\t", index_col=0)
X_test = pd.read_csv(f"{dataset}/X_test.tsv", sep="\t", index_col=0)
y_test = pd.read_csv(f"{dataset}/y_test.tsv", sep="\t", index_col=0)

# Create optuna storage
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"{dataset}/tuning_journal.log"),
)

# Create study object
study = optuna.create_study(storage=storage, direction="maximize")

# Run optimization process in parallel
with ThreadPoolExecutor(max_workers=max_workers) as pool:
    for i in range(max_workers):
        pool.submit(study.optimize, objective, n_trials=30, show_progress_bar=True)
print(f"Best params: {study.best_params}")
