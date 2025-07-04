import xgboost
from sklearn.metrics import r2_score
from concurrent.futures import ProcessPoolExecutor as ThreadPoolExecutor
import pandas as pd
import numpy as np
import optuna
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

optuna.logging.set_verbosity(optuna.logging.WARNING)

@ignore_warnings(category=ConvergenceWarning)
def objective(trial):
    # Suggest values for hyperparameters
    # n_estimators = trial.suggest_int("n_estimators", 100, 500, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-3, 1e-1, log=True)
    tree_method = trial.suggest_categorical("tree_method", ["hist", "exact"])
    max_depth = trial.suggest_int("max_depth", 6, 18)    

    model = xgboost.XGBRegressor(
        # n_estimators=n_estimators,
        learning_rate=learning_rate_init,
        tree_method=tree_method,
        random_state=42,
        max_depth=max_depth,        
    )

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = pd.DataFrame(model.predict(X_val), index=y_val.index, columns=y_val.columns)
    r2_scores = []
    for col in y_val.columns:
        r2 = r2_score(y_val[col], y_pred[col])
        r2_scores.append(r2)
    # Calculate the mean R2 score across all columns
    mean_r2 = np.mean(r2_scores)    
    return mean_r2

# Create optuns storage and study
storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(snakemake.output.optuna_storage)
)
study = optuna.create_study(storage=storage, direction="maximize", load_if_exists=True)

# Load training and validation data
X_train = pd.read_csv(snakemake.input.X_train, sep="\t", index_col=0)
y_train = pd.read_csv(snakemake.input.y_train, sep="\t", index_col=0)
X_val = pd.read_csv(snakemake.input.X_val, sep="\t", index_col=0)
y_val = pd.read_csv(snakemake.input.y_val, sep="\t", index_col=0)
# Run the optimization
with ThreadPoolExecutor(max_workers=snakemake.threads) as executor:
    for i in range(snakemake.threads):
        executor.submit(study.optimize, objective, n_trials=snakemake.params.n_trials, show_progress_bar=True, catch=(KeyboardInterrupt, SystemExit))

# Save the best trial
best_trial = study.best_trial
with open(snakemake.output.best_trial, "w") as f:
    f.write(f"Best trial: {best_trial.number}\n")
    f.write(f"Value: {best_trial.value}\n")
    f.write("Params:\n")
    for key, value in best_trial.params.items():
        f.write(f"{key}: {value}\n")