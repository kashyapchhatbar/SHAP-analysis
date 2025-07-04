import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import r2_score
import warnings
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import xgboost
import seaborn as sns
sns.set(style="ticks", context="paper", font_scale=1)

# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def record_loss(model, X_train, y_train, X_val, y_val, X_test, y_test, learning_rate, alpha, patience=20, min_delta=1e-6):
    max_iter = 2000
    
    """Train the PyTorch model and return validation R2 score"""
    model = model.to(device)    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    loss_history = []
    val_loss_history = []
    train_r2_history = []
    val_r2_history = []
    
    r2_train_scores, r2_val_scores, r2_test_scores = np.array([]), np.array([]), np.array([])
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32, device=device)    
    
    # Training loop
    model.train()
    for epoch in range(max_iter):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        train_r2_score = r2_score(y_train_tensor.detach().cpu().numpy(), outputs.detach().cpu().numpy())
        train_r2_history.append(np.mean(train_r2_score))
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_r2_score = r2_score(y_val_tensor.detach().cpu().numpy(), val_outputs.detach().cpu().numpy())
        val_r2_history.append(np.mean(val_r2_score))        
        val_loss_history.append(val_loss.item())
        model.train()
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss.item()
            epochs_no_improve = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
        
        # Stop if no improvement for 'patience' epochs
        if epochs_no_improve >= patience:
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        y_val_pred = pd.DataFrame(val_outputs.detach().cpu().numpy(), index=y_val.index, columns=y_val.columns)
        test_outputs = model(X_test_tensor)
        y_test_pred = pd.DataFrame(test_outputs.detach().cpu().numpy(), index=y_test.index, columns=y_test.columns)
        train_outputs = model(X_train_tensor)
        y_train_pred = pd.DataFrame(train_outputs.detach().cpu().numpy(), index=y_train.index, columns=y_train.columns)
        
    r2_train_scores = r2_score(y_train, y_train_pred)
    r2_val_scores = r2_score(y_val, y_val_pred)
    r2_test_scores = r2_score(y_test, y_test_pred)
    
    return loss_history, val_loss_history, train_r2_history, val_r2_history, r2_train_scores, r2_val_scores, r2_test_scores

def find_best_params(optuna_storage):
    # Create optuna storage
    try:
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(optuna_storage),
        )
    except FileNotFoundError:
        print(f"File not found: {optuna_storage}")
        print(f"Model may not be tuned yet...")
        sys.exit(1)

    study_name = optuna.study.get_all_study_names(storage=storage)[0]    

    # Load the study
    study = optuna.load_study(storage=storage, study_name=study_name)

    best_params = study.best_params

    return best_params

def pytorch_model():
    """Plot the training and validation loss history"""
    r2_train_scores, r2_val_scores, r2_test_scores = np.array([]), np.array([]), np.array([])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8, 3))
    for i in range(1, 6):
        X_train_file = snakemake.input.X_train.replace("/1/", f"/{i}/")
        y_train_file = snakemake.input.y_train.replace("/1/", f"/{i}/")
        X_val_file = snakemake.input.X_val.replace("/1/", f"/{i}/")
        y_val_file = snakemake.input.y_val.replace("/1/", f"/{i}/")
        X_test_file = snakemake.input.X_test.replace("/1/", f"/{i}/")
        y_test_file = snakemake.input.y_test.replace("/1/", f"/{i}/")
        optuna_storage_file = snakemake.input.pytorch_optuna.replace("/1/", f"/{i}/")
            
        # Load data
        X_train = pd.read_csv(X_train_file, sep="\t", index_col=0)
        y_train = pd.read_csv(y_train_file, sep="\t", index_col=0)
        X_val = pd.read_csv(X_val_file, sep="\t", index_col=0)
        y_val = pd.read_csv(y_val_file, sep="\t", index_col=0)
        X_test = pd.read_csv(X_test_file, sep="\t", index_col=0)
        to_drop = list(set(set(X_train.index).union(X_val.index)).intersection(X_test.index))
        X_test = X_test.drop(to_drop, axis=0).copy()
        y_test = pd.read_csv(y_test_file, sep="\t", index_col=0).loc[list(X_test.index)]
        
        best_params = find_best_params(optuna_storage_file)
        
        final_model = SimpleMLP(
            input_size=X_train.shape[1],
            hidden_layer_sizes=(best_params['hidden_layer_sizes_1'], 
                            best_params['hidden_layer_sizes_2']),
            output_size=y_train.shape[1],        
        )
        
        # Get loss history, validation loss history, and validation R2 history
        loss_history, val_history, train_r2_history, val_r2_history, _r2_train_scores, _r2_val_scores, _r2_test_scores = record_loss(
            final_model, X_train, y_train, X_val, y_val, X_test, y_test,
            best_params['learning_rate_init'],                          
            best_params['alpha'])
        
        if i == 1:
            ax.plot(loss_history, label='Training Loss', alpha=0.5, color='black')
            ax.plot(val_history, label='Validation Loss', alpha=0.5, color='blue')
            bx.plot(train_r2_history, label='Training R2', alpha=0.5, color='black')
            bx.plot(val_r2_history, label='Validation R2', alpha=0.5, color='blue')
        else:
            ax.plot(loss_history, alpha=0.5, color='black')
            ax.plot(val_history, alpha=0.5, color='blue')
            bx.plot(train_r2_history, alpha=0.5, color='black')
            bx.plot(val_r2_history, alpha=0.5, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSELoss')
        ax.set_title('Loss History')
        bx.set_xlabel('Epoch')
        bx.set_ylabel('R2 Score')
        bx.set_title('R2 Score History')
        
        r2_train_scores = np.append(r2_train_scores, _r2_train_scores)
        r2_val_scores = np.append(r2_val_scores, _r2_val_scores)
        r2_test_scores = np.append(r2_test_scores, _r2_test_scores)
    
    ax.legend()
    bx.legend()
    ax.set_xlim(0, 100)
    bx.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(snakemake.output.loss_plot, bbox_inches='tight')
    
    return r2_train_scores, r2_val_scores, r2_test_scores

def mlp_model():
    """Train the MLP model and return R2 scores for each fold"""
    
    r2_train_scores, r2_val_scores, r2_test_scores = np.array([]), np.array([]), np.array([])
    for i in range(1, 6):
        X_train_file = snakemake.input.X_train.replace("/1/", f"/{i}/")
        y_train_file = snakemake.input.y_train.replace("/1/", f"/{i}/")
        X_val_file = snakemake.input.X_val.replace("/1/", f"/{i}/")
        y_val_file = snakemake.input.y_val.replace("/1/", f"/{i}/")
        X_test_file = snakemake.input.X_test.replace("/1/", f"/{i}/")
        y_test_file = snakemake.input.y_test.replace("/1/", f"/{i}/")
        optuna_storage_file = snakemake.input.mlp_optuna.replace("/1/", f"/{i}/")
        best_params = find_best_params(optuna_storage_file)
    
        # Load the data
        X_train = pd.read_csv(X_train_file, sep="\t", index_col=0)
        X_val = pd.read_csv(X_val_file, sep="\t", index_col=0)
        y_train = pd.read_csv(y_train_file, sep="\t", index_col=0)        
        X_test = pd.read_csv(X_test_file, sep="\t", index_col=0)
        to_drop = list(set(set(X_train.index).union(X_val.index)).intersection(X_test.index))
        X_test = X_test.drop(to_drop, axis=0).copy()
        y_test = pd.read_csv(y_test_file, sep="\t", index_col=0).loc[list(X_test.index)]        
        y_val = pd.read_csv(y_val_file, sep="\t", index_col=0)
    
        model = MLPRegressor(
            # learning_rate_init=best_params["learning_rate_init"],
            alpha=best_params["alpha"],
            activation=best_params["activation"],
            hidden_layer_sizes=(
                best_params["hidden_layer_sizes_1"],
                best_params["hidden_layer_sizes_2"],
                # best_params["hidden_layer_sizes_3"],
            ),
            random_state=42,
            # max_iter=best_params["max_iter"]
        )
        
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_test_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=y_test.columns)
        y_val_pred = pd.DataFrame(model.predict(X_val), index=y_val.index, columns=y_val.columns)
        y_train_pred = pd.DataFrame(model.predict(X_train), index=y_train.index, columns=y_train.columns)
        r2_train_scores = np.append(r2_train_scores, r2_score(y_train, y_train_pred))
        r2_val_scores = np.append(r2_val_scores, r2_score(y_val, y_val_pred))
        r2_test_scores = np.append(r2_test_scores, r2_score(y_test, y_test_pred))
        
    return r2_train_scores, r2_val_scores, r2_test_scores

def xgboost_model():
    """Train the XGBoost model and return R2 scores for each fold"""
    r2_train_scores, r2_val_scores, r2_test_scores = np.array([]), np.array([]), np.array([])
    for i in range(1, 6):
        X_train_file = snakemake.input.X_train.replace("/1/", f"/{i}/")
        y_train_file = snakemake.input.y_train.replace("/1/", f"/{i}/")
        X_val_file = snakemake.input.X_val.replace("/1/", f"/{i}/")
        y_val_file = snakemake.input.y_val.replace("/1/", f"/{i}/")
        X_test_file = snakemake.input.X_test.replace("/1/", f"/{i}/")
        y_test_file = snakemake.input.y_test.replace("/1/", f"/{i}/")
        optuna_storage_file = snakemake.input.xgboost_optuna.replace("/1/", f"/{i}/")
        best_params = find_best_params(optuna_storage_file)
        
        # Load the data
        X_train = pd.read_csv(X_train_file, sep="\t", index_col=0)
        X_val = pd.read_csv(X_val_file, sep="\t", index_col=0)
        y_train = pd.read_csv(y_train_file, sep="\t", index_col=0)
        X_test = pd.read_csv(X_test_file, sep="\t", index_col=0)
        to_drop = list(set(set(X_train.index).union(X_val.index)).intersection(X_test.index))
        X_test = X_test.drop(to_drop, axis=0).copy()
        y_test = pd.read_csv(y_test_file, sep="\t", index_col=0).loc[list(X_test.index)]
        y_val = pd.read_csv(y_val_file, sep="\t", index_col=0)
        
        model = xgboost.XGBRegressor(
            # n_estimators=n_estimators,
            learning_rate=best_params["learning_rate_init"],
            tree_method=best_params["tree_method"],
            random_state=42,
            max_depth=best_params["max_depth"],        
        )
        
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_test_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=y_test.columns)
        y_val_pred = pd.DataFrame(model.predict(X_val), index=y_val.index, columns=y_val.columns)
        y_train_pred = pd.DataFrame(model.predict(X_train), index=y_train.index, columns=y_train.columns)
        r2_train_scores = np.append(r2_train_scores, r2_score(y_train, y_train_pred))
        r2_val_scores = np.append(r2_val_scores, r2_score(y_val, y_val_pred))
        r2_test_scores = np.append(r2_test_scores, r2_score(y_test, y_test_pred))
        
    return r2_train_scores, r2_val_scores, r2_test_scores

def linear_model():
    """Train the Linear Regression model and return R2 scores for each fold"""
    r2_train_scores, r2_val_scores, r2_test_scores = np.array([]), np.array([]), np.array([])
    for i in range(1, 6):
        X_train_file = snakemake.input.X_train.replace("/1/", f"/{i}/")
        y_train_file = snakemake.input.y_train.replace("/1/", f"/{i}/")
        X_val_file = snakemake.input.X_val.replace("/1/", f"/{i}/")
        y_val_file = snakemake.input.y_val.replace("/1/", f"/{i}/")
        X_test_file = snakemake.input.X_test.replace("/1/", f"/{i}/")
        y_test_file = snakemake.input.y_test.replace("/1/", f"/{i}/")
        optuna_storage_file = snakemake.input.linear_optuna.replace("/1/", f"/{i}/")
        best_params = find_best_params(optuna_storage_file)
        
        # Load the data
        X_train = pd.read_csv(X_train_file, sep="\t", index_col=0)
        X_val = pd.read_csv(X_val_file, sep="\t", index_col=0)
        y_train = pd.read_csv(y_train_file, sep="\t", index_col=0)        
        X_test = pd.read_csv(X_test_file, sep="\t", index_col=0)
        to_drop = list(set(set(X_train.index).union(X_val.index)).intersection(X_test.index))
        X_test = X_test.drop(to_drop, axis=0).copy()
        y_test = pd.read_csv(y_test_file, sep="\t", index_col=0).loc[list(X_test.index)]        
        y_val = pd.read_csv(y_val_file, sep="\t", index_col=0)
        
        model = LinearRegression(
            fit_intercept=best_params["fit_intercept"],
            copy_X=best_params["copy_X"],
            positive=best_params["positive"],
        )
        
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_test_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=y_test.columns)
        y_val_pred = pd.DataFrame(model.predict(X_val), index=y_val.index, columns=y_val.columns)
        y_train_pred = pd.DataFrame(model.predict(X_train), index=y_train.index, columns=y_train.columns)
        
        r2_train_scores = np.append(r2_train_scores, r2_score(y_train, y_train_pred))
        r2_val_scores = np.append(r2_val_scores, r2_score(y_val, y_val_pred))
        r2_test_scores = np.append(r2_test_scores, r2_score(y_test, y_test_pred))
        
    return r2_train_scores, r2_val_scores, r2_test_scores
    
if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check CUDA availability
    device = torch.device(snakemake.params.device if torch.cuda.is_available() else "cpu")
    
    dfs = []
    r2_train_scores, r2_val_scores, r2_test_scores = pytorch_model()
    dfs.append(pd.DataFrame({"pytorch_train_r2": r2_train_scores,
        "pytorch_val_r2": r2_val_scores,
        "pytorch_test_r2": r2_test_scores}))
    r2_train_scores, r2_val_scores, r2_test_scores = mlp_model()
    dfs.append(pd.DataFrame({"mlp_train_r2": r2_train_scores,
        "mlp_val_r2": r2_val_scores,
        "mlp_test_r2": r2_test_scores}))
    r2_train_scores, r2_val_scores, r2_test_scores = xgboost_model()
    dfs.append(pd.DataFrame({"xgboost_train_r2": r2_train_scores,
        "xgboost_val_r2": r2_val_scores,
        "xgboost_test_r2": r2_test_scores}))
    r2_train_scores, r2_val_scores, r2_test_scores = linear_model()
    dfs.append(pd.DataFrame({"linear_train_r2": r2_train_scores,
        "linear_val_r2": r2_val_scores,
        "linear_test_r2": r2_test_scores}))
    r2_df = pd.concat(dfs, axis=1)
    r2_df.to_csv(snakemake.output.r2_scores, sep="\t", index=False)
    
    r2_melt_df = pd.melt(r2_df)
    r2_melt_df["model"] = r2_melt_df["variable"].str.split("_").str[0]
    r2_melt_df["set"] = r2_melt_df["variable"].str.split("_").str[1]
    
    fig, ax = plt.subplots(figsize=(3, 1.5))
    sns.stripplot(
        data=r2_melt_df,
        y="model",
        x="value",
        hue="set",
        dodge=True,
        ax=ax,        
        size=3,
        palette=["black", "blue", "red"],
        marker="o",
        linewidth=0.1,
        edgecolor="white",
        jitter=True,
    )
    ax.set_xlabel("R2 Score")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)    
    ax.tick_params(axis='y', labelleft=False, labelright=True, left=False, right=True)
    ax.legend(loc='upper left', fontsize='x-small')
    fig.tight_layout()
    fig.savefig(snakemake.output.r2_plot, bbox_inches='tight')
    fig.savefig(snakemake.output.r2_plot.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)