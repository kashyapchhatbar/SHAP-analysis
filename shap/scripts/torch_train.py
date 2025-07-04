import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import r2_score
from concurrent.futures import ProcessPoolExecutor as ThreadPoolExecutor
import warnings
import os

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
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_model(model, X_train, y_train, X_val, y_val, learning_rate, alpha, patience=20, min_delta=1e-6):
    max_iter = 2000
    
    """Train the PyTorch model and return validation R2 score"""
    model = model.to(device)    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32, device=device)
    
    # Training loop
    model.train()
    for epoch in range(max_iter):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
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
        y_pred = pd.DataFrame(val_outputs.cpu().numpy(), index=y_val.index, columns=y_val.columns)
    
    # Calculate R2 scores for each column
    r2_scores = []
    for col in y_val.columns:
        r2 = r2_score(y_val[col], y_pred[col])
        r2_scores.append(r2)
    
    # Return mean R2 score
    return np.mean(r2_scores)


def objective(trial):
    # Suggest values for hyperparameters
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
    alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
    hidden_layer_sizes_1 = trial.suggest_int("hidden_layer_sizes_1", 16, 64)
    hidden_layer_sizes_2 = trial.suggest_int("hidden_layer_sizes_2", 16, 64)
    
    # Create model
    model = SimpleMLP(
        input_size=X_train.shape[1],
        hidden_layer_sizes=(hidden_layer_sizes_1, hidden_layer_sizes_2),
        output_size=y_train.shape[1],        
    )
    
    # Train and evaluate
    mean_r2 = train_model(model, X_train, y_train, X_val, y_val, 
                         learning_rate_init, alpha)
    
    return mean_r2


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check CUDA availability
    device = torch.device(snakemake.params.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    
    # Load data
    X_train = pd.read_csv(snakemake.input.X_train, sep="\t", index_col=0)
    y_train = pd.read_csv(snakemake.input.y_train, sep="\t", index_col=0)
    X_val = pd.read_csv(snakemake.input.X_val, sep="\t", index_col=0)
    y_val = pd.read_csv(snakemake.input.y_val, sep="\t", index_col=0)
    X_test = pd.read_csv(snakemake.input.X_test, sep="\t", index_col=0)
    y_test = pd.read_csv(snakemake.input.y_test, sep="\t", index_col=0)
    
    # Print data shapes
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print()
    
    # Create optuna storage and study
    storage_file = snakemake.output.optuna_storage
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(storage_file)
    )
    study = optuna.create_study(storage=storage, direction="maximize", load_if_exists=True)
    
    # Configuration (replace with your snakemake params)
    n_trials = snakemake.params.n_trials if hasattr(snakemake.params, 'n_trials') else 100    
    

    print(f"Starting optimization with {n_trials} trials...")
    
    # Run the optimization
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=True, 
        catch=(KeyboardInterrupt, SystemExit),
        n_jobs=1  # Use single-threaded execution for reproducibility
    )
    print(f"\nOptimization completed with {len(study.trials)} trials.")

    best_trial = study.best_trial
    
    print(f"\nOptimization completed!")
    print(f"Best trial: {best_trial.number}")
    print(f"Best R2 score: {best_trial.value:.4f}")
    print(f"Best parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best parameters
    print(f"\nTraining final model with best parameters...")
    
    final_model = SimpleMLP(
        input_size=X_train.shape[1],
        hidden_layer_sizes=(best_trial.params['hidden_layer_sizes_1'], 
                           best_trial.params['hidden_layer_sizes_2']),
        output_size=y_train.shape[1],        
    )
    
    # Train final model
    final_r2 = train_model(final_model, X_train, y_train, X_val, y_val,
                          best_trial.params['learning_rate_init'],                          
                          best_trial.params['alpha'])
    
    # Test on test set
    final_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32, device=device)
        test_outputs = final_model(X_test_tensor)
        y_test_pred = pd.DataFrame(test_outputs.cpu().numpy(), index=y_test.index, columns=y_test.columns)
    
    # Calculate test R2 scores
    test_r2_scores = []
    for col in y_test.columns:
        r2 = r2_score(y_test[col], y_test_pred[col])
        test_r2_scores.append(r2)
    
    test_mean_r2 = np.mean(test_r2_scores)
    torch.save(final_model.state_dict(), snakemake.output.model)
    
    print(f"Final validation R2: {final_r2:.4f}")
    print(f"Test R2: {test_mean_r2:.4f}")
    print(f"Model saved to: {snakemake.output.model}")
    print(f"Optuna storage saved to: {storage_file}")