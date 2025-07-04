import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import r2_score
import warnings
import shap

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
    
    target_genes_df = pd.read_csv(snakemake.input.target_genes, sep="\t", index_col=0)
    sig_target_genes_df = target_genes_df[target_genes_df["padj"] < snakemake.params.padj_threshold].copy()
    # target_genes_df = target_genes_df[target_genes_df["padj"] < snakemake.params.padj_threshold].copy()
    # target_genes_df = target_genes_df[target_genes_df["log2FoldChange"] > snakemake.params.log2fc_threshold].copy()
    target_genes = target_genes_df.index.tolist()
    
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
    storage_file = snakemake.input.optuna_storage
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(storage_file)
    )    
    best_trial = storage.get_best_trial(study_id=0)
    
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
        train_target_genes = list(set(X_train.index).intersection(sig_target_genes_df.index))
        X_train_without_target_genes = X_train.drop(train_target_genes, axis=0, errors='ignore').copy()
        X_train_summary = torch.tensor(shap.kmeans(X_train_without_target_genes, snakemake.params.kmeans).data, dtype=torch.float32, device=device)        
        test_outputs = final_model(X_test_tensor)
        y_test_pred = pd.DataFrame(test_outputs.cpu().numpy(), index=y_test.index, columns=y_test.columns)
    
    # Calculate test R2 scores
    test_r2_scores = []
    for col in y_test.columns:
        r2 = r2_score(y_test[col], y_test_pred[col])
        test_r2_scores.append(r2)
    
    test_mean_r2 = np.mean(test_r2_scores)
    
    print(f"Final validation R2: {final_r2:.4f}")
    print(f"Test R2: {test_mean_r2:.4f}")
    
    mark_column = f"{snakemake.wildcards.mark} {snakemake.wildcards.metric} {snakemake.wildcards.context}"
    try:
        mark_id = y_test.columns.get_loc(mark_column)
    except KeyError:
        print(f"Column '{mark_column}' not found in the data. Available columns: {y_test.columns.tolist()}")
        sys.exit(1)
    
    e = shap.DeepExplainer(final_model, X_train_summary)
    shap_values = e.shap_values(X_test_tensor)[:, :, mark_id]
    shap_values_df = pd.DataFrame(shap_values, index=X_test.index, columns=X_test.columns)
    shap_values_df.to_csv(snakemake.output.shap_values, sep="\t")
    print(f"SHAP values saved to {snakemake.output.shap_values}")