from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import optuna
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def mean_subset(df, df2, description):
    to_sample = int(len(df)/2)
    for i in range(50):
        to_yield = pd.concat(
            [df.sample(to_sample).median().abs(),
             df2.sample(to_sample).median().abs()], axis=1
        )
        to_yield.columns = ["med(|SHAP|) (misregulated)", "med(|SHAP|) (random)"]
        yield pd.melt(to_yield.reset_index(), id_vars="index")

def get_mean_sig_signals(new_df):
    pvalues = []
    signals = []
    mean_shap_corrs = []
    for i, j in new_df.groupby("index"):
        to_compare = []        
        for k, l in j.groupby("variable"):
            if k == "med(|SHAP|) (misregulated)":
                to_compare.append(list(l["value"].values))
                mean_shap_corrs.append(l["value"].mean())
            elif k == "med(|SHAP|) (random)":
                to_compare.append(list(l["value"].values))
        ttest = stats.ttest_rel(to_compare[0], to_compare[1])
        pvalues.append(ttest.pvalue)
        signals.append(i)
    adj_pvalues = stats.false_discovery_control(pvalues)
    sig_signals = []
    for signal, adj_pval, mean_shap_corr in zip(signals, adj_pvalues, mean_shap_corrs):
        # if adj_pval <= 0.01 and mean_shap_corr >= 0.05:
            sig_signals.append(signal)
    return sig_signals

def plot_nonlinear(degron, dataset, description):    
    shap_values = pd.read_csv(f"{degron}.shap_values.tsv", sep="\t", index_col=0)
    shap_values.columns = [i.replace(" mean ", "\n") for i in shap_values.columns]
    random_shap_values = pd.read_csv(f"{degron}.random_shap_values.tsv", sep="\t", index_col=0)
    random_shap_values.columns = [i.replace(" mean ", "\n") for i in random_shap_values.columns]
    
    pvalues = []
    signals = []
    
    shap_mean_subset = pd.concat(mean_subset(shap_values, random_shap_values, description))
    sig_signals = get_mean_sig_signals(shap_mean_subset)
    shap_mean_subset = shap_mean_subset[shap_mean_subset["index"].isin(sig_signals)].copy()
    min_height = max(shap_mean_subset["index"].nunique()*0.75, 2)
    fig, ax = plt.subplots(1, figsize=(7, min_height))
    order = list(shap_mean_subset[shap_mean_subset["variable"]=="med(|SHAP|) (misregulated)"].drop(
        "variable", axis=1).groupby("index").mean().sort_values(by="value", ascending=False).index)
    sns.barplot(data=shap_mean_subset, y="index", x="value", ax=ax,
                  order=order, errorbar="pi", hue="variable", legend=True,
                  palette=sns.color_palette(["dodgerblue", "black"]))
    # ax.grid(True)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel(f"|med(SHAP)| for {description}")
    ax.set_ylabel("")
    _ = ax.set_title(dataset)
    fig.tight_layout()
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(len(order)-0.5, -0.5)

    for xlabel in range(len(order)):        
        if xlabel % 2 == 0:            
            ax.fill_between(np.arange(xmin, xmax+0.01, 0.01), xlabel-0.5, xlabel+0.5,
                            facecolor='lightgrey', alpha=0.5, zorder=-10)
    
    # fig.savefig(f"{degron}.SHAP_median.pdf", bbox_inches="tight")
    # fig.savefig(f"{degron}.SHAP_median.svg", bbox_inches="tight")
    return shap_mean_subset

def find_best_params(dataset, top=5):
    # Create optuna storage
    try:
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(f"{dataset}/tuning_journal.log"),
        )
    except FileNotFoundError:
        print(f"File not found: {dataset}/tuning_journal.log")
        print(f"Model may not be tuned yet...")
        sys.exit(1)

    study_name = optuna.study.get_all_study_names(storage=storage)[0]    

    # Load the study
    study = optuna.load_study(storage=storage, study_name=study_name)

    trails_df = study.trials_dataframe()
    top_trails = trails_df.sort_values(by="value", ascending=False).head(100).sample(top)

    for i, j in top_trails.iterrows():
        params_dict = {}
        for k, l in j.items():
            if k.startswith("params"):
                params_dict[k.partition("_")[-1]] = l
        yield i, params_dict

def model_data(best_params, dataset):
    
    # Load the data
    X_train = pd.read_csv(f"{dataset}/X_train.tsv", sep="\t", index_col=0)
    X_val = pd.read_csv(f"{dataset}/X_val.tsv", sep="\t", index_col=0)
    y_train = pd.read_csv(f"{dataset}/y_train.tsv", sep="\t", index_col=0)
    y_val = pd.read_csv(f"{dataset}/y_val.tsv", sep="\t", index_col=0)
    X_test = pd.read_csv(f"{dataset}/X_test.tsv", sep="\t", index_col=0)
    y_test = pd.read_csv(f"{dataset}/y_test.tsv", sep="\t", index_col=0)
    y_val_test = pd.concat([y_val, y_test])
    X_val_test = pd.concat([X_val, X_test])

    model = MLPRegressor(
        alpha=best_params["alpha"],
        activation=best_params["activation"],
        hidden_layer_sizes=(
            best_params["hidden_layer_sizes_1"],
            best_params["hidden_layer_sizes_2"],
        ),
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train.mean(axis=1), y_pred_train.mean(axis=1))
    y_pred = model.predict(X_val_test)
    r2_test = r2_score(y_val_test.mean(axis=1), y_pred.mean(axis=1))

    # print(f"R2 train: {r2_train:.4f} R2 val+test score: {r2_test:.4f}")

    return r2_train, r2_test, model

def get_all_sum_shap_values(dataset, degron, marks=[], n=5000, ascending=False):
    shap_values = pd.concat([pd.read_csv(f"{degron}.shap_values.tsv", sep="\t", index_col=0),
                             pd.read_csv(f"{degron}.random_shap_values.tsv", sep="\t", index_col=0)])
    X = pd.concat([pd.read_csv(f"{dataset}/X_train.tsv", sep="\t", index_col=0),
                   pd.read_csv(f"{dataset}/X_val.tsv", sep="\t", index_col=0),
                   pd.read_csv(f"{dataset}/X_test.tsv", sep="\t", index_col=0)])
    df = pd.read_csv(degron, sep="\t", index_col=0)
    sig_genes = list(df[(df["padj"]<0.05)].index)
    overlap = {"SHAP": []}
    for mark in marks:
        overlap[mark] = []
    shap_values["log2FoldChange"] = df["log2FoldChange"]
    shap_values["sum"] = shap_values[marks].sum(axis=1)
    
    for i in range(1, len(sig_genes)+1):
        overlap["SHAP"].append(len(set(shap_values.sort_values(by="sum", ascending=ascending).head(i).index
            ).intersection(sig_genes))/i)
        for mark in marks:
            overlap[mark].append(len(set(X.sort_values(by=mark, ascending=False).head(i).index
                ).intersection(sig_genes))/i)

    fig, ax = plt.subplots(figsize=(7,2.5))
    pd.DataFrame(overlap)["SHAP"].plot(logx=True, ax=ax, linewidth=2, color="dodgerblue", legend=False)
    pd.DataFrame(overlap).drop("SHAP", axis=1).plot(logx=True, ax=ax, linewidth=2, color="grey", legend=False)
    ax.set_ylim(0.4, 1)
    ax.set_xlim(1, len(sig_genes))
    sns.despine()
    fig.tight_layout()