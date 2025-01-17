# get your imports out of the way!
import os

default_n_threads = 1
os.environ["OPENBLAS_NUM_THREADS"] = f"{default_n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{default_n_threads}"
os.environ["OMP_NUM_THREADS"] = f"{default_n_threads}"

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark import SparkConf, SparkContext
from typing import Iterator
import optuna
import sys
import shap
import pathlib
from rich.console import Console
from rich.markdown import Markdown

def print_markdown(MARKDOWN: str):
    """
    The function `print_markdown` takes a Markdown string as input and prints it to
    the console in Markdown format.

    :param MARKDOWN: The `print_markdown` function takes a Markdown formatted string
    as input and prints it to the console using the Rich library. The `MARKDOWN`
    parameter should be a string containing text formatted in Markdown syntax
    :type MARKDOWN: str
    """
    console = Console()
    console.print(Markdown(MARKDOWN))

def find_best_params(dataset):
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

    best_params = study.best_params

    return best_params

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

    y_pred = model.predict(X_val_test)
    r2_test = r2_score(y_val_test.mean(axis=1), y_pred.mean(axis=1))

    print(f"R2 val+test score: {r2_test:.4f}")

    return model

def calculate_shap(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for X in iterator:
        yield pd.DataFrame(
            np.array(explainer.shap_values(np.array(X))).sum(axis=-1),
            # np.array(explainer.shap_values(np.array(X))).mean(axis=-1),
            # changing to sum to see if it improves the correlation
            columns=X.columns,
        )


def get_shap_explainer(model, dataset):
    # Load the data
    X_train = pd.read_csv(f"{dataset}/X_train.tsv", sep="\t", index_col=0)
    X_val = pd.read_csv(f"{dataset}/X_val.tsv", sep="\t", index_col=0)
    X_test = pd.read_csv(f"{dataset}/X_test.tsv", sep="\t", index_col=0)
    X = pd.concat([X_train, X_val, X_test])

    X_train_summary = shap.kmeans(X_train, 70)
    explainer = shap.KernelExplainer(model.predict, X_train_summary)

    return explainer, X

def evaluate_attributions(
    all_input_data, attributions, degron, description
):
    # Evaluate attributions
    if pathlib.Path(degron).exists():
        degron_data = pd.read_csv(degron, sep="\t", index_col=0)        
        common_genes = list(set(attributions.index).intersection(degron_data.index))
        degron_data = degron_data.loc[common_genes]
        attributions = attributions.loc[common_genes]
        attributions["degron"] = degron_data["log2FoldChange"]

        print(f"Common genes: {len(common_genes)}")

        previous_i = 0
        attributions_sum_columns = []
        attributions_sum_df = []
        for i in range(5, len(attributions.columns) + 1, 5):
            mean_value = attributions[
                attributions.columns[list(range(previous_i, i))]
            ].sum(axis=1)
            attributions_sum_columns.append(
                attributions.columns[previous_i].split(" promoter")[0]
            )
            attributions_sum_df.append(mean_value)
            previous_i = i

        attributions_sum_df = pd.concat(
            attributions_sum_df, axis=1
        )  # Concatenate the list of dataframes
        attributions_sum_df.columns = attributions_sum_columns
        attributions_sum_df["degron"] = degron_data["log2FoldChange"]

        input_data = all_input_data.loc[common_genes]        

        print_markdown(
            "# Feature correlation with transcriptional changes after "
            f"rapid degradation {description} (n={len(common_genes):,})"
        )
        input_data["degron"] = degron_data["log2FoldChange"]
        quartile_df = pd.DataFrame(
            input_data.corr()["degron"].sort_values(ascending=False)
        )
        quartile_df.columns = ["input r"]
        quartile_df["input R²"] = quartile_df["input r"].apply(np.square)
        quartile_df.index.name = "Feature"
        quartile_df["SHAP r"] = (
            attributions.corr()["degron"].sort_values(ascending=False)
        )
        quartile_df["μ SHAP"] = attributions.mean()
        quartile_df["SHAP R²"] = quartile_df["SHAP r"].apply(np.square)
        quartile_df.drop("degron", inplace=True)

        print_markdown(
            quartile_df.sort_values(by="μ SHAP", ascending=False).to_markdown(
                floatfmt=".4f"
            )
        )
        
        # previous_i = 0
        # input_sum_columns = []
        # input_sum_df = []
        # for i in range(5, len(input_data.columns) + 1, 5):
        #     mean_value = input_data[input_data.columns[list(range(previous_i, i))]].sum(
        #         axis=1
        #     )
        #     input_sum_columns.append(
        #         input_data.columns[previous_i].split(" promoter")[0]
        #     )
        #     input_sum_df.append(mean_value)
        #     previous_i = i

        # input_sum_df = pd.concat(input_sum_df, axis=1)
        # input_sum_df.columns = input_sum_columns
        # input_sum_df["degron"] = degron_data["log2FoldChange"]

        # print_markdown(
        #     f"# Feature sum correlation with transcriptional changes after "
        #     f"rapid degradation {description} (n={len(common_genes):,})"
        # )
        # quartile_df = pd.DataFrame(
        #     input_sum_df.corr()["degron"].sort_values(ascending=False)
        # )
        # quartile_df.columns = ["input r"]
        # quartile_df["input R²"] = quartile_df["input r"].apply(np.square)        
        # quartile_df.index.name = "Feature"
        # quartile_df["SHAP r"] = (
        #     attributions_sum_df.corr()["degron"]
        #     .sort_values(ascending=False)
        # )
        # quartile_df["Σ scales μ SHAP"] = attributions_sum_df.mean()
        # quartile_df["SHAP R²"] = quartile_df["SHAP r"].apply(np.square)
        # quartile_df.drop("degron", inplace=True)

        # print_markdown(
        #     quartile_df.sort_values(by="Σ scales μ SHAP", ascending=False).to_markdown(
        #         floatfmt=".4f"
        #     )
        # )

    else:
        print(f"File {degron} does not exist. Skipping evaluation.")

def extract_best_scale(X, shap_values_df, dataset):
    # Extract the best scale
    mean_abs_shap = shap_values_df.mean().abs().sort_values(ascending=False)
    signal_scale = {}
    for feature in mean_abs_shap.index:
        signal = feature.split(" ")[0]
        scale = feature.partition(" ")[-1]
        if signal not in signal_scale:
            signal_scale[signal] = scale
    
    signal_columns = [f"{signal} {scale}" for signal, scale in signal_scale.items()]
    X_reduced = X[signal_columns]
    pathlib.Path(f"{dataset}_reduced").mkdir(parents=True, exist_ok=True)
    X_reduced.to_csv(f"{dataset}_reduced/X_reduced.tsv", sep="\t")

if __name__ == "__main__":
    dataset = sys.argv[1]
    degron = sys.argv[2]
    description = sys.argv[3]
    best_params = find_best_params(dataset)
    print(best_params)
    model = model_data(best_params, dataset)
    explainer, X = get_shap_explainer(model, dataset)
    
    degron_data = pd.read_csv(degron, sep="\t", index_col=0)
    sig_degron_data = degron_data[(degron_data["padj"] < 0.05)].copy()
    common_genes = list(set(X.index).intersection(sig_degron_data.index))
    random_X = X.drop(common_genes).copy()
    random_common_genes = list(set(random_X.index).intersection(degron_data.index))
    random_inputs_df = X.loc[random_common_genes].copy()
    inputs_df = X.loc[common_genes].copy()
    
    if pathlib.Path(f"{degron}.shap_values.tsv").exists() and pathlib.Path(
        f"{degron}.random_shap_values.tsv").exists():
        
        shap_values_df = pd.read_csv(f"{degron}.shap_values.tsv", sep="\t", index_col=0)
        random_shap_values_df = pd.read_csv(f"{degron}.random_shap_values.tsv", sep="\t", index_col=0)                
        
    else:
        
        return_schema = StructType()
        for feature in X.columns:
            return_schema = return_schema.add(StructField(feature, FloatType()))

        config = SparkConf().setAll(
            [("spark.num.executors", "1"), ("spark.executor.cores", "1")]
        )
        spark = SparkSession.builder.config(conf=config).getOrCreate()
        
        df = spark.createDataFrame(inputs_df)
        shap_values = df.mapInPandas(calculate_shap, schema=return_schema)
        shap_values_df = shap_values.toPandas()
        shap_values_df.index = inputs_df.index
        shap_values_df.to_csv(f"{degron}.shap_values.tsv", sep="\t")
        spark.stop()
        
        return_schema = StructType()
        for feature in X.columns:
            return_schema = return_schema.add(StructField(feature, FloatType()))

        config = SparkConf().setAll(
            [("spark.num.executors", "1"), ("spark.executor.cores", "1")]
        )
        spark = SparkSession.builder.config(conf=config).getOrCreate()
        
        df = spark.createDataFrame(random_inputs_df)
        random_shap_values = df.mapInPandas(calculate_shap, schema=return_schema)
        random_shap_values_df = random_shap_values.toPandas()
        random_shap_values_df.index = random_inputs_df.index
        random_shap_values_df.to_csv(f"{degron}.random_shap_values.tsv", sep="\t")
        spark.stop()
        

    evaluate_attributions(X, shap_values_df, degron, description)
    
    evaluate_attributions(X, random_shap_values_df.sample(len(shap_values_df)), 
                          degron, "random")
    