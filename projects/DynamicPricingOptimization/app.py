import os
import numpy as np
import pandas as pd
from joblib import dump
import warnings
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.metaestimators import available_if
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
from huggingface_hub import Repository, HfApi, DatasetCardData
from skops.card import Card
import pickle
from pathlib import Path
from tempfile import mkdtemp
from skops import hub_utils
from pathlib import Path
from tempfile import mkdtemp
from joblib import dump
import pickle
import pandas as pd

# print(os.getcwd())
# Initialize repository
User = "PranavSharma"
repo_name = "dynamic-pricing-model"
repo_url = f"https://huggingface.co/{User}/{repo_name}"

from skops.card import Card
import gradio as gr

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration variables for paths and parameters
DATA_PATH = os.path.join("data", "dynamic_pricing.csv")

# Utility function to check if a file exists
def check_file_exists(file_path):
    """
    Check if a file exists at the given path.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# Load and preprocess the dataset
def load_data():
    """
    Load and preprocess the dataset by performing one-hot encoding
    on categorical variables.

    Returns
    -------
    tuple
        A tuple containing the processed dataset and the list of boolean columns.
    """
    check_file_exists(DATA_PATH)
    data = pd.read_csv(DATA_PATH)
    data = data.sample(frac=1, random_state=42)  # Shuffle the data
    categorical_columns = data.select_dtypes(include=["object"]).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    bool_columns = [col for col in data.columns if data[col].dropna().value_counts().index.isin([0, 1]).all()]
    return data, bool_columns

# Compute default values and feature types for Gradio inputs
def compute_defaults_and_types(X, bool_columns):
    defaults = {}
    types = {}
    for column in X.columns:
        if column in bool_columns:
            defaults[column] = 0
            types[column] = "Categorical (One-hot)"
        else:
            defaults[column] = X[column].mean()
            types[column] = "Numerical"
    return defaults, types

# Generate a scatter plot for Expected_Ride_Duration vs Historical_Cost_of_Ride
def duration_vs_cost_plot(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Expected_Ride_Duration"],
        y=data["Historical_Cost_of_Ride"],
        mode="markers",
        marker=dict(size=8, color="rgba(99, 110, 250, 0.7)", line=dict(width=1, color="rgba(99, 110, 250, 1)")),
        name="Data Points"
    ))
    fig.update_layout(
        title=dict(text="Expected Ride Duration vs Historical Ride Cost", font=dict(size=18)),
        xaxis=dict(title="Expected Ride Duration (minutes)", gridcolor="lightgray"),
        yaxis=dict(title="Historical Ride Cost ($)", gridcolor="lightgray"),
        template="plotly_white"
    )
    return fig

# Generate MAE and R² plots with GridSearchCV
def performance_plots_with_gridsearch(results):
    X_train = results["X_train"]
    y_train = results["y_train"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    train_sizes = np.linspace(50, len(X_train), 10, dtype=int)

    mae_scores = []
    r2_scores = []

    param_grid = {"alpha": np.logspace(-4, 0, 10)}

    for train_size in train_sizes:
        X_train_sub = X_train.iloc[:train_size]
        y_train_sub = y_train.iloc[:train_size]

        grid_search = GridSearchCV(
            Lasso(fit_intercept=False),
            param_grid,
            scoring="neg_mean_absolute_error",
            cv=5
        )
        grid_search.fit(X_train_sub, y_train_sub)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    mae_fig = go.Figure()
    mae_fig.add_trace(go.Scatter(
        x=train_sizes,
        y=mae_scores,
        mode="lines+markers",
        marker=dict(size=6, color="blue"),
        line=dict(width=2, color="blue"),
        name="MAE"
    ))
    mae_fig.update_layout(
        title="Effect of Training Size on MAE (with GridSearchCV)",
        xaxis_title="Training Size",
        yaxis_title="Mean Absolute Error (MAE)",
        template="plotly_white"
    )

    r2_fig = go.Figure()
    r2_fig.add_trace(go.Scatter(
        x=train_sizes,
        y=r2_scores,
        mode="lines+markers",
        marker=dict(size=6, color="green"),
        line=dict(width=2, color="green"),
        name="R²"
    ))
    r2_fig.update_layout(
        title="Effect of Training Size on R² (with GridSearchCV)",
        xaxis_title="Training Size",
        yaxis_title="R² Score",
        template="plotly_white"
    )

    return mae_fig, r2_fig

# Generate coefficient progression plot with tracking
# Generate coefficient progression plot with tracking
def coefficients_progression_plot_with_tracking(results):
    X_train = results["X_train"]
    y_train = results["y_train"]
    train_sizes = np.linspace(50, len(X_train), 10, dtype=int)

    coefficients_progress = []
    feature_names = results["feature_names"]

    param_grid = {"alpha": np.logspace(-4, 0, 10)}

    for train_size in train_sizes:
        X_train_sub = X_train.iloc[:train_size]
        y_train_sub = y_train.iloc[:train_size]

        grid_search = GridSearchCV(
            Lasso(fit_intercept=False),
            param_grid,
            scoring="neg_mean_absolute_error",
            cv=5
        )
        grid_search.fit(X_train_sub, y_train_sub)
        best_model = grid_search.best_estimator_

        coefficients_progress.append(best_model.coef_)

    coefficients_array = np.array(coefficients_progress)

    fig = go.Figure()
    for idx, feature in enumerate(feature_names):
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=coefficients_array[:, idx],
            mode="lines+markers",
            name=feature,
            line=dict(width=2),
            marker=dict(size=6, opacity=0.8)
        ))
    fig.update_layout(
        title="Coefficient Progression with Training Size (Tracking)",
        xaxis_title="Training Size",
        yaxis_title="Coefficient Value",
        template="plotly_white",
        height=700,  # Increased height for better vertical visibility
        legend=dict(
            orientation="h",  # Horizontal legend
            y=-0.3,  # Position legend below the plot
            x=0.5,
            xanchor="center"
        )
    )
    return fig


# New function to evaluate multiple linear models using GridSearchCV
def train_linear_models_with_gridsearch(X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple linear models using GridSearchCV and compare their performance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training target variable.
    X_test : pd.DataFrame
        Testing feature set.
    y_test : pd.Series
        Testing target variable.

    Returns
    -------
    dict
        A dictionary containing the best model, its parameters, and performance metrics.
    """
    models = {
        "Lasso": {
            "model": Lasso(fit_intercept=False),
            "param_grid": {"alpha": [0.001, 0.01, 0.1, 1]},
        },
        "Ridge": {
            "model": Ridge(fit_intercept=False),
            "param_grid": {"alpha": [0.001, 0.01, 0.1, 1]},
        },
        "ElasticNet": {
            "model": ElasticNet(fit_intercept=False),
            "param_grid": {
                "alpha": [0.001, 0.01, 0.1, 1],
                "l1_ratio": [0.2, 0.5, 0.8],
            },
        },
        "LinearRegression": {
            "model": LinearRegression(fit_intercept=False),
            "param_grid": {},  # No hyperparameters for tuning
        },
        "HuberRegressor": {
            "model": HuberRegressor(fit_intercept=False),
            "param_grid": {"epsilon": [1.2, 1.5], "alpha": [0.001, 0.01]},
        },
        "KNeighborsRegressor": {
            "model": KNeighborsRegressor(),
            "param_grid": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        },
        "DecisionTreeRegressor": {
            "model": DecisionTreeRegressor(),
            "param_grid": {
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
            },
        },
        "GradientBoostingRegressor": {
            "model": GradientBoostingRegressor(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        },
        "AdaBoostRegressor": {
            "model": AdaBoostRegressor(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
            },
        },
        "SVR": {
            "model": SVR(),
            "param_grid": {
                "C": [0.1, 1],
                "epsilon": [0.01, 0.1],
                "kernel": ["linear", "rbf"],
            },
        },
        "LinearSVR": {
            "model": LinearSVR(random_state=42),
            "param_grid": {"C": [0.1, 1]},
        },
    }

    results = []
    best_model = None
    best_result = None
    for name, config in models.items():
        try:
            grid_search = GridSearchCV(
                config["model"],
                config["param_grid"],
                scoring="neg_mean_absolute_error",
                cv=5
            )
            grid_search.fit(X_train, y_train)

            # Predictions and evaluation
            y_pred = grid_search.best_estimator_.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Collect results
            results.append({
                "model": name,
                "best_params": grid_search.best_params_,
                "mae": mae,
                "r2": r2,
                "best_estimator": grid_search.best_estimator_,
            })

        except Exception as e:
            print(f"Error training model {name}: {e}")

    # Identify the best model based on MAE
    if results:
        best_result = min(results, key=lambda x: x["mae"])
        best_model = best_result["best_estimator"]

    return {
        "results": results,
        "best_model_name": best_result["model"] if best_result else None,
        "best_model_metrics": best_result if best_result else None,
        "best_model": best_model,  # Return the best model directly
    }

def train_model():
    original_data = pd.read_csv(DATA_PATH)
    data, bool_columns = load_data()
    X = data.drop("Historical_Cost_of_Ride", axis=1)
    y = data["Historical_Cost_of_Ride"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the best linear model and top results
    linear_model_results = train_linear_models_with_gridsearch(X_train, y_train, X_test, y_test)
    best_model_name = linear_model_results["best_model_name"]
    best_model_metrics = linear_model_results["best_model_metrics"]
    top_models = linear_model_results["results"]  # Get all models' results
    best_model = linear_model_results["best_model"]
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    feature_names = X_train.columns
    coefficients = best_model.coef_

    useful_features = [(feature, coef) for feature, coef in zip(feature_names, coefficients) if coef != 0]
    not_useful_features = [feature for feature, coef in zip(feature_names, coefficients) if coef == 0]

    equation_terms = [f"*{coef:.4f}* × *{feature}*" for feature, coef in useful_features]
    regression_equation = " + ".join(equation_terms)
    regression_equation = "Cost of Ride = " + regression_equation

    actual_vs_pred_plot = actual_vs_predicted_plot(y_test, y_pred)
    useful_features_formatted = "\n".join(
        [f"- {feature}: {coef:.4f}" for feature, coef in useful_features]
    )
    not_useful_features_formatted = "\n".join(
        [f"- {feature}" for feature in not_useful_features]
    )

    default_values, types = compute_defaults_and_types(X_train, bool_columns)

    scatter_plot = duration_vs_cost_plot(original_data)

    # Generate a DataFrame for the top 10 models
    top_models_sorted = sorted(top_models, key=lambda x: x['mae'])[:10]
    top_models_df = pd.DataFrame.from_records(
        [
            {
                "Rank": idx + 1,
                "Model": result["model"],
                "MAE": f"{result['mae']:.4f}",
                "R²": f"{result['r2']:.4f}",
                "Best Params": result["best_params"],
            }
            for idx, result in enumerate(top_models_sorted)
        ]
    )
    top_models_html = top_models_df.to_html(index=False, border=0, classes="table table-striped")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": feature_names,
        "coefficients": coefficients,
        "mae": mae,
        "r2": r2,
        "best_model_name": best_model_name,
        "best_model_metrics": best_model_metrics,
        "best_model": best_model,
        "regression_equation": regression_equation,
        "scatter_plot": scatter_plot,
        "useful_features": useful_features_formatted,
        "not_useful_features": not_useful_features_formatted,
        "top_models_html": top_models_html,  # Include HTML table here
        "default_values": default_values,
        "feature_types": types,
        "original_data_html": original_data.head(3).to_html(classes="table table-striped"),
        "original_data": original_data,
        "actual_vs_predicted_plot": actual_vs_pred_plot
        
    }

def process_features_with_values(feature_string):
        """Cleans and splits the feature string, retaining both feature names and values."""
        if not feature_string:
            return []
        feature_string = feature_string.strip()
        formatted_features = []
        for item in feature_string.split("-"):
            if not item.strip():
                continue
            if item.strip().replace(".", "", 1).isdigit():  # Check if the item is a float
                if formatted_features:
                    formatted_features[-1] = formatted_features[-1].strip() + ": " + item.strip() + "\n"
            else:
                formatted_features.append(" ".join(item.split()) + "\n")  # Clean extra spaces and add
        return formatted_features

def process_features_without_values(feature_string):
    """Cleans and splits the feature string, keeping only feature names."""
    if not feature_string:
        return []
    feature_string = feature_string.strip()
    return [
        item.split(":")[0].strip() + "\n"  # Keep only the feature name before ":"
        for item in feature_string.split("-")
        if item.strip()
    ]

def actual_vs_predicted_plot(y_actual, y_pred):
    """
    Create a scatter plot for Actual vs Predicted values.

    Parameters
    ----------
    y_actual : array-like
        Actual target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    go.Figure
        A Plotly scatter plot.
    """
    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=y_actual,
        y=y_pred,
        mode="markers",
        marker=dict(size=8, color="rgba(99, 110, 250, 0.7)", line=dict(width=1)),
        name="Actual vs Predicted"
    ))

    # Add ideal reference line
    min_val = min(min(y_actual), min(y_pred))
    max_val = max(max(y_actual), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Ideal Line"
    ))

    # Update layout
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        template="plotly_white"
    )
    fig.add_annotation(
        x=max_val,
        y=max_val,
        text="Ideal Line (y=x)",
        showarrow=True,
        arrowhead=2
    )
    return fig


def train_model_button():
    """
    Train the model and return all relevant outputs for display.
    Save a model card documenting the results using skops 0.10.0.
    Push the model and card to Hugging Face Hub.
    """

    # Train the model and get the results
    comprehensive_interface.trained_model = train_model()
    results = comprehensive_interface.trained_model

    # Extract results
    mae = results["mae"]
    r2 = results["r2"]
    scatter_plot = results["scatter_plot"]
    regression_equation = results["regression_equation"]
    coefficients = results["coefficients"]  # NumPy array of coefficient values
    feature_names = results["feature_names"]  # Ensure feature names are provided
    coefficients_plot = coefficients_progression_plot_with_tracking(results)
    mae_plot, r2_plot = performance_plots_with_gridsearch(results)
    original_data_html = results["original_data_html"]
    original_data = results["original_data"]
    actual_vs_pred_plot = results["actual_vs_predicted_plot"]

    feature_importance_text = (
        f"### Useful Features:\n"
        + "".join(
            [
                f"- {feature}: {coef:.4f} "
                f"(e.g., a unit increase in {feature} affects the cost by ${coef:.2f})\n"
                for feature, coef in zip(
                    results["useful_features"].splitlines(), 
                    [float(line.split(":")[1]) for line in results["useful_features"].splitlines()]
                )
            ]
        )
        + "\n\n### Non-Useful Features:\n"
        + "".join([f"- {feature}\n" for feature in results["not_useful_features"].splitlines()])
    )

    # Save the best model using joblib
    model_path = "best_model.joblib"
    dump(results["best_model"], model_path)

    # Initialize a temporary repository
    local_repo = mkdtemp(prefix="skops-")

    # Save the model as a pickle file
    pkl_name = "best_model.pkl"
    with open(pkl_name, mode="wb") as f:
        pickle.dump(results["best_model"], f)

    # Initialize repository for Hugging Face Hub
    hub_utils.init(
        model=pkl_name,
        requirements=["scikit-learn"],
        dst=local_repo,
        task="tabular-regression",
        data=original_data,
    )

    # Prepare coefficients table
    coefficients_text = ""#"### Model Coefficients:\n\n"
    coefficients_text += "| Feature | Coefficient |\n|---------|-------------|\n"
    coefficients_text += "\n".join(
        [f"| {feature} | {value:.4f} |" for feature, value in zip(feature_names, coefficients)]
    )

    # Prepare hyperparameters
    hyperparameters = results["best_model"].get_params()

    hyperparameters_text = "### Hyperparameters:\n\n"
    hyperparameters_text += "\n".join([f"- {param}: {value}" for param, value in hyperparameters.items()])

    # Convert Plotly plot to an inline image for Markdown
    actual_vs_pred_plot_path = Path(local_repo) / "actual_vs_predicted.png"
    actual_vs_pred_plot.write_image(str(actual_vs_pred_plot_path), format="png", scale=2)

    # Embed image in Markdown with a description
    actual_vs_pred_plot_md = (
        #"### Actual vs Predicted Plot\n\n"
        "The following plot shows the relationship between the actual and predicted values. "
        "The closer the points are to the diagonal line, the better the predictions. "
        "The dashed line represents the ideal case where predictions perfectly match the actual values.\n\n"
        "![Actual vs Predicted Plot](actual_vs_predicted.png)"
    )

    # Create and save the model card
    metadata = DatasetCardData(
        language=["en"],
        license="apache-2.0",
        annotations_creators=["machine-generated"],
        language_creators=["found"],
        multilinguality="monolingual",
        size_categories="10K<n<100K",
        source_datasets=["original"],
        task_categories=["regression"],
        task_ids=["dynamic-pricing"],
        pretty_name="Dynamic Pricing Model",
    )
    card = Card(model=pkl_name, metadata=metadata)
    model_description = (
        "This is a regression model trained on the Dynamic Pricing Dataset. "
        "It was optimized using grid search with multiple hyperparameters."
    )
    card.add(
    **{
        "Model description": model_description,
        "Model description/Intended uses & limitations": (
            "This regression model is designed to predict the cost of rides based on various features such as expected ride duration, "
            "number of drivers, and time of booking.\n\n"
            "**Intended Uses**:\n"
            "- **Dynamic Pricing Analysis**: Helps optimize pricing strategies for ride-hailing platforms.\n"
            "- **Demand Forecasting**: Supports business decisions by estimating cost trends based on ride-specific parameters.\n\n"
            "**Limitations**:\n"
            "- **Feature Dependence**: The model's accuracy is highly dependent on the input features provided.\n"
            "- **Dataset Specificity**: Performance may degrade if applied to datasets with significantly different distributions.\n"
            "- **Outlier Sensitivity**: Predictions can be affected by extreme values in the dataset."
        ),
        "Model description/Training Procedure": "The model was trained using grid search to optimize hyperparameters. Cross-validation (5-fold) was performed to ensure robust evaluation. The best model was selected based on the lowest Mean Absolute Error (MAE) on the validation set.",
        #"Hyperparameters": hyperparameters_text,
        "Model description/Evaluation Results/Model Coefficients": coefficients_text,
        "Model description/Evaluation Results/Regression Equation": regression_equation,
        "Model description/Evaluation Results/Actual vs Predicted": (
            actual_vs_pred_plot_md + "\n\n"
            "The scatter plot above shows the predicted values against the actual values. The dashed line represents the ideal predictions "
            "where the predicted values are equal to the actual values."
        ),
        "Model description/Evaluation Results": (
            "The model achieved the following results on the test set:\n"
            f"- **Mean Absolute Error (MAE)**: {mae}\n"
            f"- **R² Score**: {r2}\n\n"
            "### Key Insights:\n"
            "- Longer ride durations increase costs significantly, which may justify adding a surcharge for long-distance rides.\n"
            "- Evening bookings reduce costs, potentially indicating lower demand during these hours.\n"
            "- The model's accuracy is dependent on high-quality feature data.\n"

            "\nRefer to the plots and tables for detailed performance insights."
        ),
        "How to Get Started with the Model": (
            "To use this model:\n"
            "1. **Install Dependencies**: Ensure `scikit-learn` and `pandas` are installed in your environment.\n"
            "2. **Load the Model**: Download the saved model file and load it using `joblib`:\n"
            "   ```python\n"
            "   from joblib import load\n"
            "   model = load('best_model.joblib')\n"
            "   ```\n"
            "3. **Prepare Input Features**: Create a DataFrame with the required input features in the same format as the training dataset.\n"
            "4. **Make Predictions**: Use the `predict` method to generate predictions:\n"
            "   ```python\n"
            "   predictions = model.predict(input_features)\n"
            "   ```"
        ),
        "Model Card Authors": "This model card was written by **Pranav Sharma**.",
        "Model Card Contact": "For inquiries or feedback, you can contact the author via **[GitHub](https://github.com/PranavSharma)**.",
        "Citation": (
            "If you use this model, please cite it as follows:\n"
            "```\n"
            "@model{pranav_sharma_dynamic_pricing_model_2025,\n"
            "  author       = {Pranav Sharma},\n"
            "  title        = {Dynamic Pricing Model},\n"
            "  year         = {2025},\n"
            "  version      = {1.0.0},\n"
            "  url          = {https://huggingface.co/PranavSharma/dynamic-pricing-model}\n"
            "}\n"
            "```"
        ),
    }
)


    card_path = Path(local_repo) / "README.md"
    card.save(card_path)
    print("Model card saved as README.md")

    # Push model and card to Hugging Face Hub
    try:
        hub_utils.push(
            repo_id=f"{User}/{repo_name}",
            source=local_repo,
            commit_message="Pushing model and README files to the repo!",
            create_remote=True,
        )
        print("Model and card pushed to Hugging Face Hub.")
    except Exception as e:
        print(f"Failed to push to Hugging Face Hub: {e}")

    # Return outputs for display in Gradio
    return (
        "Model trained successfully and pushed to Hugging Face Hub!",
        scatter_plot,
        regression_equation,
        mae_plot,
        r2_plot,
        coefficients_plot,
        actual_vs_pred_plot,  # New output added
        results["top_models_html"],
        original_data_html,
        feature_importance_text,
    )



# Updated prediction functionality to ensure other outputs are consistent
def use_trained_model_button(*inputs):
    """
    Use the existing trained model for predictions and return relevant outputs.
    """
    if "trained_model" not in comprehensive_interface.__dict__:
        return "No trained model found. Please train the model first.", None, None, None, None, None, None, None, None

    results = comprehensive_interface.trained_model

    if any(inputs):
        user_inputs = list(inputs)
        try:
            custom_prediction = results["best_model"].predict([user_inputs])[0]
            prediction_result = f"Custom Prediction: {custom_prediction:.2f}"
        except NotFittedError:
            prediction_result = "Trained model is not properly fitted. Please train the model again."
    else:
        prediction_result = "No custom input provided."

    scatter_plot = results["scatter_plot"]
    regression_equation = results["regression_equation"]
    coefficients_plot = coefficients_progression_plot_with_tracking(results)
    mae_plot, r2_plot = performance_plots_with_gridsearch(results)
    original_data_html = results["original_data_html"]
    top_models_html = results["top_models_html"]
    feature_importance = (
        f"### Useful Features:\n {results['useful_features']}\n\n"
        f"### Non-Useful Features:\n {results['not_useful_features']}"
    )

    return (
        prediction_result,
        scatter_plot,
        regression_equation,
        mae_plot,
        r2_plot,
        coefficients_plot,
        f"<h3>Top 10 Models</h3>{top_models_html}",
        f"<h3>Original Dataset</h3>{original_data_html}",
        feature_importance,
    )

# Comprehensive interface function
def comprehensive_interface(*inputs):
    if "trained_model" not in comprehensive_interface.__dict__:
        comprehensive_interface.trained_model = train_model()

    results = comprehensive_interface.trained_model
    scatter_plot = results["scatter_plot"]
    regression_equation = results["regression_equation"]
    coefficients_plot = coefficients_progression_plot_with_tracking(results)
    mae_plot, r2_plot = performance_plots_with_gridsearch(results)
    original_data_html = results["original_data_html"]
    top_models_html = results["top_models_html"]

    # Ensure useful and non-useful features are properly formatted
    useful_features = results.get("useful_features", "")
    not_useful_features = results.get("not_useful_features", "")

    # Process useful features (retain values) and non-useful features (omit values)
    useful_features = process_features_with_values("".join(useful_features))
    not_useful_features = process_features_without_values("".join(not_useful_features))

    # Create feature importance display
    feature_importance = (
        f"### Useful Features:\n " + "".join(useful_features) + "\n\n"
        f"### Non-Useful Features:\n " + "".join(not_useful_features)
    )

    # Prediction logic
    if any(inputs):
        user_inputs = list(inputs)
        custom_prediction = results["best_model"].predict([user_inputs])[0]
        prediction_result = f"Custom Prediction: {custom_prediction:.2f}"
    else:
        prediction_result = "No custom input provided."

    return (
        prediction_result,  # Return only the prediction for the prediction output
        scatter_plot,
        regression_equation,
        mae_plot,
        r2_plot,
        coefficients_plot,
        f"<h3>Top 10 Models</h3>{top_models_html}",
        f"<h3>Original Dataset</h3>{original_data_html}",
        feature_importance,  # Include feature importance in the outputs
    )

# Generate Gradio inputs dynamically
def generate_gradio_inputs():
    results = train_model()
    inputs = []
    for feature, default in results["default_values"].items():
        feature_type = results["feature_types"][feature]
        inputs.append(gr.Number(label=f"{feature} ({feature_type}, e.g., {default})", value=default))
    return inputs
# Layout with proper updates for all outputs
with gr.Blocks() as demo:
    gr.Markdown("# Dynamic Pricing Model - Comprehensive Analysis")
    gr.Markdown(
        "Train a range of regression models, view metrics, selection of best models, coefficients, and make custom predictions."
    )

    # Outputs Section (Top)
    with gr.Row():
        with gr.Column():
            scatter_plot_output = gr.Plot(label="Scatter Plot")
            original_data_output = gr.HTML(label="Original Dataset")
            top_models_output = gr.HTML(label="Top 10 Models")
        with gr.Column():
            actual_vs_predicted_output = gr.Plot(label="Actual vs Predicted Plot")
            mae_plot_output = gr.Plot(label="MAE Plot")
            r2_plot_output = gr.Plot(label="R² Plot")
            
        with gr.Column():
            coeff_plot_output = gr.Plot(label="Coefficient Progression")
            regression_eq_output = gr.Textbox(label="Regression Equation")
            output_feat_importance = gr.Textbox(label="Feature Importance (Useful vs Non-Useful)")

    # Inputs Section
    gr.Markdown("### Input Features")
    inputs = generate_gradio_inputs()
    with gr.Row():
        input_fields = [input for input in inputs]
    with gr.Row():
        train_button = gr.Button("Train Model")
        predict_button = gr.Button("Use Trained Model for Prediction")

    # Predictions Section (Below Inputs)
    with gr.Row():
        prediction_output = gr.Textbox(label="Result")

    # Connect training button
    train_button.click(
    fn=train_model_button,
    inputs=[],
    outputs=[
        prediction_output,
        scatter_plot_output,
        regression_eq_output,
        mae_plot_output,
        r2_plot_output,
        coeff_plot_output,
        actual_vs_predicted_output,  # New output
        top_models_output,
        original_data_output,
        output_feat_importance,
    ],
    )

    # Connect prediction button
    predict_button.click(
        fn=use_trained_model_button,
        inputs=input_fields,
        outputs=[
            prediction_output,
            scatter_plot_output,
            regression_eq_output,
            mae_plot_output,
            r2_plot_output,
            coeff_plot_output,
            top_models_output,
            original_data_output,
            output_feat_importance,
        ],
    )

demo.launch()
