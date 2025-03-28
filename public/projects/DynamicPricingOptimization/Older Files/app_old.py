import os
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
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
        template="plotly_white"
    )
    return fig

# Train the model and prepare results
def train_model():
    original_data = pd.read_csv(DATA_PATH)
    data, bool_columns = load_data()
    X = data.drop("Historical_Cost_of_Ride", axis=1)
    y = data["Historical_Cost_of_Ride"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"alpha": np.logspace(-3, 0, 4)}
    grid_search = GridSearchCV(Lasso(fit_intercept=False), param_grid, scoring="neg_mean_absolute_error", cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    feature_names = X_train.columns
    coefficients = best_model.coef_

    useful_features = [(feature, coef) for feature, coef in zip(feature_names, coefficients) if coef != 0]
    not_useful_features = [feature for feature, coef in zip(feature_names, coefficients) if coef == 0]

    equation_terms = [f"{coef:.4f} * {feature}" for feature, coef in useful_features]
    regression_equation = " + ".join(equation_terms)

    useful_features_formatted = "\n".join(
        [f"- {feature}: {coef:.4f}" for feature, coef in useful_features]
    )
    not_useful_features_formatted = "\n".join(
        [f"- {feature}" for feature in not_useful_features]
    )

    default_values, types = compute_defaults_and_types(X_train, bool_columns)

    scatter_plot = duration_vs_cost_plot(original_data)

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
        "best_alpha": grid_search.best_params_["alpha"],
        "best_model": best_model,
        "regression_equation": regression_equation,
        "scatter_plot": scatter_plot,
        "useful_features": useful_features_formatted,
        "not_useful_features": not_useful_features_formatted,
        "default_values": default_values,
        "feature_types": types,
        "original_data_html": original_data.head(10).to_html(),
    }

# Comprehensive Gradio interface
def comprehensive_interface(*inputs):
    if "trained_model" not in comprehensive_interface.__dict__:
        comprehensive_interface.trained_model = train_model()

    results = comprehensive_interface.trained_model
    scatter_plot = results["scatter_plot"]
    regression_equation = results["regression_equation"]
    mae = results["mae"]
    r2 = results["r2"]
    coefficients_plot = coefficients_progression_plot_with_tracking(results)
    mae_plot, r2_plot = performance_plots_with_gridsearch(results)
    original_data_html = results["original_data_html"]

    if any(inputs):
        user_inputs = list(inputs)
        custom_prediction = results["best_model"].predict([user_inputs])[0]
        prediction_result = f"Custom Prediction: {custom_prediction:.2f}"
    else:
        prediction_result = "No custom input provided."

    return (
        scatter_plot,
        f"<h3>Original Dataset</h3>{original_data_html}",
        f"Regression Equation:\n{regression_equation}",
        f"MAE: {mae:.4f}, R²: {r2:.4f}",
        coefficients_plot,
        mae_plot,
        r2_plot,
        f"Useful Features:\n{results['useful_features']}\n\nNot Useful Features:\n{results['not_useful_features']}",
        prediction_result
    )

# Generate Gradio inputs dynamically
def generate_gradio_inputs():
    results = train_model()
    inputs = []
    for feature, default in results["default_values"].items():
        feature_type = results["feature_types"][feature]
        inputs.append(gr.Number(label=f"{feature} ({feature_type}, e.g., {default})", value=default))
    return inputs

# Build Gradio interface
gr.Interface(
    fn=comprehensive_interface,
    inputs=generate_gradio_inputs(),
    outputs=[
        gr.Plot(label="Duration vs Cost"),
        gr.HTML(label="Original Dataset"),
        gr.Textbox(label="Regression Equation"),
        gr.Textbox(label="Model Metrics (MAE, R²)"),
        gr.Plot(label="Coefficient Progression"),
        gr.Plot(label="MAE Plot"),
        gr.Plot(label="R² Plot"),
        gr.Textbox(label="Feature Importance (Useful vs Not Useful)"),
        gr.Textbox(label="Custom Prediction"),
    ],
    title="Dynamic Pricing Model - Comprehensive Analysis",
    description="Train a Lasso regression model, view metrics, coefficients, and make custom predictions.",
).launch()

# def comprehensive_interface(*inputs):
#     if "trained_model" not in comprehensive_interface.__dict__:
#         comprehensive_interface.trained_model = train_model()

#     results = comprehensive_interface.trained_model
#     scatter_plot = results["scatter_plot"]
#     regression_equation = results["regression_equation"]
#     mae = results["mae"]
#     r2 = results["r2"]
#     coefficients_plot = coefficients_progression_plot_with_tracking(results)
#     mae_plot, r2_plot = performance_plots_with_gridsearch(results)
#     original_data_html = results["original_data_html"]
#     top_models_html = results["top_models_html"]  # Get the HTML table for top models

#     if any(inputs):
#         user_inputs = list(inputs)
#         custom_prediction = results["best_model"].predict([user_inputs])[0]
#         prediction_result = f"Custom Prediction: {custom_prediction:.2f}"
#     else:
#         prediction_result = "No custom input provided."

#     return (
#         prediction_result,
#         mae_plot,
#         r2_plot,
#         coefficients_plot,
#         f"Regression Equation:\n{regression_equation}",
#         f"<h3>Top 10 Models</h3>{top_models_html}",  # Return HTML for top models
#         #f"MAE: {mae:.4f}, R²: {r2:.4f}",
#         f"Useful Features:\n{results['useful_features']}\n\nNot Useful Features:\n{results['not_useful_features']}",
#         f"<h3>Original Dataset</h3>{original_data_html}",
#         scatter_plot,

#     )

# # Generate Gradio inputs dynamically
# def generate_gradio_inputs():
#     results = train_model()
#     inputs = []
#     for feature, default in results["default_values"].items():
#         feature_type = results["feature_types"][feature]
#         inputs.append(gr.Number(label=f"{feature} ({feature_type}, e.g., {default})", value=default))
#     return inputs

# gr.Interface(
#     fn=comprehensive_interface,
#     inputs=generate_gradio_inputs(),
#     outputs=[
#         gr.Textbox(label="Custom Prediction"),

#         gr.Plot(label="MAE Plot"),
#         gr.Plot(label="R² Plot"),
#         gr.Plot(label="Coefficient Progression"),
#         gr.Textbox(label="Regression Equation"),
#         gr.HTML(label="Top 10 Models"),  # Use HTML for top models
#         #gr.Textbox(label="Model Metrics (MAE, R²)"),
#         gr.Textbox(label="Feature Importance (Useful vs Not Useful)"),
#         gr.HTML(label="Original Dataset"),
#         gr.Plot(label="Duration vs Cost"),
#     ],
#     title="Dynamic Pricing Model - Comprehensive Analysis",
#     description="Train a range of regression models, view metrics, selection of best models, coefficients, and make custom predictions.",
# ).launch()

# import gradio as gr

# # Comprehensive interface function
# def comprehensive_interface(*inputs):
#     if "trained_model" not in comprehensive_interface.__dict__:
#         comprehensive_interface.trained_model = train_model()

#     results = comprehensive_interface.trained_model
#     scatter_plot = results["scatter_plot"]
#     regression_equation = results["regression_equation"]
#     mae = results["mae"]
#     r2 = results["r2"]
#     coefficients_plot = coefficients_progression_plot_with_tracking(results)
#     mae_plot, r2_plot = performance_plots_with_gridsearch(results)
#     original_data_html = results["original_data_html"]
#     top_models_html = results["top_models_html"]  # Get the HTML table for top models

#     if any(inputs):
#         user_inputs = list(inputs)
#         custom_prediction = results["best_model"].predict([user_inputs])[0]
#         prediction_result = f"Custom Prediction: {custom_prediction:.2f}"
#     else:
#         prediction_result = "No custom input provided."

#     return (
#         prediction_result,
#         mae_plot,
#         r2_plot,
#         coefficients_plot,
#         f"Regression Equation:\n{regression_equation}",
#         f"<h3>Top 10 Models</h3>{top_models_html}",  # Return HTML for top models
#         f"Useful Features:\n{results['useful_features']}\n\nNot Useful Features:\n{results['not_useful_features']}",
#         f"<h3>Original Dataset</h3>{original_data_html}",
#         scatter_plot,
#     )

# # Generate Gradio inputs dynamically
# def generate_gradio_inputs():
#     results = train_model()
#     inputs = []
#     for feature, default in results["default_values"].items():
#         feature_type = results["feature_types"][feature]
#         inputs.append(gr.Number(label=f"{feature} ({feature_type}, e.g., {default})", value=default))
#     return inputs

# # Three-column layout with execution trigger
# with gr.Blocks() as demo:
#     gr.Markdown("# Dynamic Pricing Model - Comprehensive Analysis")
#     gr.Markdown(
#         "Train a range of regression models, view metrics, selection of best models, coefficients, and make custom predictions."
#     )

#     # Outputs Section
#     with gr.Row():
#         with gr.Column():

#             output_mae_plot = gr.Plot(label="MAE Plot")
#             output_r2_plot = gr.Plot(label="R² Plot")
#             output_coeff_plot = gr.Plot(label="Coefficient Progression")
#         with gr.Column():
#             output_reg_eq = gr.Textbox(label="Regression Equation")
#             output_top_models = gr.HTML(label="Top 10 Models")  # Use HTML for top models
#             output_feat_importance = gr.Textbox(label="Feature Importance (Useful vs Not Useful)")
#         with gr.Column():
#             output_prediction = gr.Textbox(label="Custom Prediction")
#             output_original_data = gr.HTML(label="Original Dataset")
#             output_scatter_plot = gr.Plot(label="Duration vs Cost")

#     # Inputs Section
#     gr.Markdown("### Input Features")
#     inputs = generate_gradio_inputs()
#     with gr.Row():
#         input_fields = [input for input in inputs]
#     trigger_button = gr.Button("Run Analysis")

#     # Connect inputs and outputs to the function
#     trigger_button.click(
#         fn=comprehensive_interface,
#         inputs=input_fields,
#         outputs=[
#             output_prediction,
#             output_mae_plot,
#             output_r2_plot,
#             output_coeff_plot,
#             output_reg_eq,
#             output_top_models,
#             output_feat_importance,
#             output_original_data,
#             output_scatter_plot,
#         ],
#     )

# demo.launch()
