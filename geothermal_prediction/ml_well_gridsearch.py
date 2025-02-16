import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from itertools import product
from collections import defaultdict
import warnings
import joblib
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt  # Ensure matplotlib is imported

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    variance = np.var(y_true)
    return mse / variance if variance != 0 else np.nan

def compute_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage

def compute_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def compute_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def compute_ensemble_metric(y_true, y_pred):
    mae = compute_mae(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return mae + abs(bias)

class MLFramework:
    def __init__(self, data, target_columns, group_column, feature_columns, depth_column, metric='Ensemble_Metric'):
        """
        Initialize the MLFramework.

        Parameters:
            data (pd.DataFrame): Dataset containing features, targets, and groups.
            target_columns (list): List of target columns to predict.
            group_column (str): Column name representing the group (e.g., Well_ID).
            feature_columns (list): List of feature columns.
            depth_column (str): Column name representing depth (e.g, Depth).
            metric (str): Evaluation metric to use. Defaults to 'Ensemble_Metric'.
        """
        self.data = data
        self.target_columns = target_columns
        self.group_column = group_column
        self.feature_columns = feature_columns
        self.depth_column = depth_column  

        # Define available metrics with their computation functions, optimization direction, and type
        self.metric_options = {
            'NMSE': {'func': compute_nmse, 'maximize': False, 'additive': True},
            'MAPE': {'func': compute_mape, 'maximize': False, 'additive': True},
            'MAE': {'func': compute_mae, 'maximize': False, 'additive': True},
            'R2': {'func': compute_r2, 'maximize': True, 'additive': False},
            'Ensemble_Metric': {'func': compute_ensemble_metric, 'maximize': False, 'additive': True},
        }


        # Validate the selected metric
        if metric not in self.metric_options:
            raise ValueError(f"Invalid metric '{metric}'. Choose from {list(self.metric_options.keys())}.")

        self.metric = metric
        self.metric_func = self.metric_options[self.metric]['func']
        self.metric_maximize = self.metric_options[self.metric]['maximize']

    def preprocess_data(self, model_name):
        """
        Preprocess the data for a specific model.

        Parameters:
            model_name (str): Name of the model (e.g., 'SVR', 'XGBRegressor').

        Returns:
            pd.DataFrame: Preprocessed data with reset index.
        """
        data = self.data.copy()

        # Drop rows with missing target values
        data = data.dropna(subset=self.target_columns)

        # Handle missing features based on model
        models_with_native_missing_handling = ['XGBRegressor', 'LGBMRegressor']
        if model_name not in models_with_native_missing_handling:
            data = data.dropna(subset=self.feature_columns)

        # Reset index to ensure alignment with iloc
        data = data.reset_index(drop=True)

        return data

    def custom_hyperparameter_search(self, model, param_grid, group_well_cv, preprocessed_data):
        """
        Perform a custom hyperparameter search.

        Parameters:
            model: The machine learning model instance.
            param_grid (dict): Dictionary specifying hyperparameter grid.
            group_well_cv: Cross-validation splitter for inner loop.
            preprocessed_data (pd.DataFrame): Preprocessed data for the current model.

        Returns:
            dict: Nested dictionary mapping hyperparameter combinations to their aggregated performance per target.
        """
        # Generate all possible hyperparameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))
        
        # Initialize a nested performance dictionary
        # { combination_tuple: {target1: metric1, target2: metric2, ...}, ... }
        performance_dict = defaultdict(dict)

        # Iterate over each hyperparameter combination (Outer Loop)
        for combination in all_combinations:
            params = dict(zip(param_names, combination))
            print(f"\nEvaluating Hyperparameters: {params}")

            # Initialize dictionaries to store metrics
            target_metric_sum = defaultdict(float)    # For additive metrics
            target_sample_sum = defaultdict(int)      # Number of samples per target
            target_y_true = defaultdict(list)         # For non-additive metrics
            target_y_pred = defaultdict(list)         # For non-additive metrics

            # Iterate over each fold (Inner Loop)
            for fold_idx, (train_idx, test_idx) in enumerate(group_well_cv.split(preprocessed_data, groups=preprocessed_data[self.group_column]), 1):
                # Split features and targets for training and testing
                X_train = preprocessed_data.iloc[train_idx][self.feature_columns]
                y_train = preprocessed_data.iloc[train_idx][self.target_columns]
                X_test = preprocessed_data.iloc[test_idx][self.feature_columns]
                y_test = preprocessed_data.iloc[test_idx][self.target_columns]

                for target in self.target_columns:
                    # Define the pipeline
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', clone(model))
                    ])

                    # Set hyperparameters
                    pipeline.set_params(**{f'model__{k}': v for k, v in params.items()})

                    # Fit the model
                    pipeline.fit(X_train, y_train[target])

                    # Predict on the test set
                    y_pred = pipeline.predict(X_test)

                    # Determine if the metric is additive
                    metric_info = self.metric_options[self.metric]
                    is_additive = metric_info['additive']

                    if is_additive:
                        # Compute metric
                        metric_value = self.metric_func(y_test[target], y_pred)
                        n_samples = len(y_test)
                        # Accumulate weighted sum
                        target_metric_sum[target] += metric_value * n_samples
                        target_sample_sum[target] += n_samples
                        print(f"Fold {fold_idx}, Target {target}, {self.metric}: {metric_value:.4f} (n={n_samples})")
                    else:
                        # Collect all true and predicted values
                        target_y_true[target].extend(y_test.tolist())
                        target_y_pred[target].extend(y_pred.tolist())
                        print(f"Fold {fold_idx}, Target {target}, {self.metric}: Collected predictions (n={len(y_test)})")

            # After all folds, compute aggregated metric per target
            for target in self.target_columns:
                metric_info = self.metric_options[self.metric]
                is_additive = metric_info['additive']
                if is_additive:
                    if target_sample_sum[target] > 0:
                        avg_metric = target_metric_sum[target] / target_sample_sum[target]
                    else:
                        avg_metric = np.nan
                    performance_dict[tuple(combination)][target] = avg_metric
                    print(f"Average {self.metric} for {params} on target '{target}': {avg_metric:.4f}")
                else:
                    if len(target_y_true[target]) > 0:
                        avg_metric = self.metric_func(target_y_true[target], target_y_pred[target])
                    else:
                        avg_metric = np.nan
                    performance_dict[tuple(combination)][target] = avg_metric
                    print(f"Aggregated {self.metric} for {params} on target '{target}': {avg_metric:.4f}")

        return performance_dict


    def select_best_hyperparameters(self, performance_dict, param_grid):
        """
        Select the best hyperparameters based on aggregated performance per target.

        Parameters:
            performance_dict (dict): Nested dictionary mapping hyperparameter combinations to their performance per target.
            param_grid (dict): Original hyperparameter grid.

        Returns:
            dict: Dictionary mapping each target to its best hyperparameter combination and corresponding avg_metric.
        """
        best_params_per_target = {}

        # Iterate over each target to find the best hyperparameters
        for target in self.target_columns:
            # Initialize variables to track the best combination
            best_combination = None
            best_metric = float('-inf') if self.metric_maximize else float('inf')

            # Iterate through all hyperparameter combinations
            for combination, target_metric_dict in performance_dict.items():
                current_metric = target_metric_dict[target]
                if self.metric_maximize:
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_combination = combination
                else:
                    if current_metric < best_metric:
                        best_metric = current_metric
                        best_combination = combination

            # Convert tuple back to parameter dictionary
            param_names = list(param_grid.keys())
            best_params = dict(zip(param_names, best_combination))

            best_params_per_target[target] = {
                'best_params': best_params,
                'avg_metric': best_metric
            }

            print(f"Best Hyperparameters for target '{target}': {best_params} with {self.metric}: {best_metric:.4f}")

        return best_params_per_target

    def train_final_model(self, model, best_params, target, preprocessed_data):
        """
        Train the final model on the entire dataset using the best hyperparameters.

        Parameters:
            model: The machine learning model instance.
            best_params (dict): Best hyperparameters.
            target (str): Target column.
            preprocessed_data (pd.DataFrame): Preprocessed data for training.

        Returns:
            Pipeline: Trained pipeline.
        """
        # Define the pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', clone(model))
        ])

        # Set hyperparameters
        pipeline.set_params(**{f'model__{k}': v for k, v in best_params.items()})

        # Split features and target
        X = preprocessed_data[self.feature_columns]
        y = preprocessed_data[target]

        # Fit the pipeline on the entire dataset
        pipeline.fit(X, y)

        return pipeline

    def collect_predictions(self, combined_results, models_with_params, num_splits):
        """
        Collect detailed predictions for each model and target using the best hyperparameters.

        Parameters:
            combined_results (dict): Combined results from hyperparameter tuning containing best_params and avg_metric for each model and target.
            models_with_params (dict): Dictionary where keys are model names and values are tuples of (model_instance, param_grid).
            num_splits (int): Number of cross-validation splits.

        Returns:
            pd.DataFrame: Dataset containing well_id, depth, y_true and y_pred for each target, and model for each prediction.
        """
        # Initialize GroupKFold inside the method
        group_well_cv = GroupKFold(n_splits=num_splits)

        # Initialize a list to collect prediction records
        prediction_records = []

        # Iterate over each model and target in combined_results
        for model_name, targets in combined_results.items():
            for target, details in targets.items():
                best_params = details['best_params']
                print(f"\nCollecting predictions for Model: {model_name}, Target: {target} with Best Params: {best_params}")

                # Retrieve the original model instance
                model_instance, _ = models_with_params[model_name]

                # Preprocess data for the current model
                preprocessed_data = self.preprocess_data(model_name)

                # Initialize the pipeline with best hyperparameters
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', clone(model_instance))
                ])

                # Set hyperparameters
                pipeline.set_params(**{f'model__{k}': v for k, v in best_params.items()})

                # Perform cross-validation to collect predictions
                for fold_idx, (train_idx, test_idx) in enumerate(group_well_cv.split(preprocessed_data, groups=preprocessed_data[self.group_column]), 1):
                    # Split features and targets for training and testing
                    X_train = preprocessed_data.iloc[train_idx][self.feature_columns]
                    y_train = preprocessed_data.iloc[train_idx][target]
                    X_test = preprocessed_data.iloc[test_idx][self.feature_columns]
                    y_test = preprocessed_data.iloc[test_idx][target]
                    well_ids = preprocessed_data.iloc[test_idx][self.group_column]  
                    depths = preprocessed_data.iloc[test_idx][self.depth_column]

                    # Fit the pipeline on the training data
                    pipeline.fit(X_train, y_train)

                    # Predict on the test set
                    y_pred = pipeline.predict(X_test)

                    # Collect predictions
                    for i in range(len(X_test)):
                        record = {
                            'Well_ID': well_ids.iloc[i],
                            'Depth': depths.iloc[i],
                            f'{target}_y_true': y_test.iloc[i],
                            f'{target}_y_pred': y_pred[i],
                            'Model': model_name
                        }
                        prediction_records.append(record)
                        
                    print(f"Fold {fold_idx}, Model {model_name}, Target {target}: Collected predictions for {len(X_test)} samples.")

        # After collecting all records, convert to DataFrame
        predictions_df = pd.DataFrame(prediction_records)
        
        # Reshape the DataFrame
        melted_df = predictions_df.melt(
            id_vars=['Well_ID', 'Depth', 'Model'],
            value_vars=[f'{target}_y_true' for target in self.target_columns
                        ] + [
                            f'{target}_y_pred' for target in self.target_columns],
            var_name='Metric',
            value_name='Value'
        )

        # Extract 'Target' and 'Type' from 'Metric'
        melted_df[['Target', 'Type']] = melted_df['Metric'].str.extract(r'(Th_\w+)_(y_true|y_pred)')

        # Pivot back to wide format
        pivot_df = melted_df.pivot_table(
            index=['Well_ID', 'Depth', 'Model'], 
            columns=['Target', 'Type'],          
            values='Value',
            aggfunc='first'  # Use 'first' assuming one prediction per target per group
        ).reset_index()

        # Flatten MultiIndex columns
        pivot_df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in pivot_df.columns]

        return pivot_df
        
    def evaluate_models_custom_cv(self, models_with_params, num_splits=4, random_state=None):
        """
        Evaluate multiple models using the custom hyperparameter search.

        Parameters:
            models_with_params (dict): Dictionary where keys are model names and values are tuples of (model_instance, param_grid).
            num_splits (int): Number of cross-validation splits.
            random_state (int): Random state for reproducibility.

        Returns:
            dict: Combined results containing best_params and avg_metric for each model and target.
            dict: Dictionary of trained final models.
        """
        all_scores = {}
        all_best_params = {}
        final_models = {}

        # Define cross-validation splitter for inner loop (cross-validation within hyperparameter evaluation)
        group_well_cv = GroupKFold(n_splits=num_splits)

        for model_name, (model, param_grid) in models_with_params.items():
            print(f"\n{'='*20}\nEvaluating Model: {model_name}\n{'='*20}")

            # Preprocess data once for the current model
            preprocessed_data = self.preprocess_data(model.__class__.__name__)

            # Perform custom hyperparameter search
            performance_dict = self.custom_hyperparameter_search(model, param_grid, group_well_cv, preprocessed_data)

            # Select the best hyperparameters per target
            best_params_per_target = self.select_best_hyperparameters(performance_dict, param_grid)
            all_best_params[model_name] = best_params_per_target

            # Store the best metric per target
            all_scores[model_name] = {target: details['avg_metric'] for target, details in best_params_per_target.items()}

            # Train the final model for each target using the best hyperparameters
            for target in self.target_columns:
                best_params = best_params_per_target[target]['best_params']
                final_model = self.train_final_model(model, best_params, target, preprocessed_data)
                final_models[f"{model_name}_{target}"] = final_model
                print(f"Final model trained for target '{target}' with {model_name}.")

        # Combine all_best_params and all_scores into combined_results
        combined_results = {}

        for model in all_best_params:
            combined_results[model] = {}
            for target in all_best_params[model]:
                # Format the metric name to lowercase and replace spaces with underscores
                metric_key = f"avg_{self.metric.lower().replace(' ', '_')}"
                combined_results[model][target] = {
                    'best_params': all_best_params[model][target]['best_params'],
                    metric_key: all_best_params[model][target]['avg_metric']
                }

        return combined_results, final_models

    def save_final_models(self, final_models, filepath='final_models.pkl'):
        """
        Save the final trained models to disk.

        Parameters:
            final_models (dict): Dictionary of trained models.
            filepath (str): File path to save the models.
        """
        joblib.dump(final_models, filepath)
        print(f"Final models saved to {filepath}.")

    def load_final_models(self, filepath='final_models.pkl'):
        """
        Load the final trained models from disk.

        Parameters:
            filepath (str): File path to load the models from.

        Returns:
            dict: Dictionary of trained models.
        """
        final_models = joblib.load(filepath)
        print(f"Final models loaded from {filepath}.")
        return final_models


    def plot_results_predictions(self, predictions_df, targets=None, models=None, figsize=(6, 10), alpha=0.6):
        """
        Plots target predictions vs. Depth for each model and target.

        Parameters:
            predictions_df (pd.DataFrame): DataFrame containing predictions with columns like 'Well_ID', 'Depth', 'Model',
                                        'Th_Cond_y_true', 'Th_Cond_y_pred', etc.
            targets (list, optional): List of target names to plot. Defaults to self.target_columns.
            models (list, optional): List of model names to plot. Defaults to unique models in predictions_df.
            figsize (tuple, optional): Figure size for each subplot. Defaults to (6, 5).
            alpha (float, optional): Transparency level for scatter plots. Defaults to 0.6.

        Returns:
            None
        """
        # Set default targets and models if not provided
        if targets is None:
            targets = self.target_columns
        if models is None:
            models = predictions_df['Model'].unique().tolist()

        n_models = len(models)
        n_targets = len(targets)

        # Create subplots for each model and target combination
        fig, axes = plt.subplots(nrows=n_models, ncols=n_targets, figsize=(6 * n_targets, 10 * n_models), sharex=False, sharey=False)

        # Handle cases where there's only one model or one target
        if n_models == 1 and n_targets == 1:
            axes = np.array([[axes]])
        elif n_models == 1:
            axes = np.expand_dims(axes, axis=0)
        elif n_targets == 1:
            axes = np.expand_dims(axes, axis=1)

        for i, model in enumerate(models):
            for j, target in enumerate(targets):
                ax = axes[i, j]
                # Filter data for the current model and target
                filtered_data = predictions_df[(predictions_df['Model'] == model) & predictions_df[f'{target}_y_true'].notna()]

                y_true = filtered_data[f'{target}_y_true']
                y_pred = filtered_data[f'{target}_y_pred']
                depth = filtered_data['Depth']

                # Plot y_true and y_pred against Depth
                ax.scatter(y_true, depth, label='y_true', alpha=alpha)
                ax.scatter(y_pred, depth, label='y_pred', alpha=alpha)

                # Calculate Evaluation Metrics
                nmse = round(compute_nmse(y_true, y_pred), 2)
                mape = round(compute_mape(y_true, y_pred), 2)
                mae = round(compute_mae(y_true, y_pred), 2)
                r2 = round(compute_r2(y_true, y_pred), 2)

                # Set subplot title and labels
                ax.set_title(f"{model}: NMSE {nmse} - MAPE {mape}% - MAE {mae} - R2 {r2}", fontsize=10)
                ax.set_xlabel(target)
                if j == 0:
                    ax.set_ylabel("Depth (m)")
                ax.invert_yaxis()  # Invert y-axis for depth representation
                ax.legend()
                ax.grid(True)

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
           
    def plot_model_comparison(self, results_df):
        """
        Compare models by plotting residuals vs. depth and true vs. predicted values.

        Parameters:
            results_df (pd.DataFrame): DataFrame containing model results with columns:
                                    - 'Model', 'Depth', '<target>_y_true', '<target>_y_pred'.
        """
        num_models = len(results_df['Model'].unique())
        
        for target in self.target_columns:
            true_col = f"{target}_y_true"
            pred_col = f"{target}_y_pred"
            
            # Create subplots for residuals and true vs. predicted
            fig, axes = plt.subplots(2, num_models, figsize=(6 * num_models, 14),
                                    gridspec_kw={'height_ratios': [3, 1]})
            
            for idx, model_name in enumerate(results_df['Model'].unique()):
                # Residuals vs. Depth (Top Row)
                ax_residuals = axes[0, idx] if num_models > 1 else axes[0]
                
                model_data = results_df[results_df['Model'] == model_name]
                y_true = model_data[true_col]
                y_pred = model_data[pred_col]
                residuals = y_true - y_pred
                depth = model_data[self.depth_column]
                
                ax_residuals.scatter(residuals, depth, alpha=0.5, label=f"{model_name} Residuals")
                ax_residuals.axvline(0, color='red', linestyle='--', label='Zero Residual')
                ax_residuals.invert_yaxis()
                ax_residuals.set_xlabel('Residuals')
                ax_residuals.set_ylabel('Depth (m)' if idx == 0 else "")
                ax_residuals.grid(True)
                if idx == 0 or num_models > 1:
                    ax_residuals.legend(loc='upper left')
                    
                # Calculate Evaluation Metrics
                nmse = round(compute_nmse(y_true, y_pred), 2)
                mape = round(compute_mape(y_true, y_pred), 2)
                mae = round(compute_mae(y_true, y_pred), 2)
                r2 = round(compute_r2(y_true, y_pred), 2)
                ax_residuals.set_title(f"{model_name}: NMSE {nmse} - MAPE {mape}% - MAE {mae} - R2 {r2}", fontsize=10)
                
                # True vs. Predicted (Bottom Row)
                ax_true_pred = axes[1, idx] if num_models > 1 else axes[1]
                
                ax_true_pred.scatter(y_true, y_pred, alpha=0.5, label=f"{model_name}")
                ax_true_pred.plot([0, 6], [0, 6], 'r--', label='Ideal Fit')
                ax_true_pred.set_xlabel('True Values')
                ax_true_pred.set_ylabel('Predicted Values' if idx == 0 else "")
                ax_true_pred.grid(True)
                if idx == 0 or num_models > 1:
                    ax_true_pred.legend(loc='upper left')
            
            # Set title for the entire figure
            fig.suptitle(f"Model Comparison for {target}", fontsize=16, y=0.95)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the title
            plt.show()

            

    def plot_feature_importance(self, final_models):
        """
        Plots feature importance for tree-based models for each target.

        Parameters:
        - final_models (dict): Dictionary of trained models with keys as 'ModelName_Target'.
        """
        for model_key, model in final_models.items():
            model_name, model_target = model_key.split('_', 1)
            if model and hasattr(model.named_steps['model'], 'feature_importances_'):
                importances = model.named_steps['model'].feature_importances_
                feature_names = self.feature_columns
                indices = np.argsort(importances)
                
                plt.figure(figsize=(8, 6))
                plt.barh(range(len(importances)), importances[indices], align='center')
                plt.yticks(range(len(importances)), np.array(feature_names)[indices])
                plt.xlabel('Feature Importance')
                plt.title(f'Feature Importances for {model_name} predicting {model_target}')
                plt.show()


    def plot_coefficients(self, final_models):
        """
        Plots model coefficients for linear models for each target.

        Parameters:
        - final_models (dict): Dictionary of trained models with keys as 'ModelName_Target'.
        """
        for model_key, model in final_models.items():
            model_name, model_target = model_key.split('_', 1)
            
            if model and hasattr(model.named_steps['model'], 'coef_'):
                coefficients = model.named_steps['model'].coef_
                feature_names = self.feature_columns
                indices = np.argsort(np.abs(coefficients))
                
                plt.figure(figsize=(8, 6))
                plt.barh(range(len(coefficients)), coefficients[indices], align='center')
                plt.yticks(range(len(coefficients)), np.array(feature_names)[indices])
                plt.xlabel('Coefficient Value')
                plt.title(f'Coefficients for {model_name} predicting {model_target}')
                plt.show()
                
        
