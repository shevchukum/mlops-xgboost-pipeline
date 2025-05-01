''' Trains and re-trains model, logs and promote the model to production '''

import pandas as pd
import numpy as np
import glob
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
import mlflow
import mlflow.xgboost
from datetime import datetime
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import time

# --- SETTINGS ---
DATA_FOLDER = "new_data"
TARGET = "Divorce"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "xgboost-divorce-prediction"

# --- Load and combine data ---
def load_all_data():
    master = pd.read_csv(os.path.join(DATA_FOLDER, "master_data.csv"))
    weekly_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "week_*.csv")))
    weekly = [pd.read_csv(f) for f in weekly_files]
    all_data = pd.concat([master] + weekly, ignore_index=True)

    # Assign float to all clumns to allow Nan's
    all_data[all_data.select_dtypes(include='int').columns] = all_data.select_dtypes(include='int').astype('float64')
    weekly = [df.astype({col: 'float' for col in df.select_dtypes(include='int').columns}) for df in weekly]
    
    return all_data, weekly[-4:] if len(weekly) >= 4 else weekly

# --- Calculate PSI for target variable ---
def calculate_psi(base_series, new_series, buckets=10):
    def get_bins(series):
        counts, bin_edges = np.histogram(series, bins=buckets)
        return bin_edges

    base_percents, _ = np.histogram(base_series, bins=get_bins(base_series), density=True)
    new_percents, _ = np.histogram(new_series, bins=get_bins(base_series), density=True)

    base_percents += 1e-5
    new_percents += 1e-5

    psi = np.sum((base_percents - new_percents) * np.log(base_percents / new_percents))
    return psi

# --- Apply sample weighting based on PSI ---
def apply_sample_weights(X, y, recent_df, psi):
    weights = compute_sample_weight("balanced", y)
    if psi > 0.1:
        idx_recent_divorce = recent_df[recent_df[TARGET] == 1].index
        weight_boost = 1 + min(psi, 1.0) * 5  # up to 5x weight
        for idx in idx_recent_divorce:
            if idx < len(weights):
                weights[idx] *= weight_boost
    return weights

# --- Train and log to MLflow ---
def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_data, last_4weeks = load_all_data()
    recent_df = pd.concat(last_4weeks, ignore_index=True)

    y = all_data[TARGET]
    X = all_data.drop(columns=[TARGET])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    psi = calculate_psi(base_series=all_data[TARGET], new_series=recent_df[TARGET])
    weights = apply_sample_weights(X_train, y_train, recent_df, psi)

    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200]
    }

    model = xgb.XGBClassifier(eval_metric='logloss')
    grid_search = GridSearchCV(model, param_grid, scoring='average_precision', cv=3, verbose=1)
    grid_search.fit(X_train, y_train, sample_weight=weights)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict_proba(X_val)[:, 1]
    avg_precision = average_precision_score(y_val, y_pred)
    best_model.save_model("xgb_model.json")

    with mlflow.start_run():
        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("average_precision", avg_precision)
        
        # Log model with signature
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.xgboost.log_model(
            best_model, 
            artifact_path="model", 
            signature=signature,
            model_format="ubj"
        )
    
        # Prepare model registration
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        client = MlflowClient()
        
        try:
            # Try to register model if not registered already
            new_model_version = mlflow.register_model(
                model_uri=model_uri, 
                name=EXPERIMENT_NAME
            )

            # Verify status
            if new_model_version.status != "READY":
                raise MlflowException(f"Version {new_version.version} failed to reach READY state")

            # Get current production versions
            current_prod_versions = client.search_model_versions(
                f"name='{EXPERIMENT_NAME}' and tags.deployment_status='Production'"
            )
            
            # Determine promotion
            promote = True
            if current_prod_versions:
                prod_run = client.get_run(current_prod_versions[0].run_id)
                old_metric = prod_run.data.metrics.get("average_precision", 0)
                if avg_precision <= old_metric:
                    promote = False
            
            if promote:
                # Set tag for new version
                client.set_model_version_tag(
                    name=EXPERIMENT_NAME,
                    version=new_model_version.version,
                    key="deployment_status",
                    value="Production"
                )

                # Archive previous versions
                for version in current_prod_versions:
                    client.set_model_version_tag(
                        name=EXPERIMENT_NAME,
                        version=version.version,
                        key="deployment_status",
                        value="Archived"
                    )
            
            # Add additional tags
            mlflow.set_tag("retrain_date", datetime.now().strftime("%Y-%m-%d"))
            mlflow.set_tag("retrain_batch_week", datetime.now().strftime("%V")) 

        except Exception as e:
            print(f"Failed to register model: {str(e)}")
            raise

        # Clean-up old versions and keep only last 5 versions
        all_versions = client.search_model_versions(f"name='{EXPERIMENT_NAME}'")
        if len(all_versions) > 5:
            versions_to_delete = sorted(all_versions, key=lambda x: int(x.version))[:-5]
            for v in versions_to_delete:
                client.delete_model_version(EXPERIMENT_NAME, v.version)

if __name__ == "__main__":
    train()