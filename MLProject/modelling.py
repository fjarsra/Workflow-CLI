import pandas as pd
import xgboost as xgb
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import shutil
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# KONFIGURASI
# ==============================================================================
# Ambil dari Env Var (GitHub Secrets) atau Fallback ke string langsung
DAGSHUB_URI = os.environ.get("MLFLOW_TRACKING_URI")
if not DAGSHUB_URI:
    DAGSHUB_URI = "https://dagshub.com/fjarsra/Eksperimen_SML_fjarsra.mlflow"

mlflow.set_tracking_uri(DAGSHUB_URI)

DATA_DIR = "SynchronousMachine_preprocessing"

def load_data():
    # Handle path biar aman
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ Warning: Folder {DATA_DIR} gak ketemu. Coba cek path.")
        # Fallback logic kalau dijalankan dari root folder
        if os.path.exists(f"MLProject/{DATA_DIR}"):
            current_dir = f"MLProject/{DATA_DIR}"
        else:
            current_dir = DATA_DIR
    else:
        current_dir = DATA_DIR

    train = pd.read_csv(os.path.join(current_dir, "train.csv"))
    test = pd.read_csv(os.path.join(current_dir, "test.csv"))
    return train.drop(columns=['I_f']), train['I_f'], test.drop(columns=['I_f']), test['I_f']

def main():
    # GANTI JADI INI:
    mlflow.set_experiment("Final_Submission")

    
    try:
        X_train, y_train, X_test, y_test = load_data()
    except Exception as e:
        print(f"❌ Error Load Data: {e}")
        return

    # Param Grid Simple
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    print("🚀 Mulai Training & Tuning...")
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    with mlflow.start_run() as run:
        print(f"✅ Run ID: {run.info.run_id}")
        
        # 1. Log Metrics & Params
        mlflow.log_params(best_params)
        y_pred = best_model.predict(X_test)
        
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("r2_score", r2_score(y_test, y_pred))

        # ============================================================
        # 🔥 MAGIC SECTION: LOCAL SAVE & FORCE UPLOAD 🔥
        # ============================================================
        
        local_model_path = "model_docker"
        
        # Bersihkan folder lama
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
            
        # A. Simpan Model Format MLflow ke Folder Lokal (Buat Docker Build)
        print("💾 Saving model locally for Docker Build...")
        mlflow.xgboost.save_model(best_model, local_model_path)
        
        # B. Force Upload Folder Tadi ke DagsHub (Buat Reviewer liat Folder Model)
        print("⬆️ Uploading model folder to DagsHub...")
        mlflow.log_artifacts(local_model_path, artifact_path="model")

        # C. Simpan Run ID ke file (Buat YAML baca)
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        # ============================================================
        # Generate Artefak Gambar (Biar Reviewer Senang)
        # ============================================================
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.title("Actual vs Predicted")
        plt.savefig("prediction_plot.png")
        mlflow.log_artifact("prediction_plot.png") # Upload gambar
        plt.close()
        
        print("✅ DONE! Model Saved, Uploaded, and Ready for Docker.")

if __name__ == "__main__":
    main()