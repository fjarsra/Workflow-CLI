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
# Biarkan MLflow baca kredensial dari Environment Variable (GitHub Secrets)
# Jadi kita gak perlu hardcode user/password di sini.
DAGSHUB_URI = os.environ.get("MLFLOW_TRACKING_URI")
if not DAGSHUB_URI:
    # Fallback kalau dijalankan lokal tanpa env var
    DAGSHUB_URI = "https://dagshub.com/fjarsra/Eksperimen_SML_fjarsra.mlflow"

mlflow.set_tracking_uri(DAGSHUB_URI)

# Folder Dataset (Sesuaikan struktur folder)
DATA_DIR = "SynchronousMachine_preprocessing"

def load_data():
    if not os.path.exists(DATA_DIR):
        print(f"Dataset tidak ditemukan di {DATA_DIR}, cek path!")
        raise FileNotFoundError
    
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train.drop(columns=['I_f']), train['I_f'], test.drop(columns=['I_f']), test['I_f']

def main():
    mlflow.set_experiment("Docker_Build_Experiment")
    X_train, y_train, X_test, y_test = load_data()

    # Param Grid (Minimalis biar cepet)
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    print("Mulai Training...")
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    with mlflow.start_run() as run:
        # 1. Log Metrics & Params ke DagsHub (Biar tetep ada history)
        mlflow.log_params(best_params)
        y_pred = best_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # 2. LOG MODEL KE DAGSHUB (Standar)
        mlflow.xgboost.log_model(best_model, "model")

        # ============================================================
        # 🔥 BAGIAN KRUSIAL BUAT DOCKER OFFLINE BUILD 🔥
        # ============================================================
        
        # Nama folder tujuan (Harus sama dengan yang ada di YAML)
        local_model_path = "model_docker"
        
        # Bersihkan folder lama jika ada
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        
        # SIMPAN MODEL SECARA LOKAL (Format MLflow lengkap)
        # Ini akan bikin folder 'model_docker' yang isinya: MLmodel, conda.yaml, python_env.yaml, model.xgb
        mlflow.xgboost.save_model(best_model, local_model_path)
        print(f"✅ Model tersimpan lokal di folder: {local_model_path}")

        # Simpan Run ID ke file teks (Buat dibaca YAML)
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
        
        # ============================================================
        # GENERATE ARTEFAK GAMBAR (Buat bukti di repo)
        # ============================================================
        
        # Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            sorted_idx = best_model.feature_importances_.argsort()
            plt.figure(figsize=(10, 6))
            plt.barh(X_train.columns[sorted_idx], best_model.feature_importances_[sorted_idx])
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            plt.close()

        # Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        plt.savefig("prediction_plot.png")
        plt.close()
        
        print("✅ Selesai! Model lokal & Run ID siap.")

if __name__ == "__main__":
    main()