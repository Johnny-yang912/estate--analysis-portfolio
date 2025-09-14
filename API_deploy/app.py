# API_deploy/app.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import joblib
import os

# === 依照你的訓練設定，這三個名字請保持一致 ===
DATE_COL  = "year_month"
GROUP_COL = "sector"
Y_COL     = "amount_new_house_transactions"

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_lag_pipeline.joblib")

app = FastAPI(title="XGB-Lag Pipeline API", version="1.0")

# lazy load / 熱啟動載入一次
pipe = joblib.load(MODEL_PATH)

class Batch(BaseModel):
    rows: List[Dict]   # 每列是一個 dict（欄名: 值）

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", tags=["inference"])
def predict_json(batch: Batch):
    """
    以 JSON (list of dict) 提供資料。
    規則：
      1) 同 sector 的資料須包含足夠歷史列以產生 lag/rolling（先排序）。
      2) 歷史列需提供真實 Y_COL；待預測列 Y_COL 留空/NaN。
    """
    df = pd.DataFrame(batch.rows)

    # 基本檢查
    for col in [DATE_COL, GROUP_COL]:
        if col not in df.columns:
            return {"error": f"missing required column: {col}"}

    # 排序（確保 lag 計算正確）
    df = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    # 以整批資料做預測
    preds = pipe.predict(df)

    # 只回傳「Y 缺失」那些列的預測（常見做法：待預測月/未來期）
    if Y_COL in df.columns:
        mask = df[Y_COL].isna()
    else:
        # 若完全沒帶 Y_COL，視為全部需要預測
        mask = pd.Series([True] * len(df))

    out = df.loc[mask, [GROUP_COL, DATE_COL]].copy()
    out["prediction"] = pd.Series(preds, index=df.index).loc[mask].values

    return {
        "n_rows_input": int(len(df)),
        "n_rows_pred": int(mask.sum()),
        "predictions": out.to_dict(orient="records")
    }

@app.post("/predict-file", tags=["inference"])
async def predict_file(file: UploadFile = File(...)):
    """
    以 CSV 上傳。CSV 欄位需求與 /predict 相同。
    """
    contents = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(contents))

    for col in [DATE_COL, GROUP_COL]:
        if col not in df.columns:
            return {"error": f"missing required column: {col}"}

    df = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    preds = pipe.predict(df)
    if Y_COL in df.columns:
        mask = df[Y_COL].isna()
    else:
        mask = pd.Series([True] * len(df))

    out = df.loc[mask, [GROUP_COL, DATE_COL]].copy()
    out["prediction"] = pd.Series(preds, index=df.index).loc[mask].values

    # 也把所有列的預測附上，方便下載
    df_out = df.copy()
    df_out["prediction"] = preds
    save_name = "predictions.csv"
    df_out.to_csv(save_name, index=False)

    return {
        "n_rows_input": int(len(df)),
        "n_rows_pred": int(mask.sum()),
        "preview": out.head(10).to_dict(orient="records"),
        "download": "/predictions.csv"
    }

@app.get("/predictions.csv", include_in_schema=False)
def download_csv():
    # 簡單檔案提供（開發用）
    import fastapi.responses as R
    return R.FileResponse("predictions.csv", filename="predictions.csv")
