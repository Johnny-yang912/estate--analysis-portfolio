# estate-analysis-portfolio
# 🏠 房地產交易金額預測模型 (Real Estate Forecasting API)

## 🔎專案簡介
本專案建立一個 **房地產交易金額預測系統**，從資料前處理、特徵工程、建模、調參到 API 部署，完整展現資料科學專案流程，最終模型分數達 96 分。
此模型可用於房市趨勢預測、投資與規劃決策；同時也代表我能將這套流程應用到金融、零售、醫療等其他主題的模型建構。

## 核心亮點：
1. **Pipeline** — 將特徵工程、模型建構、調參流程模組化，避免資料洩漏並確保可重現性  
2. **Lag / Rolling 時間序列工程** — 捕捉跨期資訊與移動平均特徵  
3. **多模型比較** — 線性回歸、隨機森林、XGBoost baseline → 最終選擇 XGBoost  
4. **調參** — 使用 `RandomizedSearchCV` + `TimeSeriesSplit` 找到最佳超參數  
5. **部署** — 使用 FastAPI + Uvicorn 建立 REST API，可輸入新資料並即時預測  
6. **可利用工具包** — `lag_rolling_tools`、`scikit-learn`、`xgboost`、`fastapi`、`joblib`  

## 📎 專案連結

- **完整 Notebook** 👉 [房地產預測.ipynb](房地產預測.ipynb)  
- **工具包**  
  - Lag/Rolling 特徵生成 👉 [lag_rolling_tools.py](lag_rolling_tools.py)  
  - API 部署服務 👉 [API_deploy/](API_deploy)  

---

## 📂 專案結構

```
.
├── API_deploy/ # FastAPI API 服務 (app.py)
├── models/ # 已訓練好的模型 (joblib 檔案, 已忽略大檔)
├── train/ # 原始訓練資料 (已忽略大檔)
├── lag_rolling_tools.py # 自製 lag/rolling 特徵工程工具
├── kaggle_real_estate_dataset_builder.py # 資料構建腳本
├── requirements.txt # 套件需求
├── to_predict.csv # 範例輸入
├── 房地產預測.ipynb # 建模過程 Jupyter Notebook
├── README.md # 專案說明
└── .gitignore # 忽略規則
```


---

## 📊 專案流程

1. **資料合併**

2.**建立baseline** ->原始特徵無訊號   

3. **特徵工程 (Lag / Rolling)**  
   - 建立 **滯後 (Lag)** 特徵，例如 `y(t-1)`、`y(t-3)`  
   - 建立 **移動平均 (Rolling)** 特徵，例如 `MA(3)`、`MA(6)`  

4. **建模與比較**
   ***三種模型+兩種特徵處理方式比較***  
   - 線性回歸/隨機森林/XGBboost  
   - 全部補0/刪除暖機段

5. **判斷特徵重要性**
  -lag/rolling外的參數皆為雜訊->僅保留lag/rolling

6. **調參 (Hyperparameter Tuning)**  
   - 使用 `RandomizedSearchCV`，搜尋樹深度、學習率、子樣本比例等  
   - 時間序列交叉驗證 (`TimeSeriesSplit`) 確保符合時間依賴性

7. **部署**  
   - 使用 **FastAPI** 建立 RESTful API  
   - 可透過 `/predict` 端點上傳 CSV/JSON → 回傳預測結果  
   - 本地啟動：  
     ```bash
     uvicorn API_deploy.app:app --reload --port 8000
     ```
***注:因本地算力關係，無法進行更精確的調參***

---

## 📈 模型表現

- 初始 Baseline: **約0分**  
- 加入特徵工程後: **約89分**  
- 最終最佳調參 XGBoost: **約96分**

---

## 🚀 使用方式

1. **安裝套件**
   ```bash
   pip install -r requirements.txt
```
啟動 API

uvicorn API_deploy.app:app --reload --port 5001
```
使用 Swagger UI: http://127.0.0.1:5001/docs

或用 curl：
```
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sector":"A", "year_month":"2022-08", ... }'
```
## 👉 資料來源
**Kaggle**:Real Estate Demand Prediction

**注:本專案僅供學術與技術展示，本模型成果不可直接應用於現實房地產市場。**
