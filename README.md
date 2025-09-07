# estate--analysis-portfolio
# 房地產交易資料建模實作

## 📌 專案簡介
本專案以 **Kaggle 房地產預測競賽** 釋出的公開資料集為例，實作從 **資料整合 → 特徵工程 → 建模 → 驗證 → 提交檔產出** 的完整流程。  
專案定位為 **技能展示 (Skill Demonstration)**，重點在於展現資料處理與機器學習建模能力，而非比賽排名。  

---

## 📎 專案檔案
- [完整 Notebooks: 房地產預測.ipynb](./房地產預測.ipynb)  
- [資料合併: kaggle_real_estate_dataset_builder.py](./kaggle_real_estate_dataset_builder.py)  

---

## 📂 資料來源
- 來源：Kaggle 房地產交易預測比賽公開資料集  
- 資料內容涵蓋：
  - **新建住宅交易**（num/area/price/amount）
  - **二手房交易**
  - **土地交易**
  - **POI 資訊**（商業、教育、交通、醫療）
  - **城市經濟指標**（人口、GDP、收入、就業）
- 目標變數（Target）：  
  - `new_house_transaction_amount`  
  - 單位：萬元  

---

## 🛠️ 技術重點
1. **資料整合**  
   - 多來源 CSV 合併（人口、經濟、房地產、POI）  
   - 保留唯一目標欄位，避免 target 洩漏  

2. **特徵工程**  
   - 時間特徵轉換：`month_sin`, `month_cos`  
   - 時間序列衍生：`lag1`, `lag3`, `lag12`, `ma3`, `ma6`  
   - 缺值處理：`SimpleImputer`  
   - 類別特徵編碼  

3. **建模流程**  
   - `scikit-learn` Pipeline 組裝  
   - `TimeSeriesSplit` 交叉驗證  
   - 自定義 MAPE 評估函數（處理 y=0 問題）  
   - 嘗試模型：
     - Linear Regression
     - Random Forest
     - XGBoost  

4. **提交檔產出**  
   - 確保輸出格式與 `sample_submission.csv` 一致  
   - `id`, `new_house_transaction_amount`  

---

## 🔍 學習收穫
- 熟悉多來源資料的整合流程，包含房地產交易、人口、經濟、POI 等多表格合併。  
- 實作時間序列特徵工程（lag, rolling, cyclical encoding），並應用於預測任務中。  
- 建立並使用 scikit-learn Pipeline，結合交叉驗證與自訂評估指標（MAPE）。  
- 體會到專案初期「明確定義目標欄位」與「確認單位一致」的重要性，確保建模方向正確。  


- **完整實戰流程演練**  
  - 從資料工程 → 特徵工程 → 模型驗證 → 生成提交檔。  
  - 熟悉了時間序列建模的常見陷阱與解法。  

---

## 🚀 後續方向
- 應用相同流程於其他公開資料集（股票分析、顧客行為分析）  
- 強化特徵選擇與模型調參（GridSearchCV, Bayesian Optimization）  
- 與商業問題結合，延伸至 **投資決策建議** 或 **房價趨勢分析**  

---

## 📎 技術棧
- **Python** (pandas, numpy)  
- **scikit-learn** (Pipeline, cross-validation, metrics)  
- **XGBoost**  
- **Matplotlib / Seaborn** (資料視覺化)  

---
