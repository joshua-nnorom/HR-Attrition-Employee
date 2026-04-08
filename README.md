# HR Employee Attrition Predictor — Streamlit App

A production-ready Streamlit dashboard for predicting employee attrition risk using the XGBoost model trained in `HrEmpAttrition_Optimised__2_.ipynb`.

---

## 🚀 Quick Start

### 1. Generate model files from notebook
Run every cell in your notebook. The last two cells save:
```
XG_model.pkl        ← trained XGBoost classifier
model_columns.pkl   ← list of feature column names
```
Copy both files into the same folder as `app.py`.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
project/
├── app.py               ← Streamlit application
├── requirements.txt     ← Python dependencies
├── XG_model.pkl         ← trained XGBoost model  [generate from notebook]
└── model_columns.pkl    ← feature column names   [generate from notebook]
```

---

## 🌐 Deployment Options

### Streamlit Community Cloud (free)
1. Push project to a GitHub repository
2. Visit https://share.streamlit.io
3. Connect your repo, set `app.py` as entrypoint
4. Deploy — no server needed

### Heroku
```bash
# Add Procfile:
echo "web: streamlit run app.py --server.port $PORT" > Procfile
git push heroku main
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## ✨ Features

| Tab | Contents |
|-----|----------|
| 🎯 Prediction & Risk | Live attrition probability gauge, risk radar chart, KPI cards, prioritised HR recommendations |
| 📊 Analytics Dashboard | Department breakdown, income vs attrition, age distribution, overtime analysis, satisfaction heatmap |
| 📈 Feature Insights | XGBoost feature importances, normalised employee scores, all-model comparison table |
| ℹ️ Model Info | Pipeline details, improvement summary, setup instructions |

---

## 🤖 Model Details

- **Algorithm:** XGBoost (600 estimators, max_depth=5, lr=0.03)
- **Imbalance handling:** SMOTE + scale_pos_weight (~5.2)
- **ROC-AUC:** 0.929 · **F1:** 0.681 · **Recall:** 64.3%
- **Features:** 30 (one-hot encoded categoricals + numeric)
