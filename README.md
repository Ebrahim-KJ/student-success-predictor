# 🎓 Student Success Predictor

An end-to-end machine learning system that predicts student dropout risk and academic underperformance — early enough for institutions to intervene.

Built with a full ML pipeline: data ingestion, feature engineering, multi-model comparison, hyperparameter optimisation (via genetic algorithms), SHAP explainability, causal inference analysis, and a live Streamlit dashboard.

---

## 🚀 Live Demo
👉 **[Launch the App](https://your-streamlit-link-here)** ← *(deploy to Streamlit Cloud and paste link)*

---

## 🎯 Problem Statement

Universities lose significant resources to student dropout, yet most interventions come too late. This system uses real student data to:

- **Predict** which students are at risk of failure or dropout
- **Explain** the top factors driving each student's risk score
- **Estimate** the causal impact of interventions (tutoring, parental support, etc.)

---

## 📁 Project Structure

```
student-success-predictor/
├── data/
│   ├── raw/                    # Original UCI dataset
│   └── processed/              # Cleaned & engineered features
├── src/
│   ├── data/
│   │   ├── load_data.py        # Data ingestion & downloading
│   │   └── preprocess.py       # Cleaning & encoding
│   ├── features/
│   │   └── build_features.py   # Feature engineering
│   ├── models/
│   │   ├── train.py            # Multi-model training with MLflow
│   │   └── evaluate.py         # Metrics, SHAP, visualisations
│   ├── optimization/
│   │   └── hyperparameter_opt.py  # Genetic algorithm + Optuna HPO
│   └── causal/
│       └── causal_inference.py # DoWhy causal analysis
├── app/
│   └── streamlit_app.py        # Live interactive dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── models/                     # Saved model artefacts
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/student-success-predictor.git
cd student-success-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the data
```bash
python src/data/load_data.py
```

---

## 🏃 Running the Project

### Train all models
```bash
python src/models/train.py
```

### View MLflow experiment tracking
```bash
mlflow ui
# Open http://localhost:5000
```

### Launch the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

### Run hyperparameter optimisation
```bash
python src/optimization/hyperparameter_opt.py
```

### Run causal inference analysis
```bash
python src/causal/causal_inference.py
```

---

## 🧠 Models Compared

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble tree model |
| XGBoost | Gradient boosted trees |
| Neural Network | PyTorch MLP |

All experiments tracked via **MLflow**.

---

## 🔬 Hyperparameter Optimisation

Implements two approaches (directly extending Master's thesis work):
- **Genetic Algorithm** — custom evolutionary search over the hyperparameter space
- **Optuna (TPE)** — probabilistic Bayesian optimisation

---

## 🔍 Explainability

Uses **SHAP (SHapley Additive exPlanations)** to show:
- Global feature importance across all students
- Individual explanations per student prediction

---

## 📊 Causal Inference

Goes beyond correlation — uses **DoWhy** to estimate the *causal effect* of interventions like:
- Extra tutoring support
- Family educational support
- Reduced alcohol consumption

---

## 📦 Docker

```bash
docker build -t student-success-predictor .
docker run -p 8501:8501 student-success-predictor
```

---

## 📚 Dataset

[UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)  
P. Cortez and A. Silva, 2008. Using Data Mining to Predict Secondary School Student Performance.

---

## 👤 Author

**Ebrahim Kashif Jahangiri**  
MSc Data Science for Public Policy — Cornell University  
MEng Electrical Engineering with Management — Imperial College London  
📧 Ebrahim.KJahangiri@Gmail.com
