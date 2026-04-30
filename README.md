# 🚚 Food Delivery ETA Prediction & Analysis

A machine learning-powered system to predict food delivery time (ETA), classify delays, and analyze delivery patterns using clustering. Includes an interactive Streamlit dashboard for real-time insights and simulation.



## 📌 Features

* **ETA Prediction (Regression)**

  * Predict delivery time based on distance, traffic, weather, and other factors
  * Models used: Linear Regression, Decision Tree, Random Forest

* **Delay Classification**

  * Classifies deliveries into:

    * On-Time
    * Slight Delay
    * High Delay

* **Clustering Analysis**

  * Identifies patterns in delivery data using:

    * DBSCAN
    * K-Means
  * Visualizes relationships like:

    * Distance vs Time
    * Deliveries vs Time
    * Festival impact

* **Interactive Dashboard (Streamlit)**

  * Model training and evaluation
  * Visual analytics
  * Real-time ETA simulation

---

## 🧠 Tech Stack

* **Python**
* **Pandas, NumPy** – Data processing
* **Scikit-learn** – ML models
* **Matplotlib, Seaborn** – Visualization
* **Streamlit** – Dashboard UI

---

## 📂 Project Structure

```
Food-Delivery-ETA-Prediction/
│
├── data/
│   └── raw/
│       └── Zomato Dataset.csv
│
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── regression_model.py
│   │   └── classification_model.py
│   ├── clustering/
│   │   └── clustering.py
│   └── main.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd Food-Delivery-ETA-Prediction
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

---

## 📊 Dashboard Modules

* **Overview** → Dataset insights
* **Regression** → Train & evaluate ETA models
* **Classification** → Delay prediction metrics
* **Clustering** → Pattern visualization
* **ETA Simulator** → Predict delivery time for custom input

---

## 🔍 Key Insights

* Distance and traffic are major factors affecting delivery time
* Slight delays are hardest to classify accurately
* Clustering reveals patterns in delivery efficiency

---

## 📈 Example Output

* Classification accuracy ~83%
* ETA prediction error ~±3–5 minutes (MAE)

---

## 🚀 Future Improvements

* Real-time API integration
* Map-based visualizations
* Deep learning models for temporal prediction

---

## 📜 License

This project is for academic and learning purposes.
