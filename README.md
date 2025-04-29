# 🤖 AI DataForge: An End-to-End Machine Learning App

**AI DataForge** is a no-code, interactive machine learning platform built using **Streamlit**. This app empowers users to upload any dataset, visualize data, train ML models, and make predictions — all in one smooth interface.

---

## 🚀 Key Features

- 📂 Upload any `.csv` dataset
- 📊 Visual Exploratory Data Analysis (EDA)
- 🧹 Automatic data cleaning & encoding
- 🎯 Target column selection
- 🤖 Train ML models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- 📈 View metrics:
  - Accuracy
  - F1 Score
  - Confusion Matrix
  - Classification Report
- 💾 Save trained model as `.pkl`
- 🔮 Load a saved model for live predictions

---

## 🛠 Tech Stack

- **Python 3**
- **Streamlit** – for the web UI
- **Pandas** – data manipulation
- **Scikit-learn** – machine learning models
- **XGBoost** – advanced gradient boosting
- **Seaborn & Plotly** – visualization
- **Joblib** – model serialization

---

## 📁 Project Structure

    AI_DataForge/
    ├── app.py                # Main Streamlit application
    ├── data.csv              # Any data file in csv format
    ├── README.md             # This documentation file
    └── model.pkl             # Saved model (optional)


---

## ⚙️ Installation & Setup

1. **Clone the repository** or download the files.
2. (Optional but recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate     # On Windows
    source venv/bin/activate  # On Linux/Mac
    ```
3. **Run the app**:
    ```bash
    streamlit run app.py
    ```

---

## 📦 How to Use

1. Upload a CSV file.
2. Explore the dataset using automatic EDA tools.
3. Select a target column for prediction.
4. Train your preferred ML model.
5. Evaluate results and optionally save the model.
6. Load a saved model to make predictions from manual input.

---

## 🧠 Project Status

✅ Phase 1: EDA + Preprocessing  
✅ Phase 2: Model Training + Evaluation  
✅ Phase 3: Save and Load Model + Predict  
📈 More enhancements like AutoML, regression support, and deployment coming soon!

---

## 📄 License

This project is for educational and learning purposes. No commercial license implied.

