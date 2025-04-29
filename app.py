import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

# Page config
st.set_page_config(page_title="AI DataForge - EDA + ML", layout="wide")

# Title
st.title("📊 AI DataForge: Auto EDA + ML Training")
st.markdown("Upload any dataset, explore it visually, train a classification model, and save it!")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Dataset preview
    with st.expander("📌 Dataset Preview"):
        st.dataframe(df.head())

    # Summary
    with st.expander("📋 Dataset Summary"):
        st.write(f"**Shape:** {df.shape}")
        st.write("**Column Types:**")
        st.write(df.dtypes.value_counts())
        st.write("**Missing Values (%):**")
        st.write((df.isnull().mean() * 100).round(2))

    # Column types
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Histograms
    st.subheader("📈 Numeric Column Distributions")
    selected_num = st.multiselect("Select numeric columns", num_cols, default=num_cols[:2])
    for col in selected_num:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    if len(num_cols) > 1:
        st.subheader("📉 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Categorical Value Counts
    if len(cat_cols) > 0:
        st.subheader("🧾 Categorical Columns - Value Counts")
        selected_cat = st.multiselect("Select categorical columns", cat_cols, default=cat_cols[:2])
        for col in selected_cat:
            st.write(f"**{col}**")
            st.bar_chart(df[col].value_counts())

    # Model Training
    st.subheader("🎯 Model Training & Evaluation")
    target_column = st.selectbox("Select Target Column", df.columns)

    if target_column:
        df = df.dropna(subset=[target_column])
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical features
        label_encoders = {}
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        # Fill missing values
        X = X.fillna(X.median(numeric_only=True))

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model choice
        model_choice = st.radio("Choose Model", ["Random Forest", "Logistic Regression", "XGBoost"])

        if model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        st.markdown("### 🧾 Evaluation Metrics")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
        st.write(f"**F1 Score:** {f1_score(y_test, y_pred, average='weighted'):.4f}")

        st.markdown("### 📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.markdown("### 📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Save model
        st.markdown("### 💾 Save Trained Model")
        file_name = st.text_input("Enter file name (without extension)", "model")
        if st.button("Save Trained Model"):
            joblib.dump(model, f"{file_name}.pkl")
            st.success(f"Model saved successfully as **{file_name}.pkl** ✅")

else:
    st.info("👈 Please upload a CSV file using the sidebar to begin.")
