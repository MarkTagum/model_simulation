import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Streamlit App
st.title("Synthetic Data Generator for Movie Rating Prediction")

# Sidebar for Configuration
st.sidebar.header("Data Generation Parameters")

# Step 1: Configure Features
st.sidebar.subheader("Feature Configuration")
n_features = st.sidebar.number_input("Number of Features", min_value=1, max_value=20, value=5)
n_classes = st.sidebar.number_input("Number of Classes (Ratings)", min_value=2, max_value=10, value=3)
class_names = []
for i in range(n_classes):
    class_name = st.sidebar.text_input(f"Class {i+1} Name", value=f"Rating {i+1}")
    class_names.append(class_name)

# Step 2: Class-Specific Settings
st.sidebar.subheader("Class-Specific Settings")
class_settings = {}
for class_name in class_names:
    with st.sidebar.expander(f"{class_name} Settings"):
        class_weight = st.number_input(f"Weight for {class_name}", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
        class_settings[class_name] = class_weight

# Step 3: Sample Size and Train/Test Split
st.sidebar.subheader("Sample Size and Train/Test Split")
sample_size = st.sidebar.number_input("Sample Size", min_value=10, max_value=10000, value=1000)
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=20)

# Step 4: Generate Synthetic Data
st.header("Generate Synthetic Data")
if st.button("Generate Data"):
    with st.spinner("Generating synthetic data..."):
        # Generate synthetic data
        X, y = make_classification(
            n_samples=sample_size,
            n_features=n_features,
            n_classes=n_classes,
            weights=[class_settings[class_name] for class_name in class_names],
            random_state=42
        )

        # Map class labels to class names
        y = np.array([class_names[label] for label in y])

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # Display Data
        st.subheader("Generated Data")
        st.write(f"Number of Features: {n_features}")
        st.write(f"Number of Classes: {n_classes}")
        st.write(f"Sample Size: {sample_size}")
        st.write(f"Test Size (%): {test_size}")

        # Display Train and Test Data
        st.subheader("Train Data")
        train_data = pd.DataFrame(X_train, columns=[f"Feature {i+1}" for i in range(n_features)])
        train_data["Rating"] = y_train
        st.dataframe(train_data)

        st.subheader("Test Data")
        test_data = pd.DataFrame(X_test, columns=[f"Feature {i+1}" for i in range(n_features)])
        test_data["Rating"] = y_test
        st.dataframe(test_data)

        # Download Data
        st.subheader("Download Data")
        train_csv = train_data.to_csv(index=False).encode('utf-8')
        test_csv = test_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Train Data", data=train_csv, file_name="train_data.csv", mime="text/csv")
        st.download_button("Download Test Data", data=test_csv, file_name="test_data.csv", mime="text/csv")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ by [Your Name]")