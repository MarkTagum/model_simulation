import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Function to generate synthetic movie data
def generate_synthetic_movie_data(features, class_settings, sample_size):
    data = {feature: [] for feature in features}
    data['Class'] = []

    for class_name, settings in class_settings.items():
        for _ in range(sample_size):
            row = [np.random.normal(settings[f'Mean for {feature}'], settings[f'Std Dev for {feature}']) for feature in features]
            data['Class'].append(class_name)
            for idx, feature in enumerate(features):
                data[feature].append(row[idx])

    return pd.DataFrame(data)

# Streamlit App
st.title("Movie Rating Prediction")

# Sidebar for Data Generation Parameters
st.sidebar.header("Synthetic Data Generation")

# Feature Configuration
st.sidebar.subheader("Feature Configuration")
feature_names = st.sidebar.text_input("Enter feature names (comma-separated):", "Budget (USD), Runtime (min), Popularity")
features = [feature.strip() for feature in feature_names.split(",")]

# Class Configuration
st.sidebar.subheader("Class Configuration")
class_names = st.sidebar.text_input("Enter class names (comma-separated):", "Action, Comedy, Drama")
classes = [class_name.strip() for class_name in class_names.split(",")]

# Class-Specific Settings
st.sidebar.subheader("Class-Specific Settings")

class_settings = {
    "Action": {
        "Budget_mean": 50000000,
        "Budget_std": 10000000,
        "Runtime_mean": 120,
        "Runtime_std": 15,
        "Release_Year_mean": 2015,
        "Release_Year_std": 5,
        "Rating_mean": 7.5,
        "Rating_std": 0.5
    },
    "Comedy": {
        "Budget_mean": 20000000,
        "Budget_std": 5000000,
        "Runtime_mean": 90,
        "Runtime_std": 10,
        "Release_Year_mean": 2018,
        "Release_Year_std": 3,
        "Rating_mean": 6.8,
        "Rating_std": 0.4
    },
    "Drama": {
        "Budget_mean": 30000000,
        "Budget_std": 8000000,
        "Runtime_mean": 110,
        "Runtime_std": 12,
        "Release_Year_mean": 2016,
        "Release_Year_std": 4,
        "Rating_mean": 7.2,
        "Rating_std": 0.3
    },
    "Sci-Fi": {
        "Budget_mean": 80000000,
        "Budget_std": 20000000,
        "Runtime_mean": 130,
        "Runtime_std": 18,
        "Release_Year_mean": 2017,
        "Release_Year_std": 6,
        "Rating_mean": 7.8,
        "Rating_std": 0.6
    },
    "Horror": {
        "Budget_mean": 15000000,
        "Budget_std": 3000000,
        "Runtime_mean": 95,
        "Runtime_std": 8,
        "Release_Year_mean": 2019,
        "Release_Year_std": 2,
        "Rating_mean": 5.5,
        "Rating_std": 0.7
    }
}

for class_name in classes:
    with st.sidebar.expander(f"{class_name} Settings"):
        class_config = {}
        for feature in features:
            mean = st.sidebar.number_input(f"Mean for {feature}", value=100.0, key=f"{class_name}_{feature}_mean")
            std_dev = st.sidebar.number_input(f"Std Dev for {feature}", value=10.0, key=f"{class_name}_{feature}_std")
            class_config[f"Mean for {feature}"] = mean
            class_config[f"Std Dev for {feature}"] = std_dev
        class_settings[class_name] = class_config

# Sample Size
sample_size = st.sidebar.number_input("Number of samples", min_value=100, max_value=100000, value=500, step=100)

# Generate Data Button
if st.sidebar.button("Generate Data"):
    df = generate_synthetic_movie_data(features, class_settings, sample_size)
    st.success("Synthetic data generated successfully!")
    st.write(df)

    # Save data to session state
    st.session_state['data'] = df

# Train/Test Split Configuration
if 'data' in st.session_state:
    st.sidebar.subheader("Train/Test Split Configuration")
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.9, value=0.2, step=0.1)

    # Split data
    df = st.session_state['data']
    X = df[features]
    y = df['Class']

    # One-hot encode categorical features (e.g., Genre)
    X = pd.get_dummies(X, columns=features, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train Model Button
    if st.sidebar.button("Train Model"):
        # Train a RandomForestClassifier
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("Model trained successfully!")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")

        # Save model to session state
        st.session_state['model'] = model

# Movie Rating Prediction
if 'model' in st.session_state:
    st.header("Movie Rating Prediction")

    # Input features for prediction
    st.subheader("Enter Movie Details for Prediction")
    budget = st.number_input("Budget (USD)", min_value=100000, max_value=100000000, value=50000000)
    runtime = st.number_input("Runtime (min)", min_value=60, max_value=240, value=120)
    popularity = st.number_input("Popularity", min_value=0.0, max_value=100.0, value=50.0)

    # Prepare input data
    input_data = pd.DataFrame({
        'Budget (USD)': [budget],
        'Runtime (min)': [runtime],
        'Popularity': [popularity]
    })

    # One-hot encode categorical features
    input_data = pd.get_dummies(input_data, columns=features, drop_first=True)

    # Align input data with training data columns
    for col in X_train.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X_train.columns]

    # Predict Button
    if st.button("Predict Rating"):
        model = st.session_state['model']
        prediction = model.predict(input_data)
        st.success(f"Predicted Movie Rating: {prediction[0]:.2f}")