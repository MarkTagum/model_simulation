import streamlit as st # type: ignore
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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
class_settings = {}

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
    try:
        df = generate_synthetic_movie_data(features, class_settings, sample_size)
        st.success("Synthetic data generated successfully!")
        st.write(df)

        # Save data to session state
        st.session_state['data'] = df
    except Exception as e:
        st.error(f"Error generating data: {e}")

# Train/Test Split Configuration
if 'data' in st.session_state:
    st.sidebar.subheader("Train/Test Split Configuration")
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.9, value=0.2, step=0.1)

    # Split data
    df = st.session_state['data']
    X = df[features]
    y = df['Class']

    # One-hot encode categorical features (if any)
    X = pd.get_dummies(X, columns=features, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train Model Button
    if st.sidebar.button("Train Model"):
        try:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
            model.fit(X_train, y_train)

            # Predict on test data
            y_pred = model.predict(X_test)

            # Evaluate model
            st.success("Model trained successfully!")
            st.write("Best Parameters:", model.best_params_)
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

            # Save model to session state
            st.session_state['model'] = model
        except Exception as e:
            st.error(f"Error training model: {e}")

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
    if st.button("Predict Class"):
        try:
            model = st.session_state['model']
            prediction = model.predict(input_data)
            st.success(f"Predicted Movie Class: {prediction[0]}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")