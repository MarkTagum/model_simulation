import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App
st.title("Movie Rating Prediction")

# Sidebar for Configuration
st.sidebar.header("Configuration")

# Step 1: Feature and Target Configuration
st.sidebar.subheader("Step 1: Feature and Target Configuration")

# Feature Names
feature_names = st.sidebar.text_input("Enter Feature Names (comma-separated)", "Budget,Runtime,Genre,Release_Year")
feature_names = [name.strip() for name in feature_names.split(",")]

# Target Name
target_name = st.sidebar.text_input("Enter Target Name", "Rating")

# Step 2: Class-Specific Settings
st.sidebar.subheader("Step 2: Class-Specific Settings")

# Default class-specific settings
default_class_settings = {
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

# Initialize session state for class settings
if "class_settings" not in st.session_state:
    st.session_state.class_settings = default_class_settings

# Function to update class settings
def update_class_settings(genre, param, value):
    st.session_state.class_settings[genre][param] = float(value)

# Display class-specific settings with adjustable inputs
for genre, settings in st.session_state.class_settings.items():
    st.sidebar.subheader(f"{genre} Settings")
    with st.sidebar.expander(f"{genre} Settings", expanded=False):
        for param, value in settings.items():
            new_value = st.number_input(f"{param.replace('_', ' ').title()}", value=value, key=f"{genre}_{param}")
            update_class_settings(genre, param, new_value)

# Step 3: Sample Size and Train/Test Split
st.sidebar.subheader("Step 3: Sample Size and Train/Test Split")

# Sample Size
sample_size = st.sidebar.number_input("Number of Samples", min_value=10, max_value=50000, value=500)

# Train/Test Split
test_size = st.sidebar.slider("Test Size (percentage)", 0.1, 0.5, 0.3)

# Generate Synthetic Data Button
if st.sidebar.button("Generate Synthetic Data"):
    st.session_state.data_generated = False

    # Generate Synthetic Data
    np.random.seed(42)

    # Initialize lists for features and target
    Budget = []
    Runtime = []
    Genre = []
    Release_Year = []
    Rating = []

    # Generate data for each genre
    for genre, settings in st.session_state.class_settings.items():
        n = int(sample_size / len(st.session_state.class_settings))  # Equal distribution across genres
        Budget.extend(np.random.normal(settings["Budget_mean"], settings["Budget_std"], n))
        Runtime.extend(np.random.normal(settings["Runtime_mean"], settings["Runtime_std"], n))
        Release_Year.extend(np.random.normal(settings["Release_Year_mean"], settings["Release_Year_std"], n))
        Rating.extend(np.random.normal(settings["Rating_mean"], settings["Rating_std"], n))
        Genre.extend([genre] * n)

    # Create DataFrame
    data = pd.DataFrame({
        "Budget": Budget,
        "Runtime": Runtime,
        "Genre": Genre,
        "Release_Year": Release_Year,
        "Rating": Rating
    })

    # Save to session state
    st.session_state.data = data
    st.session_state.data_generated = True

# Display Generated Data
if "data_generated" in st.session_state and st.session_state.data_generated:
    st.header("Generated Synthetic Data")
    st.write(st.session_state.data.head())

    # Train/Test Split
    X = st.session_state.data[feature_names]
    y = st.session_state.data[target_name]

    # Encode categorical variables
    X = pd.get_dummies(X, columns=["Genre"], drop_first=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train a Random Forest Regressor Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on Test Set
    y_pred = model.predict(X_test)

    # Evaluate the Model
    st.header("Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R-squared (R²): {r2_score(y_test, y_pred):.2f}")

    # Simulate New Data
    st.header("Simulation")
    new_data = pd.DataFrame({
        "Budget": np.random.randint(100000, 100000000, 100),
        "Runtime": np.random.randint(60, 200, 100),
        "Genre": np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror'], 100),
        "Release_Year": np.random.randint(1980, 2023, 100)
    })

    # Encode categorical variables
    new_data = pd.get_dummies(new_data, columns=["Genre"], drop_first=True)

    # Predict Movie Ratings
    new_predictions = model.predict(new_data)
    new_data[target_name] = new_predictions

    st.write("Simulated Movie Ratings:")
    st.write(new_data.head())