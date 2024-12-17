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

# Pre-defined Feature and Target Options
feature_options = ["Budget", "Runtime", "Genre", "Release_Year"]
target_options = ["Rating"]

# Select Features
selected_features = st.sidebar.multiselect("Select Features", feature_options, default=feature_options)

# Select Target
selected_target = st.sidebar.selectbox("Select Target", target_options)

# Step 2: Sample Size and Train/Test Split
st.sidebar.subheader("Step 2: Sample Size and Train/Test Split")

# Sample Size
sample_size = st.sidebar.number_input("Sample Size", min_value=10, max_value=10000, value=1000)

# Train/Test Split
test_size = st.sidebar.slider("Test Size (percentage)", 0.1, 0.5, 0.2)

# Generate Synthetic Data Button
if st.sidebar.button("Generate Synthetic Data"):
    st.session_state.data_generated = False

    # Generate Synthetic Data
    np.random.seed(42)

    # Features
    Budget = np.random.randint(100000, 100000000, sample_size)  # Movie budget in USD
    Runtime = np.random.randint(60, 200, sample_size)  # Movie runtime in minutes
    Genre = np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror'], sample_size)  # Movie genre
    Release_Year = np.random.randint(1980, 2023, sample_size)  # Release year

    # Target: Movie Rating (1 to 10)
    Rating = 5 + 0.0001 * Budget + 0.01 * Runtime + np.random.normal(0, 1, sample_size)
    Rating = np.clip(Rating, 1, 10)  # Ensure ratings are between 1 and 10

    # Create DataFrame
    data = pd.DataFrame({
        "Budget": Budget,
        "Runtime": Runtime,
        "Genre": Genre,
        "Release_Year": Release_Year,
        "Rating": Rating
    })

    # Filter data based on selected features and target
    data = data[selected_features + [selected_target]]

    # Save to session state
    st.session_state.data = data
    st.session_state.data_generated = True

# Display Generated Data
if "data_generated" in st.session_state and st.session_state.data_generated:
    st.header("Generated Synthetic Data")
    st.write(st.session_state.data.head())

    # Train/Test Split
    X = st.session_state.data[selected_features]
    y = st.session_state.data[selected_target]

    # Encode categorical variables
    X = pd.get_dummies(X, columns=["Genre"] if "Genre" in selected_features else [], drop_first=True)

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

    # Filter new data based on selected features
    new_data = new_data[selected_features]

    # Encode categorical variables
    new_data = pd.get_dummies(new_data, columns=["Genre"] if "Genre" in selected_features else [], drop_first=True)

    # Predict Movie Ratings
    new_predictions = model.predict(new_data)
    new_data[selected_target] = new_predictions

    st.write("Simulated Movie Ratings:")
    st.write(new_data.head())