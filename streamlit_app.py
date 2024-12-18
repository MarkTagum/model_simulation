import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Function to generate synthetic movie data
def generate_synthetic_movie_data(features, sample_size):
    data = {feature: [] for feature in features}
    data['Rating'] = []

    # Generate synthetic data
    for _ in range(sample_size):
        # Example: Random values for features
        budget = np.random.randint(100000, 100000000)  # Budget in USD
        runtime = np.random.randint(60, 240)  # Runtime in minutes
        genre = np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror'])  # Genre
        year = np.random.randint(1980, 2023)  # Release year
        popularity = np.random.uniform(0, 100)  # Popularity score

        # Simulate a movie rating (target variable)
        rating = np.random.uniform(1, 10)  # Rating between 1 and 10

        # Append to data
        data['Budget (USD)'].append(budget)
        data['Runtime (min)'].append(runtime)
        data['Genre'].append(genre)
        data['Release Year'].append(year)
        data['Popularity'].append(popularity)
        data['Rating'].append(rating)

    return pd.DataFrame(data)

# Streamlit App
st.title("Movie Rating Prediction")

# Sidebar for Data Generation Parameters
st.sidebar.header("Synthetic Data Generation")

# Feature Configuration
st.sidebar.subheader("Feature Configuration")
features = ['Budget (USD)', 'Runtime (min)', 'Genre', 'Release Year', 'Popularity']
st.sidebar.write("Features: ", ", ".join(features))

# Sample Size
sample_size = st.sidebar.number_input("Number of samples", min_value=100, max_value=100000, value=500, step=100)

# Generate Data Button
if st.sidebar.button("Generate Data"):
    df = generate_synthetic_movie_data(features, sample_size)
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
    y = df['Rating']

    # One-hot encode categorical features (e.g., Genre)
    X = pd.get_dummies(X, columns=['Genre'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train Model Button
    if st.sidebar.button("Train Model"):
        # Train a RandomForestRegressor
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
    genre = st.selectbox("Genre", ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror'])
    year = st.number_input("Release Year", min_value=1980, max_value=2023, value=2020)
    popularity = st.number_input("Popularity", min_value=0.0, max_value=100.0, value=50.0)

    # Prepare input data
    input_data = pd.DataFrame({
        'Budget (USD)': [budget],
        'Runtime (min)': [runtime],
        'Genre': [genre],
        'Release Year': [year],
        'Popularity': [popularity]
    })

    # One-hot encode categorical features
    input_data = pd.get_dummies(input_data, columns=['Genre'], drop_first=True)

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