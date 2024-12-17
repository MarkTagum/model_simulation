import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Streamlit App
st.title("Traffic Light Control Simulation")

# Sidebar for Configuration
st.sidebar.header("Configuration")

# Step 1: Feature and Class Configuration
st.sidebar.subheader("Step 1: Feature and Class Configuration")

# Feature Names
feature_names = st.sidebar.text_input("Enter Feature Names (comma-separated)", "Lane1_Vehicles,Lane2_Vehicles,Time_of_Day,Weather")
feature_names = [name.strip() for name in feature_names.split(",")]

# Class Names
class_names = st.sidebar.text_input("Enter Class Names (comma-separated)", "Green_Lane1,Green_Lane2")
class_names = [name.strip() for name in class_names.split(",")]

# Step 2: Class-Specific Settings
st.sidebar.subheader("Step 2: Class-Specific Settings")

# Class-Specific Rules
st.sidebar.write("Define rules for class assignment:")
lane1_threshold = st.sidebar.slider("Lane1_Vehicles Threshold for Green_Lane1", 0, 100, 25)
lane2_threshold = st.sidebar.slider("Lane2_Vehicles Threshold for Green_Lane2", 0, 100, 25)

# Step 3: Sample Size and Train/Test Split
st.sidebar.subheader("Step 3: Sample Size and Train/Test Split")

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
    Lane1_Vehicles = np.random.randint(0, 50, sample_size)
    Lane2_Vehicles = np.random.randint(0, 50, sample_size)
    Time_of_Day = np.random.choice(['Morning', 'Afternoon', 'Evening'], sample_size)
    Weather = np.random.choice(['Sunny', 'Rainy', 'Cloudy'], sample_size)

    # Target: Traffic Light State
    Traffic_Light_State = []
    for i in range(sample_size):
        if Lane1_Vehicles[i] > lane1_threshold:
            Traffic_Light_State.append(class_names[0])  # Green_Lane1
        elif Lane2_Vehicles[i] > lane2_threshold:
            Traffic_Light_State.append(class_names[1])  # Green_Lane2
        else:
            Traffic_Light_State.append(np.random.choice(class_names))  # Random assignment

    # Create DataFrame
    data = pd.DataFrame({
        feature_names[0]: Lane1_Vehicles,
        feature_names[1]: Lane2_Vehicles,
        feature_names[2]: Time_of_Day,
        feature_names[3]: Weather,
        "Traffic_Light_State": Traffic_Light_State
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
    y = st.session_state.data["Traffic_Light_State"]

    # Encode categorical variables
    X = pd.get_dummies(X, columns=[feature_names[2], feature_names[3]], drop_first=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train a Decision Tree Model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Predict on Test Set
    y_pred = model.predict(X_test)

    # Evaluate the Model
    st.header("Model Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Simulate New Data
    st.header("Simulation")
    new_data = pd.DataFrame({
        feature_names[0]: np.random.randint(0, 50, 100),
        feature_names[1]: np.random.randint(0, 50, 100),
        feature_names[2]: np.random.choice(['Morning', 'Afternoon', 'Evening'], 100),
        feature_names[3]: np.random.choice(['Sunny', 'Rainy', 'Cloudy'], 100)
    })

    # Encode categorical variables
    new_data = pd.get_dummies(new_data, columns=[feature_names[2], feature_names[3]], drop_first=True)

    # Predict Traffic Light States
    new_predictions = model.predict(new_data)
    new_data["Traffic_Light_State"] = new_predictions

    st.write("Simulated Traffic Light States:")
    st.write(new_data.head())