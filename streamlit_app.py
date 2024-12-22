import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import seaborn as sns  # For EDA visualizations
import matplotlib.pyplot as plt  # For EDA visualizations
import plotly.graph_objects as go  # For enhanced table display
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Add custom CSS for better styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .subheader {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .container {
            padding: 20px;
        }
        .sidebar .sidebar-content {
            padding: 20px;
        }
        .streamlit-expanderHeader {
            font-size: 18px;
            font-weight: bold;
            color: #007ACC;
        }
        .streamlit-expander {
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

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
st.title("Synthetic Movie Data Generator and Classifier", anchor="title")

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
            mean = st.number_input(f"Mean for {feature} ({class_name})", value=100.0, key=f"{class_name}_{feature}_mean")
            std_dev = st.number_input(f"Std Dev for {feature} ({class_name})", value=10.0, key=f"{class_name}_{feature}_std")
            class_config[f"Mean for {feature}"] = mean
            class_config[f"Std Dev for {feature}"] = std_dev
        class_settings[class_name] = class_config

# Sample Size
sample_size = st.sidebar.number_input("Number of samples", min_value=100, max_value=100000, value=500, step=100)

# Generate Data and Train Model Button
if st.sidebar.button("Generate Data & Train Model"):
    try:
        # Generate the synthetic data
        df = generate_synthetic_movie_data(features, class_settings, sample_size)
        st.session_state['data'] = df  # Store the data in session_state
        st.success("Synthetic data generated successfully!")

        # Display the synthetic data generated
        st.write("### Sample of Generated Data:")
        st.write(df.head())

        # Save data to session state
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Synthetic Data as CSV",
            data=csv,
            file_name="synthetic_movie_data.csv",
            mime="text/csv"
        )

        # Train the model right after generating data
        # Split data
        X = df[features]
        y = df['Class']

        # Encode class labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

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
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

        st.subheader("Best Model Performance")
        st.write(f"**Best Model:** RandomForestClassifier")
        st.write(f"**Accuracy:** {accuracy:.4f}")

        # Convert classification report to a DataFrame
        report_df = pd.DataFrame(classification_rep).transpose()

        # Display the classification report
        st.write("### Classification Report (Best Model):")
        st.dataframe(report_df)

        # Optional: Use Plotly for an enhanced table
        fig = go.Figure(
            data=[go.Table(
                header=dict(values=list(report_df.columns), fill_color="paleturquoise", align="left"),
                cells=dict(values=[report_df[col] for col in report_df.columns], fill_color="lavender", align="left")
            )]
        )
        st.plotly_chart(fig)

        # Save model and label encoder to session state
        st.session_state['model'] = model
        st.session_state['label_encoder'] = label_encoder


        # Display histograms for each feature
        st.write("### Feature Distribution")
        for feature in features:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[feature], kde=True, color="teal")
            plt.title(f"Distribution of {feature}", fontsize=18)
            st.pyplot(plt)

        # Class distribution
        st.write("### Class Distribution")
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Class', data=df, palette='Set2')
        plt.title("Class Distribution", fontsize=18)
        st.pyplot(plt)

        # Correlation matrix heatmap using Plotly for interactivity
        st.write("### Correlation Matrix")
        corr_matrix = df[features].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis',
            colorbar=dict(title='Correlation')
        ))
        st.plotly_chart(fig)

        # Boxplots by class
        st.write("### Boxplots by Class")
        for feature in features:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x='Class', y=feature, data=df, palette='Set2')
            plt.title(f"Boxplot of {feature} by Class", fontsize=18)
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Error: {e}")
