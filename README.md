# Heart Disease Prediction Project

This is an end-to-end Machine Learning project to predict the likelihood of heart disease in a patient based on their medical attributes. The final model is an SVM (Support Vector Machine) wrapped in a user-friendly Streamlit web application.

## Project Structure
- `data_exploration.py`: Script to load, clean, and explore the dataset. Generates EDA visualisations.
- `maincode.py`: Script to train ML models (Logistic Regression, Random Forest, SVM) and evaluate them.
- `HeartApp.py`: Streamlit application code.
- `requirements.txt`: Python package dependencies.
- `heart_disease_model.pkl` & `scaler.pkl`: The saved ML model and scaler.
- `heart.csv`: Raw dataset.
- `heart_cleaned.csv`: Cleaned dataset.

## How to Run Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run HeartApp.py
   ```
   The app will open in your default web browser at `http://localhost:8501`.

## How to Deploy Online (Streamlit Community Cloud)

To share your project with the world for free, follow these steps:

1. **Create a GitHub Repository**:
   - Go to [GitHub](https://github.com/) and log in (or create an account).
   - Create a new public repository (e.g., `heart-disease-prediction`).
   - Upload all the files from this directory to your new GitHub repository.

2. **Deploy on Streamlit Community Cloud**:
   - Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with your GitHub account.
   - Click on **New app**.
   - Select your newly created `heart-disease-prediction` repository.
   - Set the `Branch` to `main` (or `master`).
   - Set the `Main file path` to `HeartApp.py`.
   - Click **Deploy!**
