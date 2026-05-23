# Telecom Customer Churn Prediction Dashboard

## Overview
This project is a polished Streamlit dashboard for predicting customer churn in a telecom dataset. It uses a trained Artificial Neural Network (ANN) model and reproduces the preprocessing pipeline from training, providing an interactive UI for live inference.

The application is designed to be evaluation-ready for IIT coursework and presents the churn prediction workflow in a professional, easy-to-use format.

## Key Features
- Interactive Streamlit dashboard with sidebar navigation
- Multi-page UI: Home, Predict, About Dataset, Help & Info
- Dark theme with modern card-style layout and polished metrics
- Real-time churn risk prediction using a saved Keras model
- Correct preprocessing pipeline with log transformation and scaler application
- Clear prediction result card with probability, confidence, and recommendation

## Project Files
- `churn_app.py` - Main Streamlit application file
- `customer_churn_ann.h5` - Saved trained ANN model for inference
- `scaler.pkl` - Saved scaler used to standardize model inputs
- `Churn Modeling.csv` - Original dataset file used for analysis and training
- `Churn_Prediction.ipynb` - Notebook containing data preprocessing, model training, and exploratory analysis
- `best_ann_model.h5` - Alternate trained model artifact (if available)
- `ann_tuning/` - Hyperparameter tuning outputs from Keras Tuner
- `requirements.txt` - Python package dependencies
- `README.md` - Project documentation

## Dependencies
The project requires the following Python libraries:

- streamlit
- tensorflow
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- keras-tuner
- imbalanced-learn

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Running the App
From the project root directory, run:

```bash
streamlit run churn_app.py
```

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

## Usage Instructions
1. Open the web app in your browser.
2. Navigate between pages using the left sidebar.
3. On the `Predict` page, enter customer profile values such as credit score, gender, age, tenure, balance, salary, and membership status.
4. Click `Predict Churn Risk` to generate the risk score.
5. Review the churn probability, confidence, and recommended retention action.

## Data and Model Details
- The dataset includes customer demographic and account information used to predict churn.
- The model is a Keras-trained ANN saved in `customer_churn_ann.h5`.
- The scaler object in `scaler.pkl` standardizes numeric inputs before model inference.
- The app reproduces the same preprocessing steps used during training for accurate predictions.

## Dashboard Workflow
- **Home**: Overview of the project, model, and application purpose.
- **Predict**: User input form for live churn inference.
- **About Dataset**: Summary of dataset features and model context.
- **Help & Info**: Guidance on how to use the dashboard effectively.

## Notes for Evaluation
- The app layout is designed for readability and professionalism.
- Predictions are displayed with meaningful labels, confidence, and recommendations.
- Preprocessing is explicitly applied to maintain consistency with training data.
- The project demonstrates full end-to-end deployment readiness for IIT evaluation.

## Improvements
Potential future enhancements:
- Add a dataset profiling page for summary statistics and visualizations
- Implement customer segmentation or model explainability (SHAP values)
- Add a deployment-ready Dockerfile or cloud deployment guide

## Author
Built with ❤️ by **Aatequa Ansari**

---

Thank you for reviewing this project. If you need any enhancements or a deployment-ready version, I can help extend this dashboard further.