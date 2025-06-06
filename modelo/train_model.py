# ----------------------------------------------------------------------------------------------------------------
# Python Script for Heart Disease Prediction
# ----------------------------------------------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import dump, load # Importar dump e load para salvar/carregar modelos
import shap # Importar SHAP

try:
    import kagglehub
except ImportError:
    print("kagglehub is not installed. Please install it using 'pip install kagglehub' to download datasets from Kaggle.")
    print("Proceeding without Kaggle download for demonstration purposes, assuming 'heart.csv' is in the current directory.")
    kagglehub = None

# --- Configuration ---
DATASET_PATH = 'heart.csv'
TARGET_COLUMN = 'target'
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5 # Número de folds para validação cruzada

# --- Model Saving Configuration ---
OUTPUT_MODEL_DIR = 'modelo'
MODEL_FILENAME = 'model_pipeline.pkl'
EXPLAINER_FILENAME = 'shap_explainer.pkl'
FEATURE_ORDER_FILENAME = 'feature_order.pkl'

FULL_MODEL_PATH = os.path.join(OUTPUT_MODEL_DIR, MODEL_FILENAME)
FULL_EXPLAINER_PATH = os.path.join(OUTPUT_MODEL_DIR, EXPLAINER_FILENAME)
FULL_FEATURE_ORDER_PATH = os.path.join(OUTPUT_MODEL_DIR, FEATURE_ORDER_FILENAME)


# --- Kaggle Integration Configuration ---
KAGGLE_USERNAME = "lrgsouza"
KAGGLE_KEY = "65eb4b4b48bd6caccc9fb2a6b9cb40ba"
KAGGLE_DATASET_REF = "johnsmith88/heart-disease-dataset"

# --- Dataset Column Descriptions (for reference, not used in execution logic) ---
DATA_COLUMNS_INFO = {
    'age': 'age',
    'sex': 'sex',
    'cp': 'chest pain type (4 values)',
    'trestbps': 'resting blood pressure',
    'chol': 'serum cholestoral in mg/dl',
    'fbs': 'fasting blood sugar > 120 mg/dl',
    'restecg': 'resting electrocardiographic results (values 0,1,2)',
    'thalach': 'maximum heart rate achieved',
    'exang': 'exercise induced angina',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'the slope of the peak exercise ST segment',
    'ca': 'number of major vessels (0-3) colored by flourosopy',
    'thal': 'thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',
    'target': 'target: 0 =healthy; 1 = heart disease'
}

# --- 1. Load Data ---
print("Step 1: Loading data...")

df = None
if kagglehub:
    try:
        kaggle_token = {"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
            json.dump(kaggle_token, f)
        os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_REF)
        csv_path = os.path.join(dataset_path, DATASET_PATH)
        df = pd.read_csv(csv_path)
        print(f"Data downloaded and loaded successfully from Kaggle. Shape: {df.shape}")
    except Exception as e:
        print(f"An error occurred while downloading from Kaggle: {e}. Attempting to load from local file.")

if df is None:
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Data loaded successfully from local file. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file {DATASET_PATH} was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        exit()

# --- Define Features ---
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Check if all defined features exist in the DataFrame
missing_num_features = [f for f in numerical_features if f not in df.columns]
missing_cat_features = [f for f in categorical_features if f not in df.columns]
if missing_num_features or missing_cat_features or TARGET_COLUMN not in df.columns:
    print("Error: Missing expected columns in the dataset.")
    if missing_num_features: print(f"Missing numerical: {missing_num_features}")
    if missing_cat_features: print(f"Missing categorical: {missing_cat_features}")
    if TARGET_COLUMN not in df.columns: print(f"Missing target column: {TARGET_COLUMN}")
    exit()

print("-" * 60)

# --- 4. Initial Data Exploration (Simplified) ---
print("Step 4: Initial Data Exploration...")
print(f"Total missing values: {df.isnull().sum().sum()}")
print("-" * 60)

# --- 7. Exploratory Data Analysis (EDA) (Simplified) ---
print("Step 7: Exploratory Data Analysis (EDA)...")

# --- 7.3 Feature Engineering ---
df_engineered = df.copy()
df_engineered['age_chol_interaction'] = df_engineered['age'] * df_engineered['chol']
df_engineered['oldpeak_risk_group'] = pd.cut(df_engineered['oldpeak'], bins=[-0.1, 1.0, 2.0, max(df_engineered['oldpeak'])],
                                      labels=['Low', 'Medium', 'High'], right=True, include_lowest=True)

# Update Feature Lists after Feature Engineering
numerical_features_updated = numerical_features + ['age_chol_interaction']
categorical_features_updated = categorical_features + ['oldpeak_risk_group']

# Prepare X and y for Modeling (incorporating engineered features)
X = df_engineered.drop(TARGET_COLUMN, axis=1)
y = df_engineered[TARGET_COLUMN]

# Save original feature order before any transformations
original_feature_order = X.columns.tolist()


print("-" * 60)

# --- 8. Data Preparation for Modeling ---
print("Step 8: Data Preparation for Modeling...")
print("Missing value imputation, scaling, and encoding are handled within the pipeline.")
print("-" * 60)

# --- 5. Sampling Methods (Actual Split for Training/Testing) ---
print("Step 5: Applying Sampling Methods (Train/Test Split)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")
print("-" * 60)

# --- 9. Pipelines and Custom Transformations ---
print("Step 9: Pipelines and Transformations (Definition)...")

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_updated),
        ('cat', categorical_transformer, categorical_features_updated)
    ],
    remainder='passthrough'
)
print("Preprocessing pipelines defined.")
print("-" * 60)


# --- 10. Machine Learning (Model Building) ---
print("Step 10: Machine Learning (Model Building)...")

model_performances = {}
trained_models = {}

# --- 10.1 Training Gaussian Naive Bayes ---
print("\n10.1.1 Training Gaussian Naive Bayes Model:")
model_pipeline_gnb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])
try:
    model_pipeline_gnb.fit(X_train, y_train)
    gnb_cv_scores = cross_val_score(model_pipeline_gnb, X, y, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
    gnb_mean_accuracy = gnb_cv_scores.mean()
    model_performances['Gaussian Naive Bayes'] = gnb_mean_accuracy
    trained_models['Gaussian Naive Bayes'] = model_pipeline_gnb
    print(f"Gaussian Naive Bayes trained. Mean CV Accuracy: {gnb_mean_accuracy:.4f}")
except Exception as e:
    print(f"Error training Gaussian Naive Bayes: {e}")
    model_performances['Gaussian Naive Bayes'] = None

# --- 10.2 Model Tuning (Logistic Regression with GridSearchCV) ---
print("\n10.1.2 Training Logistic Regression Model (with GridSearchCV):")
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'))
])
param_grid_lr = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}
grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1, verbose=0)

try:
    grid_search_lr.fit(X_train, y_train)
    model_performances['Logistic Regression (Tuned)'] = grid_search_lr.best_score_
    trained_models['Logistic Regression (Tuned)'] = grid_search_lr.best_estimator_
    print(f"Logistic Regression GridSearchCV completed. Best CV Accuracy: {grid_search_lr.best_score_:.4f}")
    print(f"Best parameters: {grid_search_lr.best_params_}")
except Exception as e:
    print(f"Error tuning Logistic Regression: {e}")
    model_performances['Logistic Regression (Tuned)'] = None

# --- 10.2 Model Comparison and Selection ---
print("\n10.2 Model Comparison and Selection:")

comparison_df = pd.DataFrame.from_dict(model_performances, orient='index', columns=['Mean CV Accuracy'])
comparison_df = comparison_df.sort_values(by='Mean CV Accuracy', ascending=False)

print("\n--- Model Performance Comparison (Mean CV Accuracy) ---")
print(comparison_df.to_string(float_format="%.4f"))
print("-" * 50)

best_model_name = comparison_df.index[0] if not comparison_df.empty else None
final_model_for_prediction = None

if best_model_name and model_performances[best_model_name] is not None:
    final_model_for_prediction = trained_models[best_model_name]
    print(f"\nBest model selected: {best_model_name}")
else:
    print("\nCould not determine a best model or no models trained successfully. Exiting.")
    exit()

print("-" * 60)


# --- 11. Machine Learning (Model Testing) ---
print("\nStep 11: Machine Learning (Model Testing)...")

print(f"\nEvaluating {best_model_name} on the Hold-out Test Set:")
if final_model_for_prediction is None:
    print("Error: No model selected for prediction. Exiting evaluation.")
    exit()

try:
    y_pred = final_model_for_prediction.predict(X_test)
    y_pred_proba = final_model_for_prediction.predict_proba(X_test)[:, 1]
except Exception as e:
    print(f"An error occurred during model prediction: {e}")
    exit()

print(f"\nModel Accuracy on Hold-out Test Set: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report on Hold-out Test Set:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Heart Disease (1)']))
print("\nConfusion Matrix on Hold-out Test Set:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=['Actual Healthy', 'Actual Disease'], columns=['Predicted Healthy', 'Predicted Disease']))

# --- 11.3 Performance on Training Data ---
y_train_pred = final_model_for_prediction.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nModel Accuracy on Training Set: {train_accuracy:.4f} (check for overfitting)")

# --- 11.4 Feature Importance ---
print("\n11.4 Feature Importance:")
classifier_step = final_model_for_prediction.named_steps['classifier']

if isinstance(classifier_step, LogisticRegression):
    try:
        fitted_preprocessor = final_model_for_prediction.named_steps['preprocessor']
        feature_names_after_preprocessing = fitted_preprocessor.get_feature_names_out()
        coefficients = pd.Series(classifier_step.coef_[0], index=feature_names_after_preprocessing)
        print("\nCoefficients (Logistic Regression - Absolute values sorted):")
        print(coefficients.sort_values(key=abs, ascending=False).head(15))
    except Exception as e:
        print(f"Could not retrieve LR coefficients. Error: {e}")
elif isinstance(classifier_step, GaussianNB):
    print("Gaussian Naive Bayes classifier does not provide direct feature importance scores like tree-based models.")
    print("Importance is implicitly related to feature distributions (mean/variance) for each class.")
else:
    print("Feature importance/coefficients not directly available for the selected model type.")

print("-" * 60)


# --- 12. Model Saving and SHAP Artifacts ---
print("\nStep 12: Model Saving and SHAP Artifacts...")

if not os.path.exists(OUTPUT_MODEL_DIR):
    os.makedirs(OUTPUT_MODEL_DIR)

if final_model_for_prediction:
    try:
        # Saving the model pipeline
        dump(final_model_for_prediction, FULL_MODEL_PATH)
        print(f"Best model pipeline saved to {FULL_MODEL_PATH}")
        
        # Load model for verification
        loaded_model = load(FULL_MODEL_PATH)
        print("Model loaded successfully for verification.")

        # --- SHAP Explainer Generation and Saving ---
        print("\nGenerating and saving SHAP explainer...")
        # Transform X_train using the preprocessor from the final pipeline
        X_train_transformed_for_shap = final_model_for_prediction.named_steps['preprocessor'].transform(X_train)
        
        # Create explainer based on the classifier in the pipeline
        explainer = shap.Explainer(final_model_for_prediction.named_steps['classifier'], X_train_transformed_for_shap)
        
        # Save the explainer
        dump(explainer, FULL_EXPLAINER_PATH)
        print(f"SHAP Explainer saved to {FULL_EXPLAINER_PATH}")

        # Save the original feature order
        dump(original_feature_order, FULL_FEATURE_ORDER_PATH)
        print(f"Feature order saved to {FULL_FEATURE_ORDER_PATH}")

    except Exception as e:
        print(f"An error occurred while saving model, explainer, or feature order: {e}")
else:
    print("No best model was selected to save.")

print("-" * 60)

print("\n--- Script Finished ---")