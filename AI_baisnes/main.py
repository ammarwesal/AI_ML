from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import numpy as np
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bias Detection API",
    description="API for detecting bias in AI model predictions",
    version="1.0.0"
)

# === Global variables ===
model = None
encoders = {}
X_columns = []
dataset_info = {}

# === Load saved model and encoders if available ===
def load_saved_artifacts():
    global model, encoders, X_columns, dataset_info
    try:
        if all(os.path.exists(f) for f in ["model.pkl", "encoders.pkl", "columns.pkl"]):
            model = joblib.load("model.pkl")
            encoders = joblib.load("encoders.pkl")
            X_columns = joblib.load("columns.pkl")
            if os.path.exists("dataset_info.pkl"):
                dataset_info = joblib.load("dataset_info.pkl")
            logger.info("Successfully loaded saved model and encoders")
    except Exception as e:
        logger.error(f"Error loading saved artifacts: {e}")

load_saved_artifacts()

@app.get("/")
def read_root():
    return {
        "message": "Hello, Ammar! Your Bias Detection API is running.",
        "model_status": "Loaded" if model is not None else "Not trained",
        "endpoints": ["/upload_data", "/train_model", "/predict", "/bias_report"]
    }

class InputData(BaseModel):
    data: Dict[str, Any]

@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    """Upload CSV data for training"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        contents = await file.read()
        with open("adult.csv", "wb") as f:
            f.write(contents)
        
        # Validate the uploaded file
        columns = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ]
        
        df = pd.read_csv("adult.csv", names=columns, header=None)
        logger.info(f"File uploaded successfully. Shape: {df.shape}")
        
        return {
            "message": "File uploaded successfully",
            "rows": len(df),
            "columns": len(df.columns),
            "sample_data": df.head(3).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/train_model")
def train_model():
    """Train the bias detection model"""
    global model, encoders, X_columns, dataset_info

    try:
        if not os.path.exists("adult.csv"):
            raise HTTPException(status_code=400, detail="No data file found. Please upload data first.")

        columns = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ]

        # Load and clean data
        df = pd.read_csv("adult.csv", names=columns, header=None)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Clean data - remove leading/trailing whitespace
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()
        
        # Handle missing values more systematically
        encoders = {}
        for col in df.select_dtypes(include='object').columns:
            # Replace '?' with 'Unknown'
            df[col] = df[col].replace('?', 'Unknown')
            df[col] = df[col].fillna("Unknown")
            
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        # Store dataset info for bias analysis
        dataset_info = {
            'total_samples': len(df),
            'feature_names': list(df.columns),
            'categorical_features': list(df.select_dtypes(include='object').columns)
        }

        X = df.drop("income", axis=1)
        y = df["income"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Train model
        model = LogisticRegression(max_iter=3000, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        X_columns = list(X.columns)

        # Save artifacts
        joblib.dump(model, "model.pkl")
        joblib.dump(encoders, "encoders.pkl")
        joblib.dump(X_columns, "columns.pkl")
        joblib.dump(dataset_info, "dataset_info.pkl")

        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model trained successfully with accuracy: {accuracy:.4f}")
        
        return {
            "message": "Model trained successfully",
            "accuracy": accuracy,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features": len(X_columns),
            "classification_report": {
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1_score": report["weighted avg"]["f1-score"]
            }
        }
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/predict")
def predict(input_data: InputData):
    """Make prediction for given input"""
    global model, encoders, X_columns

    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train the model first.")

    try:
        data_df = pd.DataFrame([input_data.data])
        
        # Check for required columns
        missing_columns = set(X_columns) - set(data_df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {list(missing_columns)}"
            )

        # Encode categorical features
        for col in data_df.select_dtypes(include='object').columns:
            if col in encoders:
                le = encoders[col]
                known_labels = set(le.classes_)
                
                # Handle unknown categories
                data_df[col] = data_df[col].apply(
                    lambda x: x if x in known_labels else "Unknown"
                )
                
                # Add 'Unknown' to encoder if not present
                if "Unknown" not in known_labels:
                    le.classes_ = np.append(le.classes_, "Unknown")
                
                data_df[col] = le.transform(data_df[col])
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"No encoder found for column: {col}"
                )

        # Add missing columns with default values
        for col in X_columns:
            if col not in data_df.columns:
                data_df[col] = 0

        # Make prediction
        prediction_proba = model.predict_proba(data_df[X_columns])[0]
        prediction = model.predict(data_df[X_columns])[0]
        
        # Convert prediction back to original label
        income_encoder = encoders.get("income")
        if income_encoder:
            prediction_label = income_encoder.inverse_transform([int(prediction)])[0]
            class_labels = income_encoder.classes_
        else:
            prediction_label = int(prediction)
            class_labels = [0, 1]

        return {
            "prediction": prediction_label,
            "confidence": float(max(prediction_proba)),
            "probabilities": {
                str(class_labels[i]): float(prob) 
                for i, prob in enumerate(prediction_proba)
            }
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/bias_report")
def bias_report():
    """Generate comprehensive bias analysis report"""
    global model, encoders, X_columns, dataset_info

    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train the model first.")

    try:
        # Load and preprocess data
        columns = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ]
        
        df = pd.read_csv("adult.csv", names=columns, header=None)
        
        # Clean and encode data consistently
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('?', 'Unknown')
            df[col] = df[col].fillna("Unknown")
            
            if col in encoders:
                le = encoders[col]
                known_labels = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known_labels else "Unknown")
                
                if "Unknown" not in known_labels:
                    le.classes_ = np.append(le.classes_, "Unknown")
                
                df[col] = le.transform(df[col])

        # Split data
        X = df.drop("income", axis=1)
        y = df["income"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Get predictions
        y_pred = model.predict(X_test)
        
        # Analyze bias across multiple sensitive attributes
        bias_analysis = {}
        sensitive_features = ['sex', 'race']
        
        for feature in sensitive_features:
            if feature in X_test.columns:
                sensitive_values = X_test[feature]
                
                # Validate lengths
                if not (len(y_test) == len(y_pred) == len(sensitive_values)):
                    logger.error(f"Length mismatch for {feature}")
                    continue
                
                try:
                    # Calculate fairness metrics directly
                    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_values)
                    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_values)
                    
                    # Calculate metrics using MetricFrame for accuracy only
                    metrics = MetricFrame(
                        metrics={"accuracy": lambda yt, yp: (yt == yp).mean()},
                        y_true=y_test,
                        y_pred=y_pred,
                        sensitive_features=sensitive_values
                    )
                    
                    # Get group names if encoder exists
                    group_names = {}
                    if feature in encoders:
                        encoder = encoders[feature]
                        for encoded_val in metrics.by_group.index.get_level_values(feature).unique():
                            try:
                                group_names[encoded_val] = encoder.inverse_transform([encoded_val])[0]
                            except:
                                group_names[encoded_val] = f"Group_{encoded_val}"
                    
                    bias_analysis[feature] = {
                        "overall_accuracy": float(metrics.overall["accuracy"]),
                        "demographic_parity_difference": float(dp_diff),
                        "equalized_odds_difference": float(eo_diff),
                        "by_group_accuracy": {
                            group_names.get(idx, f"Group_{idx}"): float(val) 
                            for idx, val in metrics.by_group["accuracy"].items()
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Error calculating metrics for {feature}: {e}")
                    bias_analysis[feature] = {"error": str(e)}

        # Generate bias interpretation
        interpretation = generate_bias_interpretation(bias_analysis)
        
        return {
            "bias_analysis": bias_analysis,
            "interpretation": interpretation,
            "dataset_info": dataset_info,
            "model_performance": {
                "overall_accuracy": float((y_test == y_pred).mean()),
                "total_predictions": len(y_test)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating bias report: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating bias report: {str(e)}")

def generate_bias_interpretation(bias_analysis: Dict) -> Dict:
    """Generate human-readable interpretation of bias metrics"""
    interpretation = {
        "summary": [],
        "recommendations": []
    }
    
    for feature, metrics in bias_analysis.items():
        if "error" in metrics:
            continue
            
        dp_diff = abs(metrics.get("demographic_parity_difference", 0))
        eo_diff = abs(metrics.get("equalized_odds_difference", 0))
        
        # Interpret demographic parity
        if dp_diff > 0.1:
            interpretation["summary"].append(
                f"HIGH BIAS detected in {feature.upper()}: Demographic parity difference = {dp_diff:.3f}"
            )
            interpretation["recommendations"].append(
                f"Consider bias mitigation techniques for {feature} (threshold: >0.1)"
            )
        elif dp_diff > 0.05:
            interpretation["summary"].append(
                f"MODERATE BIAS detected in {feature.upper()}: Demographic parity difference = {dp_diff:.3f}"
            )
        else:
            interpretation["summary"].append(
                f"LOW BIAS in {feature.upper()}: Demographic parity difference = {dp_diff:.3f}"
            )
        
        # Interpret equalized odds
        if eo_diff > 0.1:
            interpretation["summary"].append(
                f"HIGH BIAS in equal opportunity for {feature.upper()}: Difference = {eo_diff:.3f}"
            )
    
    if not interpretation["recommendations"]:
        interpretation["recommendations"].append("Bias levels are within acceptable thresholds.")
    
    return interpretation

@app.get("/model_info")
def model_info():
    """Get information about the current model"""
    if model is None:
        return {"status": "No model trained"}
    
    return {
        "model_type": "Logistic Regression",
        "features": len(X_columns),
        "feature_names": X_columns,
        "encoders": list(encoders.keys()),
        "dataset_info": dataset_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)