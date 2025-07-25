# Bias Detection API for AI Models

A RESTful API built with FastAPI that analyzes bias in AI model predictions using the UCI Adult dataset. This project implements fairness metrics to detect and evaluate bias across different demographic groups.

## 🎯 Project Overview

This API provides a comprehensive solution for detecting bias in machine learning models by:
- Training a logistic regression classifier on the Adult Income dataset
- Analyzing fairness across sensitive attributes (gender and race)
- Computing key bias metrics including Demographic Parity and Equalized Odds
- Providing easy-to-use endpoints for model training, prediction, and bias analysis

## 📋 Features

- **Data Upload**: Upload CSV datasets for training
- **Model Training**: Train logistic regression models with automatic preprocessing
- **Prediction**: Make predictions on new data points
- **Bias Analysis**: Comprehensive bias reporting with fairness metrics
- **Interactive Documentation**: Auto-generated Swagger UI for easy testing

## 🏗️ Project Structure

```
AI_baisnes/
├── main.py                 # Main FastAPI application
├── adult.csv              # Training dataset (uploaded via API)
├── model.pkl              # Trained model (generated)
├── encoders.pkl           # Label encoders (generated)
├── columns.pkl            # Feature columns (generated)
├── dataset_info.pkl       # Dataset metadata (generated)
├── uploaded_adult.csv     # Backup of uploaded data
└── __pycache__/           # Python cache directory
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone or download the project files**
   ```bash
   # Create project directory
   mkdir bias-detection-api
   cd bias-detection-api
   ```

2. **Install required packages**
   ```bash
   pip install fastapi uvicorn pandas scikit-learn joblib numpy fairlearn
   ```

3. **Download the Adult dataset**
   - Download from: [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
   - Or use any CSV with the required columns (see API documentation)

4. **Run the application**
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access the API**
   - API: http://127.0.0.1:8000
   - Interactive Documentation: http://127.0.0.1:8000/docs

## 🚀 API Endpoints

### 1. Root Endpoint
```bash
GET /
```
**Description**: Check API status and available endpoints

**Response Example**:
```json
{
  "message": "Hello, Ammar! Your Bias Detection API is running.",
  "model_status": "Loaded",
  "endpoints": ["/upload_data", "/train_model", "/predict", "/bias_report"]
}
```

### 2. Upload Data
```bash
POST /upload_data
```
**Description**: Upload CSV training data

**Usage**:
```bash
curl -X POST "http://127.0.0.1:8000/upload_data" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@adult.csv"
```

**Response**: Data validation summary with sample records

### 3. Train Model
```bash
POST /train_model
```
**Description**: Train the bias detection model

**Usage**:
```bash
curl -X POST "http://127.0.0.1:8000/train_model" \
     -H "accept: application/json"
```

**Response Example**:
```json
{
  "message": "Model trained successfully",
  "accuracy": 0.8534,
  "training_samples": 22792,
  "test_samples": 9769,
  "features": 14,
  "classification_report": {
    "precision": 0.853,
    "recall": 0.853,
    "f1_score": 0.853
  }
}
```

### 4. Make Predictions
```bash
POST /predict
```
**Description**: Predict income for new data points

**Usage**:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "age": 35,
         "workclass": "Private",
         "education": "Bachelors",
         "marital_status": "Married-civ-spouse",
         "occupation": "Exec-managerial",
         "relationship": "Husband",
         "race": "White",
         "sex": "Male",
         "capital_gain": 0,
         "capital_loss": 0,
         "hours_per_week": 40,
         "native_country": "United-States"
       }
     }'
```

**Response Example**:
```json
{
  "prediction": ">50K",
  "confidence": 0.73,
  "probabilities": {
    "<=50K": 0.27,
    ">50K": 0.73
  }
}
```

### 5. Bias Report
```bash
GET /bias_report
```
**Description**: Generate comprehensive bias analysis

**Usage**:
```bash
curl -X GET "http://127.0.0.1:8000/bias_report" \
     -H "accept: application/json"
```

**Sample Response**:
```json
{
  "bias_analysis": {
    "sex": {
      "overall_accuracy": 0.811,
      "demographic_parity_difference": 0.127,
      "equalized_odds_difference": 0.183,
      "by_group_accuracy": {
        "Female": 0.891,
        "Male": 0.771
      }
    },
    "race": {
      "overall_accuracy": 0.811,
      "demographic_parity_difference": 0.220,
      "equalized_odds_difference": 0.264,
      "by_group_accuracy": {
        "White": 0.799,
        "Black": 0.907,
        "Asian-Pac-Islander": 0.772,
        "Amer-Indian-Eskimo": 0.92,
        "Other": 0.850
      }
    }
  },
  "interpretation": {
    "summary": [
      "HIGH BIAS detected in SEX: Demographic parity difference = 0.127",
      "HIGH BIAS detected in RACE: Demographic parity difference = 0.220"
    ],
    "recommendations": [
      "Consider bias mitigation techniques for sex (threshold: >0.1)",
      "Consider bias mitigation techniques for race (threshold: >0.1)"
    ]
  }
}
```

### 6. Model Information
```bash
GET /model_info
```
**Description**: Get current model details

## 📊 Bias Metrics Explained

### 1. Demographic Parity Difference
- **Definition**: Measures the difference in positive prediction rates between groups
- **Range**: -1 to 1 (closer to 0 is better)
- **Interpretation**: 
  - < 0.05: Low bias
  - 0.05-0.10: Moderate bias  
  - > 0.10: High bias (requires attention)

### 2. Equalized Odds Difference
- **Definition**: Measures fairness in true positive and false positive rates across groups
- **Range**: -1 to 1 (closer to 0 is better)
- **Interpretation**: Higher values indicate unequal treatment of different groups

### 3. Group Accuracy
- **Definition**: Prediction accuracy for each demographic group
- **Purpose**: Identifies which groups may be disadvantaged by the model

## 🎯 Bias Detection Results

Based on the API output, our analysis revealed:

### Gender Bias (SEX)
- **Status**: ⚠️ HIGH BIAS DETECTED
- **Demographic Parity Difference**: 0.127 (>0.1 threshold)
- **Equalized Odds Difference**: 0.183
- **Impact**: Males and females receive significantly different prediction rates
- **Group Performance**:
  - Female accuracy: 89.1%
  - Male accuracy: 77.1%

### Racial Bias (RACE)
- **Status**: ⚠️ HIGH BIAS DETECTED  
- **Demographic Parity Difference**: 0.220 (>0.1 threshold)
- **Equalized Odds Difference**: 0.264
- **Impact**: Different racial groups show substantial disparities in predictions
- **Group Performance**:
  - White: 79.9%
  - Black: 90.7%
  - Asian-Pac-Islander: 77.2%
  - Amer-Indian-Eskimo: 92.0%
  - Other: 85.0%

## ⚠️ Bias Implications

The detected bias means:

1. **Discriminatory Impact**: The model may unfairly advantage or disadvantage certain demographic groups
2. **Legal Concerns**: Such bias could violate anti-discrimination laws in hiring/lending contexts  
3. **Ethical Issues**: Perpetuates societal inequalities through automated decision-making
4. **Business Risk**: Could lead to lost opportunities and reputational damage

## 🛡️ Recommended Mitigation Strategies

1. **Data Preprocessing**:
   - Remove sensitive attributes from training
   - Apply sampling techniques to balance groups
   - Use synthetic data generation

2. **Algorithmic Approaches**:
   - Implement fairness constraints during training
   - Use bias-aware algorithms (e.g., FairLearn library)
   - Apply post-processing calibration

3. **Evaluation & Monitoring**:
   - Regular bias audits on new data
   - A/B testing with fairness metrics
   - Continuous monitoring in production

## 🧪 Testing the API

### Using Swagger UI
1. Navigate to http://127.0.0.1:8000/docs
2. Try each endpoint interactively
3. View request/response schemas

### Using Postman
Import the API endpoints and test with sample data

### Using Python Requests
```python
import requests

# Test prediction
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "data": {
            "age": 30,
            "workclass": "Private",
            "education": "Masters",
            "sex": "Female",
            "race": "Black"
            # ... other required fields
        }
    }
)
print(response.json())
```

## 🔧 Technical Implementation

### Model Architecture
- **Algorithm**: Logistic Regression with L2 regularization
- **Features**: 14 demographic and socioeconomic variables
- **Target**: Binary income classification (≤50K vs >50K)
- **Performance**: ~85% accuracy on test set

### Data Processing
- Automatic handling of missing values (replaced with "Unknown")
- Label encoding for categorical variables
- Stratified train/test split (70/30)
- Feature consistency validation

### Fairness Metrics Implementation
- Uses Microsoft FairLearn library for standard bias metrics
- Implements MetricFrame for group-wise analysis
- Provides interpretable bias thresholds and recommendations

## 🚀 Future Enhancements

1. **Additional Fairness Metrics**: Individual fairness, counterfactual fairness
2. **Bias Mitigation**: Built-in preprocessing and postprocessing techniques
3. **Model Comparison**: Support for multiple algorithms and fairness trade-offs
4. **Visualization**: Interactive dashboards for bias analysis
5. **Real-time Monitoring**: Continuous bias detection in production

## 📚 Dependencies

```txt
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
scikit-learn==1.3.2
numpy==1.25.2
fairlearn==0.9.0
joblib==1.3.2
pydantic==2.5.0
```

## 🤝 Contributing

This project was developed as part of an AI bias detection assignment. For questions or improvements, please refer to the course materials or contact the instructor.

## 📄 License

This project is for educational purposes as part of an AI/ML course assignment.

---

**Author**: Ammar  
**Course**: AI/ML Bias Detection  
**Date**: June 2025

**Note**: This API demonstrates bias detection techniques for educational purposes. In production systems, additional validation, security measures, and compliance checks would be required.
