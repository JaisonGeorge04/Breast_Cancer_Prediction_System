**🎗️ Breast Cancer Diagnosis Prediction System**
A Machine Learning application that predicts breast cancer diagnosis (malignant vs. benign) using clinical measurement data, built with Scikit-learn and deployed as an interactive real-time prediction tool using Streamlit.

**📋 Overview**

The **Breast Cancer Diagnosis Prediction System** applies a Gradient Boosting classification model to the Wisconsin Breast Cancer dataset, enabling real-time diagnosis prediction based on clinical input values. The project covers the complete Machine Learning pipeline — from data preprocessing and feature scaling to model training, evaluation, and deployment — with trained model artifacts stored on AWS S3 for cloud-based model management.

The goal of this system is to act as a decision-support tool, highlighting potential malignancy risk based on clinical measurements while emphasizing high recall to minimize the risk of false negatives in a medical context.

**✨ Key Features**

🧬 𝑴𝒂𝒄𝒉𝒊𝒏𝒆 𝑳𝒆𝒂𝒓𝒏𝒊𝒏𝒈 𝑪𝒍𝒂𝒔𝒔𝒊𝒇𝒊𝒄𝒂𝒕𝒊𝒐𝒏 — Predicts malignant vs. benign diagnosis using Gradient Boosting
🧹 𝑫𝒂𝒕𝒂 𝑷𝒓𝒆𝒑𝒓𝒐𝒄𝒆𝒔𝒔𝒊𝒏𝒈 𝑷𝒊𝒑𝒆𝒍𝒊𝒏𝒆 — Handles data cleaning, feature scaling, and preparation using Pandas and Scikit-learn
📊 𝑴𝒐𝒅𝒆𝒍 𝑬𝒗𝒂𝒍𝒖𝒂𝒕𝒊𝒐𝒏 — Assessed using accuracy, precision, recall, and confusion matrix analysis
⚡ 𝑹𝒆𝒂𝒍-𝑻𝒊𝒎𝒆 𝑷𝒓𝒆𝒅𝒊𝒄𝒕𝒊𝒐𝒏𝒔 — Interactive Streamlit interface for instant, user-driven predictions
☁️ 𝑪𝒍𝒐𝒖𝒅-𝑩𝒂𝒔𝒆𝒅 𝑴𝒐𝒅𝒆𝒍 𝑺𝒕𝒐𝒓𝒂𝒈𝒆 — Trained model artifacts stored securely in AWS S3 for easy updates and retrieval
🎯 𝑹𝒆𝒄𝒂𝒍𝒍-𝑭𝒐𝒄𝒖𝒔𝒆𝒅 𝑶𝒑𝒕𝒊𝒎𝒊𝒛𝒂𝒕𝒊𝒐𝒏 — Tuned to minimize false negatives, critical for medical diagnosis use cases


**🛠️ Tech Stack**

CategoryTechnologyLanguagePythonML LibraryScikit-learnAlgorithmGradient BoostingData HandlingPandas, NumPyWeb InterfaceStreamlitCloud StorageAWS S3Version ControlGit & GitHub


**🏗️ System Architecture**

┌──────────────┐     ┌───────────────┐     ┌────────────────┐
│   Dataset    │────▶│  Preprocessing │────▶│  Gradient       │
│  (Clinical   │     │  (Pandas,      │     │  Boosting Model │
│   Features)  │     │   Scikit-learn)│     │  (Training)     │
└──────────────┘     └───────────────┘     └────────┬───────┘
                                                      │
                                                      ▼
                                            ┌──────────────────┐
                                            │  Model Artifact   │
                                            │  Stored on AWS S3 │
                                            └────────┬─────────┘
                                                      │
                                                      ▼
                                            ┌──────────────────┐
                                            │  Streamlit App    │
                                            │  (Real-Time       │
                                            │   Prediction UI)  │
                                            └──────────────────┘

The pipeline begins with the Wisconsin Breast Cancer dataset, which is cleaned and scaled before training a Gradient Boosting classifier. The trained model is serialized and stored in AWS S3, allowing the Streamlit front-end to load it on demand and generate real-time predictions from user-provided clinical inputs.

**📊 Model Details**

Aspect Description Dataset Wisconsin Breast Cancer DatasetProblem TypeBinary Classification (Malignant / Benign)AlgorithmGradient Boosting Classifier Preprocessing Missing value handling, feature scalingEvaluation MetricsAccuracy, Precision, Recall, F1-score, Confusion MatrixKey FocusHigh recall for malignant class to minimize false negatives


**🚀 Getting Started**

Prerequisites

Python 3.10+
pip
AWS account (if using S3 for model storage)

**Installation & Setup**

bash# Clone the repository
git clone https://github.com/<your-username>/breast-cancer-prediction-system.git
cd breast-cancer-prediction-system

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Running the Application

bashstreamlit run app.py

The application will open in your browser at http://localhost:8501.

Environment Variables (if loading model from AWS S3)

envAWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name
MODEL_FILE_KEY=model.pkl


**🧪 Model Training & Evaluation**

To retrain the model from scratch:

bashpython train_model.py

This script performs:

•Data loading and preprocessing
•Feature scaling using Scikit-learn
•Train/test split
•Model training using Gradient Boosting
•Evaluation using accuracy, precision, recall, and confusion matrix
•Model serialization for deployment


**📈 Future Enhancements**

 Add cross-validation for more robust performance evaluation
 Experiment with additional algorithms (Random Forest, XGBoost) for comparison
 Add feature importance visualization to improve model explainability
 Implement an automated model retraining pipeline with CI/CD
 Add unit tests for preprocessing and prediction functions

**⚠️ Disclaimer**

This project is intended as a decision-support and educational tool only. It is not a substitute for professional medical diagnosis, and predictions should not be used for actual clinical decision-making without expert medical review.


**👤 Author**

Jaison George
📧 jaisongeorge699@gmail.com
🔗 LinkedIn :-https://www.linkedin.com/in/jaison-george-887891310/ 
🔗 Portfolio :- https://jaisongeorge04.github.io/PortFolio/
