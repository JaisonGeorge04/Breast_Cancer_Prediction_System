import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title and Subtitle Card */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(42, 82, 152, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
        pointer-events: none;
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 1.15rem;
        font-weight: 300;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Modern Styling for Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
        padding-top: 1rem;
    }
    
    /* Cards and Input Containers */
    .tab-card {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }
    
    /* Result Cards */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        margin-top: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.2);
        animation: slideIn 0.5s ease-out;
    }
    .result-card.benign {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        color: #1b4332;
    }
    .result-card.malignant {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        color: #590d22;
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .result-desc {
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    /* Button Customization */
    .stButton>button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2.5rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(42, 82, 152, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        margin-top: 1.5rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(42, 82, 152, 0.4) !important;
    }
    .stButton>button:active {
        transform: translateY(0) !important;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
try:
    gbc_model = joblib.load('gradient_boosting_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    import sys
    import sklearn
    st.error(f"Error loading model assets: {e}")
    st.info(f"Python version: {sys.version}")
    st.info(f"Scikit-learn version: {sklearn.__version__}")
    st.info("Make sure 'gradient_boosting_model.pkl' and 'scaler.pkl' are in the application directory.")
    st.stop()

# Define feature names lists
features_mean = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]
features_se = [
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se'
]
features_worst = [
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]
feature_names = features_mean + features_se + features_worst

# Sample cases data
presets = {
    "Select Preset": None,
    "Benign Tumor Case (Sample)": [
        13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 
        0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 
        15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
    ],
    "Malignant Tumor Case (Sample)": [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
}

# Initialize session state for features
for feat in feature_names:
    if feat not in st.session_state:
        st.session_state[feat] = 0.0

# Sidebar controls for loading preset
st.sidebar.title("🛠️ Control Panel")
st.sidebar.write("Quickly populate the input fields with standard sample cases to test the predictor.")

def load_preset():
    selected = st.session_state.preset_selection
    if selected in presets and presets[selected] is not None:
        values = presets[selected]
        for feat, val in zip(feature_names, values):
            st.session_state[feat] = float(val)

st.sidebar.selectbox(
    "Load Sample Case Data",
    options=list(presets.keys()),
    key="preset_selection",
    on_change=load_preset
)

st.sidebar.divider()
st.sidebar.write("### About the Predictor")
st.sidebar.info(
    "This system uses a **Gradient Boosting Classifier** trained on the Wisconsin Breast Cancer Dataset. "
    "It processes 30 physical features of breast mass fine needle aspirate (FNA) measurements to make predictions."
)

# Header Section
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🧬 Breast Cancer Diagnosis Predictor</h1>
    <div class="header-subtitle">Evaluate physical measurements of cell nuclei from breast mass aspirates to predict the likelihood of benign or malignant diagnosis.</div>
</div>
""", unsafe_allow_html=True)

st.write("### Enter Mass Measurements")
st.write("Fill in the fields below manually, or use the **Control Panel** in the sidebar to load preset cases.")

# Input Form layout
input_data = {}

# Tabs for 3 categories of measurements
tab1, tab2, tab3 = st.tabs(["📊 Mean Measurements", "📈 Standard Error (SE)", "🚨 Worst (Extreme) Measurements"])

with tab1:
    st.markdown('<div class="tab-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features_mean):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            label = feature.replace('_', ' ').title()
            input_data[feature] = st.number_input(
                f"**{label}**",
                key=feature,
                format="%.5f"
            )
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="tab-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features_se):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            label = feature.replace('_', ' ').upper()
            input_data[feature] = st.number_input(
                f"**{label}**",
                key=feature,
                format="%.6f"
            )
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="tab-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features_worst):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            label = feature.replace('_', ' ').title()
            input_data[feature] = st.number_input(
                f"**{label}**",
                key=feature,
                format="%.5f"
            )
    st.markdown('</div>', unsafe_allow_html=True)


# Convert and Scale using the populated input_data dictionary directly
input_df = pd.DataFrame([input_data])[feature_names]

# Buttons
col_btn, col_empty = st.columns([1, 2])
with col_btn:
    predict_clicked = st.button("Generate Diagnostic Prediction")

# Predict and Output
if predict_clicked:
    try:
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        
        # Predict Class and Probability
        prediction = gbc_model.predict(input_scaled)
        probabilities = gbc_model.predict_proba(input_scaled)[0]
        
        # Calculate confidences
        prob_malignant = probabilities[0]
        prob_benign = probabilities[1]
        
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="result-card benign">
                <div class="result-title">🟢 Prediction: Likely BENIGN</div>
                <div class="result-desc">
                    The model predicts that the cellular characteristics are consistent with a <strong>Benign</strong> (non-cancerous) tumor.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"#### Prediction Confidence Details")
            st.progress(float(prob_benign))
            st.write(f"🧬 **Benign Likelihood:** {prob_benign * 100:.2f}% | **Malignant Likelihood:** {prob_malignant * 100:.2f}%")
        else:
            st.markdown(f"""
            <div class="result-card malignant">
                <div class="result-title">🔴 Warning: Likely MALIGNANT</div>
                <div class="result-desc">
                    The model predicts that the cellular characteristics are consistent with a <strong>Malignant</strong> (cancerous) tumor.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"#### Prediction Confidence Details")
            st.progress(float(prob_malignant))
            st.write(f"🧬 **Malignant Likelihood:** {prob_malignant * 100:.2f}% | **Benign Likelihood:** {prob_benign * 100:.2f}%")
            
        # Clinical Disclaimer
        st.caption(
            "⚠️ **Disclaimer:** This predictor is a machine learning assistant trained on historical research datasets. "
            "It is meant to assist diagnostic workflows and is not a substitute for clinical analysis or professional medical advice."
        )
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")