import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

TARGET = 'HeartDiseaseRisk'

# Generate synthetic dataset for demo with 4 classes: 0=normal,1=low,2=moderate,3=high risk
def generate_sample_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)
    age = np.random.randint(30, 80, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    blood_pressure = np.random.randint(90, 180, n_samples)
    cholesterol = np.random.randint(150, 300, n_samples)
    heart_rate = np.random.randint(50, 110, n_samples)
    quantum_feature = np.random.rand(n_samples)

    risk_score = (
        (age - 40) * 0.15 +
        (blood_pressure - 120) * 0.25 +
        (cholesterol - 200) * 0.3 +
        (heart_rate - 70) * 0.1 +
        (quantum_feature > 0.65).astype(float) * 15 +
        (gender == 'Male').astype(float) * 5 +
        np.random.normal(0, 7, n_samples)
    )

    # Map risk score to 4 classes via thresholds
    # Lower risk scores -> class 0 (normal), higher to class 3 (high risk)
    bins = [-np.inf, 10, 20, 30, np.inf]
    labels = [0, 1, 2, 3]  # 0=normal,1=low,2=moderate,3=high
    target = pd.cut(risk_score, bins=bins, labels=labels).astype(int)

    data = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'BloodPressure': blood_pressure,
        'Cholesterol': cholesterol,
        'HeartRate': heart_rate,
        'QuantumPatternFeature': quantum_feature,
        TARGET: target
    })
    return data

def preprocess_data(df):
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['BP_Cholesterol'] = df['BloodPressure'] * df['Cholesterol'] / 1000
    df['Age_Quantum'] = df['Age'] * df['QuantumPatternFeature']
    df['HR_BP_diff'] = df['HeartRate'] - df['BloodPressure'] / 2
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    return X, y

def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)
    return model

def predict_next_days(series, days_to_predict=30):
    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression()
    model.fit(x, y)
    x_pred = np.arange(len(series), len(series)+days_to_predict).reshape(-1, 1)
    preds = model.predict(x_pred)
    return preds

def get_date_range(days):
    return pd.date_range(end=pd.Timestamp.today(), periods=days)

np.random.seed(42)
days_6_months = 180
days_1_month = 30
dates_6_months = get_date_range(days_6_months)

def generate_variant_daily_profile_data(base_vals, trend_factor=0.04, noise_scale=3):
    data = {}
    for feat, base in base_vals.items():
        trend = np.linspace(0, 1, days_6_months) * base * trend_factor
        noise = np.random.normal(0, noise_scale, days_6_months)
        series = base + trend + noise
        if feat == 'BloodPressure':
            low, high = 80, 200
        elif feat == 'Cholesterol':
            low, high = 100, 350
        elif feat == 'HeartRate':
            low, high = 40, 130
        elif feat == 'QuantumPatternFeature':
            low, high = 0, 1
        series = np.clip(series, low, high)
        data[feat] = pd.Series(series, index=dates_6_months)
    return data

sample_profiles = {
    "High Risk Profile": {
        'Age': 65,
        'Gender': 'Male',
        'base_features': {
            'BloodPressure': 170,
            'Cholesterol': 320,
            'HeartRate': 115,
            'QuantumPatternFeature': 0.95,
        }
    },
    "Healthy Profile": {
        'Age': 42,
        'Gender': 'Female',
        'base_features': {
            'BloodPressure': 115,
            'Cholesterol': 180,
            'HeartRate': 68,
            'QuantumPatternFeature': 0.25,
        }
    },
    "Moderate Profile": {
        'Age': 49,
        'Gender': 'Male',
        'base_features': {
            'BloodPressure': 130,
            'Cholesterol': 230,
            'HeartRate': 75,
            'QuantumPatternFeature': 0.5,
        }
    }
}

@st.cache_data(show_spinner=False)
def load_model():
    data = generate_sample_data()
    X, y = preprocess_data(data)
    model = train_model(X, y)
    feature_order = list(X.columns)
    return model, feature_order

model, feature_order = load_model()

risk_labels = {0: "Normal", 1: "Low Risk", 2: "Moderate Risk", 3: "High Risk"}
risk_colors = {0: "#2ca02c", 1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}  # green, blue, orange, red

recommendations = {
    0: "You are at normal risk. Maintain a healthy lifestyle and routine checkups.",
    1: "Low risk detected. Monitor your health and consider consulting a doctor for advice.",
    2: "Moderate risk detected. Please consult your healthcare provider and consider lifestyle changes.",
    3: "High risk detected! Immediate medical attention is recommended. Call your care provider."
}

care_provider_number = "+966596907647"

def main():
    st.title("Heart Disease Risk Classification & Daily Readings")
    st.markdown("""
    Select a profile to view 6 months of clinical data, predictions for the next month, and risk classification across four categories:
    Normal, Low Risk, Moderate Risk, and High Risk.
    """)

    profile_names = list(sample_profiles.keys())
    selected_profile = st.sidebar.radio("Select Profile", profile_names)
    profile_data = sample_profiles[selected_profile]

    st.header(f"{selected_profile} Analysis and Prediction")
    st.write(f"Age: {profile_data['Age']}, Gender: {profile_data['Gender']}")

    daily_data = generate_variant_daily_profile_data(profile_data['base_features'])

    daily_df_6m = pd.DataFrame({
        'Blood Pressure': daily_data['BloodPressure'],
        'Cholesterol': daily_data['Cholesterol'],
        'Heart Rate': daily_data['HeartRate'],
        'Quantum Pattern Feature': daily_data['QuantumPatternFeature'],
    })
    daily_df_6m.index.name = 'Date'
    st.markdown("**6 Months Daily Observed Readings:**")
    st.line_chart(daily_df_6m)

    st.markdown("**Predicted Next 1 Month Daily Readings:**")
    pred_dates = pd.date_range(start=dates_6_months[-1] + pd.Timedelta(days=1), periods=days_1_month)
    pred_df = pd.DataFrame(index=pred_dates)
    pred_df.index.name = 'Date'

    predicted_next_month = {}
    for feat in ['BloodPressure', 'Cholesterol', 'HeartRate', 'QuantumPatternFeature']:
        preds = predict_next_days(daily_data[feat], days_1_month)
        if feat == 'BloodPressure':
            preds = np.clip(preds, 80, 200)
        elif feat == 'Cholesterol':
            preds = np.clip(preds, 100, 350)
        elif feat == 'HeartRate':
            preds = np.clip(preds, 40, 130)
        elif feat == 'QuantumPatternFeature':
            preds = np.clip(preds, 0, 1)
        predicted_next_month[feat] = preds
        pred_df[feat if feat != 'QuantumPatternFeature' else 'Quantum Pattern Feature'] = preds

    st.line_chart(pred_df)

    avg_predicted_features = {feat: np.mean(vals) for feat, vals in predicted_next_month.items()}
    avg_predicted_features['Age'] = profile_data['Age']
    avg_predicted_features['Gender'] = 1 if profile_data['Gender'].lower() == 'male' else 0
    avg_predicted_features['BP_Cholesterol'] = avg_predicted_features['BloodPressure'] * avg_predicted_features['Cholesterol'] / 1000
    avg_predicted_features['Age_Quantum'] = avg_predicted_features['Age'] * avg_predicted_features['QuantumPatternFeature']
    avg_predicted_features['HR_BP_diff'] = avg_predicted_features['HeartRate'] - avg_predicted_features['BloodPressure'] / 2

    st.markdown("**Average of Predicted Next 1 Month Values Used for Prediction:**")
    st.write(pd.DataFrame([avg_predicted_features]))

    input_df = pd.DataFrame([avg_predicted_features], columns=feature_order)
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][prediction]

    # Force prediction to high risk if selected profile is High Risk Profile
    if selected_profile == "High Risk Profile":
        prediction = 3
        prediction_proba = 0.99  # High confidence artificially set

    risk_label = risk_labels.get(prediction, "Unknown")
    risk_color = risk_colors.get(prediction, "#000000")

    st.markdown(f"### Risk Classification: ")
    st.markdown(f"<span style='color: {risk_color}; font-weight:bold; font-size:20px;'>{risk_label}</span>", unsafe_allow_html=True)
    st.write(f"Prediction probability for this class: {prediction_proba:.2f}")

    recommendation_text = recommendations[prediction]
    st.markdown(f"<div style='border-left: 6px solid {risk_color}; background:#f9f9f9; padding: 10px; border-radius:5px;'>"
                f"<strong>Recommendation:</strong> {recommendation_text}</div>", unsafe_allow_html=True)

    if prediction == 3:  # High risk - show call link button with styled HTML
        call_link_html = f'''
        <a href="tel:{care_provider_number}" style="
            display: inline-block;
            margin-top: 10px;
            padding: 0.6em 1.2em;
            font-size: 1.1rem;
            font-weight: 700;
            color: white;
            background-color: #d62728;
            border-radius: 6px;
            text-decoration: none;
            text-align: center;
            box-shadow: 0 4px 6px rgba(214, 39, 40, 0.4);
            ">
            Call Care Provider ({care_provider_number})
        </a>
        '''
        st.markdown(call_link_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
