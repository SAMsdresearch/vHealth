import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

TARGET = 'HeartDisease'

# Generate synthetic dataset for demo (hidden in UI)
def generate_sample_data(n_samples=500, random_state=42):
    np.random.seed(random_state)
    data = pd.DataFrame({
        'Age': np.random.randint(30, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'BloodPressure': np.random.randint(90, 180, n_samples),
        'Cholesterol': np.random.randint(150, 300, n_samples),
        'HeartRate': np.random.randint(50, 110, n_samples),
        'QuantumPatternFeature': np.random.rand(n_samples),
    })
    data[TARGET] = ((data['Age'] > 50) & 
                    (data['BloodPressure'] > 130) | 
                    (data['Cholesterol'] > 240) | 
                    (data['QuantumPatternFeature'] > 0.7)).astype(int)
    return data

def preprocess_data(df):
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    return X, y

def train_model(X, y):
    model = RandomForestClassifier(random_state=42)
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
    "Sick Profile": {
        'Age': 58,
        'Gender': 'Male',
        'base_features': {
            'BloodPressure': 145,
            'Cholesterol': 270,
            'HeartRate': 90,
            'QuantumPatternFeature': 0.75,
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

def main():
    st.title("Heart Disease Daily Readings & Prediction")
    st.markdown("""
    This app presents daily clinical indicator readings over 6 months for two profiles (Sick and Healthy), 
    predicts the next 1 month daily values, and assesses heart disease risk.  
    If the risk is high, it recommends calling a care provider via a button.
    """)

    care_provider_number = "+966596907647"

    st.header("Profiles Analysis and Prediction")

    for profile_name, profile_data in sample_profiles.items():
        st.subheader(profile_name)
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

        predicted_next_month = {}
        st.markdown("**Predicted Next 1 Month Daily Readings:**")

        pred_dates = pd.date_range(start=dates_6_months[-1] + pd.Timedelta(days=1), periods=days_1_month)
        pred_df = pd.DataFrame(index=pred_dates)
        pred_df.index.name = 'Date'

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

        st.markdown("**Average of Predicted Next 1 Month Values (Used for Prediction):**")
        st.write(pd.DataFrame([avg_predicted_features]))

        input_df = pd.DataFrame([avg_predicted_features], columns=feature_order)
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][prediction]

        if prediction == 1:
            st.error(f"Prediction: HIGH risk of Heart Disease detected with probability {prediction_proba:.2f}!")
            st.warning(f"**Recommendation:** Please call your care provider immediately: {care_provider_number}")
            # Replace button with clickable link for mobile call
            call_link_html = f'''
            <a href="tel:{care_provider_number}" style="
                display: inline-block;
                padding: 0.5em 1em;
                font-size: 1rem;
                font-weight: 600;
                color: white;
                background-color: #e63946;
                border-radius: 5px;
                text-decoration: none;
                text-align: center;
                ">
                Call Care Provider ({care_provider_number})
            </a>
            '''
            st.markdown(call_link_html, unsafe_allow_html=True)
        else:
            st.success(f"Prediction: LOW risk of Heart Disease with probability {prediction_proba:.2f}.")

        st.markdown("---")

    st.caption("Note: This model and data are simulations for demonstration only and not intended for medical use.")

if __name__ == "__main__":
    main()
