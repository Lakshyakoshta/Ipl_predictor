import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

# Load model and preprocessing tools
model = load_model('ipl_model.h5')
scaler = joblib.load('scaler.pkl')
venue_encoder = joblib.load('venue_encoder.pkl')
batting_team_encoder = joblib.load('batting_team_encoder.pkl')
bowling_team_encoder = joblib.load('bowling_team_encoder.pkl')
striker_encoder = joblib.load('striker_encoder.pkl')
bowler_encoder = joblib.load('bowler_encoder.pkl')

# Load dataset
df = pd.read_csv('ipl_data.csv')

# --- Page Config ---
st.set_page_config(page_title="IPL Score Predictor", layout="centered")

# --- Custom CSS for a sporty vibe ---
st.markdown("""
    <style>
    .main {
        background-color: #101820;
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #F2AA4C;
    }

    .stButton>button {
        background-color: #F2AA4C;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 2em;
        font-size: 16px;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #ffc107;
        color: black;
        transform: scale(1.05);
    }

    .stSelectbox>div>div {
        font-size: 16px;
        color: black;
    }

    .block-container {
        padding-top: 2rem;
    }

    .big-score {
        font-size: 48px;
        font-weight: 700;
        color: #00ffcc;
        text-align: center;
        margin-top: 20px;
    }

    .header-banner {
        background: url('https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_480,q_50/esci/media/motion/2024/0413/dm_240413_INTL_IPL_Banner.jpg') no-repeat center;
        background-size: cover;
        padding: 4rem 2rem;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .header-banner h1 {
        color: white;
        font-size: 3rem;
        text-shadow: 2px 2px #000;
    }

    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<div class="header-banner"><h1>🏆 IPL Score Predictor</h1><p>Smash into the future of cricket prediction!</p></div>', unsafe_allow_html=True)

# --- Input Layout ---
st.markdown("### 🏏 Match Details")
col1, col2 = st.columns(2)

with col1:
    venue = st.selectbox("🏟️ Venue", sorted(df['venue'].unique()), help="Select the stadium")
    batting_team = st.selectbox("🧤 Batting Team", sorted(df['bat_team'].unique()), help="Who's batting?")
    striker = st.selectbox("👤 Striker", sorted(df['batsman'].unique()), help="Batsman currently on strike")

with col2:
    bowling_team = st.selectbox("🎯 Bowling Team", sorted(df['bowl_team'].unique()), help="Who's bowling?")
    bowler = st.selectbox("🥎 Bowler", sorted(df['bowler'].unique()), help="Bowler currently delivering")

# --- Predict Button ---
st.markdown("### 🔮 Prediction")
if st.button("🔥 Predict Now"):
    try:
        # Encode inputs
        input_data = np.array([
            venue_encoder.transform([venue])[0],
            batting_team_encoder.transform([batting_team])[0],
            bowling_team_encoder.transform([bowling_team])[0],
            striker_encoder.transform([striker])[0],
            bowler_encoder.transform([bowler])[0]
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        predicted_score = int(prediction[0, 0])

        # Display Result
        st.markdown(f"<div class='big-score'>🏏 Predicted Score: {predicted_score}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

# --- Footer ---
st.markdown("---")

