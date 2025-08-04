import streamlit as st
import pickle
import pandas as pd
import os

# --- Custom CSS for background and header font ---
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Sans+Sheriff&display=swap');
#
#     /* Set full-page background */
#     .stApp {
#         background-image: url('https://i.pinimg.com/1200x/92/98/f5/9298f5cccf709b5b51104f496fe4b8e5.jpg');
#         background-size: cover;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }
#
#     /* Header styling */
#     h1 {
#         font-family: 'Sans Sheriff', Arial, sans-serif;
#         color: #DCCFE3 !important;
#         letter-spacing: 2px;
#         background: none !important;
#         padding: 0.5rem 0;
#         border-radius: 8px;
#         margin-bottom: 0.5em;
#     }
#
#     /* Central block container styling */
#     .block-container {
#         background-color: rgba(0,0,0,0.75) !important;
#         border-radius: 15px;
#         padding: 2rem;
#         margin-top: 2rem;
#     }
#
#     /* Make text readable */
#     .stMarkdown, .stButton, .stSlider, .stSelectbox {
#         color: white !important;
#     }
#
#     /* SLIDER CUSTOM STYLES START */
#     div[data-baseweb="slider"] {
#         color: #DCCFE3 !important;
#     }
#
#     div[data-baseweb="slider"] .css-1n76uvr {
#         background-color: #DCCFE3 !important;
#     }
#
#     div[data-baseweb="slider"] .css-1o3elg4 {
#         background-color: #5c5470 !important;
#     }
#
#     div[data-baseweb="slider"] .css-14g5p5u {
#         background-color: #DCCFE3 !important;
#         border: 2px solid #fff;
#     }
#
#     div[data-baseweb="slider"] .css-1h77wgb {
#         background-color: #DCCFE3 !important;
#         color: black !important;
#     }
#     /* SLIDER CUSTOM STYLES END */
#     </style>
# """, unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sans+Sheriff&display=swap');

    /* Set full-page background */
    .stApp {
        background-image: url('https://i.pinimg.com/1200x/92/98/f5/9298f5cccf709b5b51104f496fe4b8e5.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    h1 {
        font-family: 'Sans Sheriff', Arial, sans-serif;
        color: #DCCFE3 !important;
        letter-spacing: 2px;
    }

    .block-container {
        background-color: rgba(0, 0, 0, 0.75) !important;
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
    }

    /* Make text readable */
    .stMarkdown, .stButton, .stSelectbox {
        color: white !important;
    }

    /* --- SLIDER COLOR FIX START --- */

    /* Change thumb color */
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #DCCFE3 !important;
        border: 2px solid white;
    }

    /* Change track color */
    div[data-baseweb="slider"] > div > div {
        background: linear-gradient(to right, #dccfe3, #5c5470) !important;
    }

    /* Value popover tooltip */
    div[data-baseweb="slider"] [data-testid="stTooltipLabel"] {
        color: #DCCFE3 !important;
    }

    /* Hover effect (optional) */
    div[data-baseweb="slider"] [role="slider"]:hover {
        box-shadow: 0 0 10px #dccfe3;
        transform: scale(1.05);
        transition: 0.2s;
    }

    /* --- SLIDER COLOR FIX END --- */
    </style>
""", unsafe_allow_html=True)



# Title with Sans Sheriff font and magenta color
st.title("Traffic Type Prediction")

# Description below header
st.markdown("""
<div style="font-family: 'Sans Sheriff', Arial, sans-serif; color: #fff; font-size: 1.1rem; margin-bottom: 2rem;">
    <b>About this app:</b> <br>
    This machine learning model, trained by <b>Tulika</b>, predicts whether network traffic is
<span style="color:#00e676;"><b>Normal</b></span> or <span style="color:#ff1744;"><b>Suspicious</b></span>.<br>
    <b>How to use:</b><br>
    - Enter the network data: Bytes In, Bytes Out, Creation and Event times, and Source Country.<br>
    - Click <b>Predict</b> to get a real-time verdict.<br>
    <b>Output:</b> The model will analyze your input and classify the traffic accordingly.<br>
</div>
""", unsafe_allow_html=True)

# --- Your app logic continues below ---
MODEL_PATH = "rf_model.pkl"
COLUMNS_PATH = "model_columns.pkl"

@st.cache_resource
def load_model_and_columns():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        st.error("Model files are missing. Please upload rf_model.pkl and model_columns.pkl.")
        return None, None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(COLUMNS_PATH, "rb") as f:
            model_columns = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
    return model, model_columns

model, model_columns = load_model_and_columns()
country_cols = ["AE", "AT", "CA", "DE", "IL", "NL", "US"]

bytes_in = st.number_input("Bytes In", min_value=0)
bytes_out = st.number_input("Bytes Out", min_value=0)
creation_hour = st.slider("Creation Hour", 0, 23)
creation_day = st.slider("Creation Day", 1, 31)
event_hour = st.slider("Event Hour", 0, 23)
event_day = st.slider("Event Day", 1, 31)
country = st.selectbox("Source Country", country_cols)

def predict_traffic(bytes_in, bytes_out, creation_hour, creation_day, event_hour, event_day, country):
    bytes_total = bytes_in + bytes_out
    country_encoding = [1 if c == country else 0 for c in country_cols]
    input_data = [
        bytes_in, bytes_out, creation_hour, creation_day,
        event_hour, event_day, bytes_total
    ] + country_encoding
    df = pd.DataFrame([input_data], columns=model_columns)
    try:
        result = model.predict(df)[0]
        return result
    except Exception as e:
        return f"Prediction error: {e}"

if st.button("Predict"):
    if model is not None and model_columns is not None:
        prediction = predict_traffic(bytes_in, bytes_out, creation_hour, creation_day, event_hour, event_day, country)
        if prediction == 1:
            st.success('Predicted as: <span style="color:#ff1744;"><b>Suspicious</b></span>', unsafe_allow_html=True)
        elif prediction == 0:
            st.success('Predicted as: <span style="color:#00e676;"><b>Normal</b></span>', unsafe_allow_html=True)
        else:
            st.error(prediction)
    else:
        st.error("Could not run prediction. Model or columns missing.")


