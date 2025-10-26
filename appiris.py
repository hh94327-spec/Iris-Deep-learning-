# app.py — Minimal Streamlit for Iris (Keras model + pickle preprocess)
import json, pickle, numpy as np, pandas as pd, streamlit as st, tensorflow as tf

ART = "artifacts_keras"
MODEL_FILE = f"{ART}/model_keras_tuned.keras"  # or model_keras.keras if you prefer baseline
SCALER_FILE = f"{ART}/scaler_X.pkl"
LE_FILE = f"{ART}/label_encoder_y.pkl"
META_FILE = f"{ART}/metadata.json"

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = pickle.load(open(SCALER_FILE, "rb"))
    le = pickle.load(open(LE_FILE, "rb"))
    meta = json.load(open(META_FILE, "r", encoding="utf-8"))
    return model, scaler, le, meta

st.set_page_config(page_title="Iris Classifier (Keras)", page_icon="🌸", layout="centered")
st.title("🌸 Iris Classifier (Keras)")

model, scaler, le, meta = load_artifacts()
feature_cols = meta["feature_columns"]

with st.form("iris_form"):
    st.write("**Enter feature values** (match training order)")
    c1, c2 = st.columns(2)
    v = {}
    for i, col in enumerate(feature_cols):
        default_val = 0.0
        (c1 if i % 2 == 0 else c2)
        v[col] = (c1 if i % 2 == 0 else c2).number_input(col, value=float(default_val), step=0.1, format="%.2f")
    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([v], columns=feature_cols)
    Xs = scaler.transform(X.values)
    prob = model.predict(Xs, verbose=0)[0]
    pred_idx = int(np.argmax(prob))
    pred_name = le.inverse_transform([pred_idx])[0]

    st.markdown(f"### ✅ Prediction: **{pred_name}**")
    st.write("**Probabilities:**")
    for name, p in zip(le.classes_, prob):
        st.write(f"- {name}: {100*p:.2f}%")
