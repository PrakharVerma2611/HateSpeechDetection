import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('xgb_hate_speech_model.pkl')
vectorizer = joblib.load('xgb_tfidf_vectorizer.pkl')

def predict_hate(text):
    cleaned = text.lower()
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]
    label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
    return label_map[label]

# Streamlit UI
st.set_page_config(page_title="Hate Speech Detector", page_icon="🧠")
st.title("🧠 Hate Speech Detection")
st.markdown("Enter a sentence to check if it contains hate speech or offensive language.")

user_input = st.text_area("Your text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        result = predict_hate(user_input)
        st.success(f"📝 Prediction: **{result}**")
