import streamlit as st
import pickle

# Load model, vectorizer, dan encoder
with open("svm_model_dana.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# UI
st.set_page_config(page_title="Analisis Sentimen DANA", layout="centered")
st.title("🔍 Analisis Sentimen pada Aplikasi DANA")
st.markdown("Masukkan ulasan pengguna, dan sistem akan memprediksi apakah sentimennya positif, negatif, atau netral.")

# Input
text = st.text_area("Masukkan ulasan:", height=150)

if st.button("Analisis Sentimen"):
    if text.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")
    else:
        vector = vectorizer.transform([text])
        pred_int = model.predict(vector)[0]
        pred_label = label_encoder.inverse_transform([pred_int])[0]
        st.success(f"Prediksi Sentimen: **{pred_label.capitalize()}**")
