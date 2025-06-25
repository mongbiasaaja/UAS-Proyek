import streamlit as st
import pickle

# Load model dan vectorizer
model = pickle.load(open("svm_model_dana.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("Analisis Sentimen Aplikasi DANA")
st.markdown("Masukkan ulasan pengguna dan klik tombol untuk mengetahui sentimen (positif, negatif, atau netral).")

# Input dari user
text = st.text_area("Tulis ulasan di sini:")

if st.button("Analisis Sentimen"):
    if text.strip() == "":
        st.warning("Silakan masukkan ulasan terlebih dahulu.")
    else:
        # Preprocess sesuai dengan training
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]
        st.success(f"Prediksi Sentimen: **{prediction.upper()}**")
