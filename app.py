import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from langdetect import detect
from googletrans import Translator
import time

# Custom NLTK data path for deployment
nltk.data.path.append('./nltk_data')

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
translator = Translator()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('voting_model.pkl', 'rb'))

# Streamlit interface
st.set_page_config(page_title="Spam Detector", page_icon="🚫", layout="centered")

st.title("📩 Multilingual SMS Spam Classifier")

input_sms = st.text_area("✉️ Enter your message (English, Bengali, Hindi supported):", height=150)



if st.button("🔍 Predict"):
    with st.spinner("Processing... Please wait."):
        try:
            lang = detect(input_sms)
            if lang != 'en':
                translated = translator.translate(input_sms, dest='en')
                translated_text = translated.text
                st.info(f"🌐 Translated to English: {translated_text}")
                final_input = translated_text
            else:
                final_input = input_sms

            # Preprocess
            transformed_sms = transform_text(final_input)

            # Vectorize and Predict
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            # Display result
            st.markdown("---")
            if result == 1:
                st.error("🚫 This message is **SPAM**!", icon="🚫")
            else:
                st.success("✅ This message is **NOT SPAM**!", icon="✅")
            st.markdown("---")
        except:
            st.error("❌ Could not detect or translate the message. Please enter valid text.")
