import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True) 


# --- Load Model and Tokenizer (this is done only once) ---
@st.cache_resource
def load_model():
    """Loads the fine-tuned BERT model and tokenizer from the saved directory."""
    try:
        model_path = './saved_model'
        model = TFBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except OSError:
        st.error("Model not found. Ensure the 'saved_model' directory is in the same folder as app.py.")
        return None, None

model, tokenizer = load_model()

# --- Text Preprocessing Function ---
# This function must be identical to the one used in training
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+', '', text) # Remove URLs, mentions, hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Streamlit App User Interface ---
st.set_page_config(page_title="Fake News Detector", page_icon="🔎", layout="wide")
st.title("🔎 Fake News Detector")
st.markdown("This application uses a fine-tuned BERT model to analyze news text and classify it as **Real** or **Fake**.")

user_input = st.text_area("Enter a news article or text to analyze:", "", height=250)

if st.button("Analyze News"):
    if model is not None and tokenizer is not None:
        if user_input.strip():
            with st.spinner('Analyzing... This may take a moment.'):
                # 1. Preprocess the user's input
                cleaned_input = preprocess_text(user_input)

                # 2. Tokenize the text for BERT
                inputs = tokenizer(cleaned_input, max_length=128, padding='max_length', truncation=True, return_tensors="tf")

                # 3. Make a prediction
                outputs = model(inputs)
                prediction = tf.nn.softmax(outputs.logits, axis=1)
                prediction_class = tf.argmax(prediction, axis=1).numpy()[0]

                # 4. Display the result
                st.subheader("Analysis Result")
                if prediction_class == 1: # 1 corresponds to 'Real'
                    st.success("✅ This appears to be **Real News**.")
                else: # 0 corresponds to 'Fake'
                    st.error("❌ This appears to be **Fake News**.")
        else:
            st.warning("Please enter some text to analyze.")