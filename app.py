import streamlit as st
import joblib
import re
import pandas as pd

# Page config
st.set_page_config(page_title="Fake Review Detector", page_icon="🔍")
st.title("🔍 Fake Review Detector")
st.write("Enter a review to check if it's FAKE or REAL")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/fake_review_detector.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Simple text cleaning - NO NLTK!
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Input
review = st.text_area("📝 Enter your review:", height=150)

# Example reviews
example = st.selectbox(
    "Or try an example:",
    ["", 
     "Buy buy buy buy buy!!! Best product ever!!!",
     "Works as expected, good quality for the price.",
     "OMG amazing!!! Perfect perfect!!! Love it!!!",
     "The product arrived on time and works fine."]
)

if example and not review:
    review = example

# Analyze button
if st.button("🔍 Check Review", type="primary"):
    if review:
        cleaned = clean_text(review)
        prediction = model.predict([cleaned])[0]
        
        st.markdown("---")
        st.subheader("📊 Result")
        
        if prediction == 1:
            st.error("### 🚨 FAKE REVIEW")
            st.markdown("⚠️ This appears to be **computer-generated**")
        else:
            st.success("### ✅ REAL REVIEW")
            st.markdown("👍 This appears to be **human-written**")
        
        # Show confidence
        try:
            proba = model.predict_proba([cleaned])[0]
            confidence = proba[1] if prediction == 1 else proba[0]
            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(confidence)
        except:
            pass
    else:
        st.warning("Please enter a review")