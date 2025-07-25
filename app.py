import streamlit as st
from transformers import pipeline

# This is a special Streamlit command to cache the model.
# It means the model will only be loaded once, making the app faster.
@st.cache_resource
def load_model():
    """Loads the sentiment analysis model from Hugging Face."""
    return pipeline("sentiment-analysis")

# --- App UI and Logic ---

# Set the title of the web app
st.title("Sentiment Analysis App 🤖")

# Add a description
st.write("Enter text to see if its sentiment is POSITIVE or NEGATIVE.")

# Load the model using our function
sentiment_pipeline = load_model()

# Create a text area for user input
user_input = st.text_area("Your text here:")

# Create a button that the user will click to analyze the text
if st.button("Analyze"):
    if user_input:
        # If there is input, run it through the pipeline
        result = sentiment_pipeline(user_input)[0]
        label = result['label']
        score = result['score']

        # Display the results
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence:** {score:.4f}") # Display score with 4 decimal places
    else:
        # If there is no input, show a warning
        st.warning("Please enter some text to analyze.")