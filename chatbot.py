import streamlit as st
import requests

st.set_page_config(page_title="AEC / BIM Chatbot", page_icon="🏗")

st.title("🏗 AEC / BIM Chatbot – Free Edition")

api_key = st.secrets["GROQ_API_KEY"]

def ask_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",   # FREE MODEL
        "messages": [
            {"role": "system", "content": "You are an AEC + BIM expert assistant. Answer like a professional architect/engineer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, json=data, headers=headers)
    return response.json()["choices"][0]["message"]["content"]

user_input = st.text_area("Ask something about AEC / BIM:")

if st.button("Send"):
    if user_input.strip():
        st.write("### Answer:")
        st.write(ask_groq(user_input))
