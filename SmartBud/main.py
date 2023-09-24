import streamlit as st
import smartchat

def main():
    st.set_page_config(
        page_title="SmartBud - Extraction des données par ChatBot",
        page_icon="🤖",  # Chatbot icon
        layout="wide",
    )

    # Logo and Title
    st.title("🤖 SmartBud - Extraction des données par ChatBot")
    st.markdown("#### Innovation in Data Extraction")  # Add a subtitle

    # Load SmartChat
    smartchat.main()

if __name__ == "__main__":
    main()
