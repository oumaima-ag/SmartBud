import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import csv
import io
import fitz
from summarizer import Summarizer
import base64

# Télécharger les ressources nécessaires pour NLTK and BERT
nltk.download('punkt')

# Fonction pour extraire la réponse à partir du modèle "question_answering"
def get_answer(question, context):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

# Function to create a download link for a file
def get_binary_file_downloader_html(file_path):
    with open(file_path, 'rb') as file:
        data = file.read()
    b64 = base64.b64encode(data).decode('utf-8')
    return f'<a href="data:file/csv;base64,{b64}" download="{file_path}" target="_blank">Download {file_path}</a>'

def main():
    # Initialize session state variables
    if 'article_text' not in st.session_state:
        st.session_state.article_text = None
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

    # Create a CSV file for storing the data
    csv_filename = "smartbud_data.csv"

    # Add a "Clear Session" button
    clear_session = st.button("Clear Session")
    if clear_session:
        st.session_state.article_text = None
        st.session_state.search_history = []

    link = st.text_input("Entrez le lien de l'article :")
    extract_button = st.button("Extraire le texte et générer le résumé")

    if extract_button:
        if link:
            if link.lower().endswith(".pdf"):
                response = requests.get(link)
                if response.status_code == 200:
                    pdf_stream = io.BytesIO(response.content)

                    pdf_file = fitz.open(stream=pdf_stream, filetype="pdf")
                    article_text = ""
                    for page_num in range(pdf_file.page_count):
                        page = pdf_file.load_page(page_num)
                        page_text = page.get_text()
                        article_text += page_text

                    max_length = 5120
                    if len(article_text) > max_length:
                        article_text = article_text[:max_length]

                    st.session_state.article_text = article_text
                    st.success("Texte extrait avec succès !")

                    summarizer = Summarizer()
                    summary = summarizer(article_text)
                    st.subheader("Résumé généré :")
                    st.write(summary)
            else:
                response = requests.get(link)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
                    st.session_state.article_text = article_text
                    st.success("Texte extrait avec succès !")

                    article_filename = link.replace("/", "").replace(":", "") + ".csv"

                    with open(article_filename, mode="w", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file, delimiter=";")
                        writer.writerow(["Lien", "Texte extrait"])
                        writer.writerow([link, article_text])

    # Save search history
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    if extract_button and link:
        st.session_state.search_history.append(link)

    # Display search history
    st.sidebar.title("Search History")
    if st.session_state.search_history:
        st.sidebar.write("Recent Searches:")
        for i, search in enumerate(st.session_state.search_history[::-1]):
            st.sidebar.write(f"{i + 1}. {search}")

    if "article_text" in st.session_state and st.session_state.article_text is not None:
        # Affichage du texte extrait et du résumé
        st.header("Texte extrait de l'article :")
        st.text_area("", value=st.session_state.article_text, height=300)

        # Générer le résumé
        summarizer = Summarizer()
        summary = summarizer(st.session_state.article_text)
        st.subheader("Résumé généré :")
        st.write(summary)

        user_input = st.text_input("Posez une question :")

        if st.button("Obtenir la réponse"):
            answer = get_answer(user_input, st.session_state.article_text)
            if answer.strip() == "" or answer == "[CLS]":
                answer = "La réponse n'existe pas dans cet article."
            st.subheader("Réponse:")
            st.write(answer)

            # Save data to CSV
            with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow([link, summary, user_input, answer])

    # User Feedback Section
    st.sidebar.title("User Feedback")
    feedback = st.sidebar.radio("Rate the Quality of the Extracted Summary or Answer:",
                               options=["Excellent", "Good", "Fair", "Poor"])
    if feedback:
        st.sidebar.write(f"Thank you for your feedback! You rated the quality as: {feedback}")

    # User Guide Section
    st.sidebar.title("User Guide")
    st.sidebar.markdown("Welcome to SmartBud! Here's how to use it:")
    st.sidebar.markdown("- Enter the link to the article you want to extract.")
    st.sidebar.markdown("- Click on 'Extract Text and Generate Summary' to get the article text and summary.")
    st.sidebar.markdown("- You can then ask questions about the article in the 'Ask a Question' section.")
    st.sidebar.markdown("- Click 'Obtain Answer' to get answers to your questions.")
    st.sidebar.markdown("- Use the 'Clear Session' button to start fresh.")



    # Download Data Section
    st.sidebar.title("Download Data")
    if st.button("Download Data"):
        st.sidebar.write(f"Downloading data as {csv_filename}...")
        st.sidebar.markdown(get_binary_file_downloader_html(csv_filename), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
