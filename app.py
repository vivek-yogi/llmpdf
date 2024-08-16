import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_conversational_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def user_input(user_question):
    data = get_pdf_text("pdf")
    chunks = get_text_chunks(data)

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_index = Chroma.from_texts(chunks, embeddings).as_retriever()

    question = user_question
    docs = vector_index.get_relevant_documents(question)

    chain = get_conversational_chain()

    response = chain(
    {"input_documents":docs, "question": question}
    , return_only_outputs=True)

    print(response)

    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        uploaded_file = st.file_uploader("Choose a file")
        os.makedirs("pdf", exist_ok=True)
        if uploaded_file is not None:
            pdf_file_path = "pdf/mypdf.pdf"
            print(pdf_file_path)
            bytes_data = uploaded_file.getvalue()
            with open(pdf_file_path, 'wb') as f: 
                f.write(bytes_data)
                st.success("Done")



if __name__ == "__main__":
    main()