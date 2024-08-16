import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


prompt_template = """
  Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
  provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
"""


prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])


def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def get_pdf_text(path):
    loader = PyPDFDirectoryLoader("pdf/")
    data = loader.load_and_split()
    context = "\n".join(str(p.page_content) for p in data)

    return context


def get_text_chunks(context):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_text(context)
    return texts


def user_input(user_question):
    data = get_pdf_text("pdf")
    chunks = get_text_chunks(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_texts(chunks, embeddings).as_retriever()
    question = user_question
    docs = vector_index.get_relevant_documents(question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs,"question":question},
        return_only_output=True)
    print(response)
    st.write("reply: ",response["output_text"])



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