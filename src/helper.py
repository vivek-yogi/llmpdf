import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import google.generativeai as genai 

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(path):
    loader = PyPDFDirectoryLoader(path)
    data   = loader.load_and_split()
    context = "\n".join(str(p.page_content) for p in data)
    return context

def get_text_chunks(context):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(context)
    return texts

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model = 'gemini-pro',
                                   temperature =0.3)
    prompt = PromptTemplate(template = prompt_template,
                            input_variables = ['context','question'])
    chain = load_qa_chain(model,chain_type = 'stuff',prompt=prompt)
    return chain

