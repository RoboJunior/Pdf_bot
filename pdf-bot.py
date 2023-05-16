import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os



def main():
    load_dotenv()
    
    st.set_page_config(page_title='PDF-Chatbot',page_icon="🤖")

    st.header("How can i help you?")

    pdf = st.file_uploader("Upload your pdf here ",type="pdf")

    #extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        #split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap=200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)
        
        #creating embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks,embedding=embeddings)

        user_input = st.text_input("Ask your question ? ")
        if user_input:
            docs = knowledge_base.similarity_search(user_input)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            response = chain.run(input_documents=docs,question=user_input)
            st.write(response)

if __name__ == "__main__":
    main()

