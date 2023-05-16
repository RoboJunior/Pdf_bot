# Pdf_bot
This project is about langchain model which is able to answer any kind of answer thrown to the model from the uploaded pdf .

It works by extracting the text from the pdf and splits it into chunks and the chunks are further converted into embedding which is vector representation of the meaning of the text and which is going to be stored in your knowledge base when a user asks a question to the model it embedds the user query and converts it into same vector representation which the performs a schematic search algorithm here we are using FAISS algorithm(Facebook AI Similarity Search) which searches for the match and using LLM(Large Language Model) like Chatgpt the ranked results are shown to the user as text
