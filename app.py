import streamlit as st
import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

# Initialize OpenAI and Chroma clients
openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_store_wu")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

client = OpenAI(api_key=openai_key)

# Function to query documents (same as in your original script)
"""def query_documents_modified(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    return relevant_chunks"""
# Function to query documents
def query_documents(question, n_results=2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results) # no of the results to be returned basically if its 2 then 2 most relevant chunks will be returned ie 2 docs. ie it can be chunk1 and chunk2 of single news.

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks, results

# Function to generate a response (same as in your original script)
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "Du bist ein Assistent für Frage-Antwort-Aufgaben. Verwende die folgenden abgerufenen Kontextstücke, um die Frage zu beantworten."
        "Wenn du die Antwort nicht weißt, sage, dass du es nicht weißt."
        "Verwende maximal drei Sätze und halte die Antwort kurz."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message.content
    return answer

# Streamlit App
def main():
    st.title("Chatbot der Multifunktions-Arena Würzburg")
    
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_question = st.chat_input("Stellen Sie Ihre Frage zur zukünftigen Arena von Würzburg")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Process new user question
    if user_question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_question
        })
        
        # Retrieve relevant chunks and generate response
        try:
            #relevant_chunks = query_documents_modified(user_question)
            returned = query_documents(user_question)
            relevant_chunks = returned[0]
            response = generate_response(user_question, relevant_chunks)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response
            })
        
        except Exception as e:
            st.error(f"Ein Fehler ist aufgetreten: {str(e)}")
    
    # Optional: Add a sidebar with document information
    st.sidebar.title("Dokumenten-Informationen")
    st.sidebar.info("Dieser Chatbot basiert auf einem Retrieval Augmented Generation (RAG) System.")
    st.sidebar.info("Gesamtzahl der indizierten Nachrichtenartikel: 528")
    st.sidebar.info(f"Gesamtzahl der indizierten chunks: {collection.count()}")
    

if __name__ == "__main__":
    main()