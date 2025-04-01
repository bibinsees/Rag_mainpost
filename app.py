import streamlit as st
import re
import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions



# Initialize OpenAI and Chroma clients
import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


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


# Function to query documents
def query_documents(question, n_results=2):
    # Assuming collection.query returns the result (same as your code)
    results = collection.query(query_texts=question, n_results=n_results) 

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    
    # Extract the IDs (cleaned)
    ids = results['ids'][0]  # Extract the list of IDs
    cleaned_ids = [re.sub(r'_chunk\d+', '', id) for id in ids]
    
    # Define the path to the text files
    path = r'C:\Users\k54739\RAG\Rag_mainpost\wu_txts'

    # Regular expressions to extract publication date
    pub_date_pattern = r"Published Date of above article:(.*)"
    
    # Prepare the citation data (ID and publication date)
    citation_data = []
    
    for file_name in cleaned_ids:
        file_path = os.path.join(path, file_name)
        file_path = os.path.join(path, file_name)
        

    
        # Check if the file exists before reading it
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # Extract publication date using regex
                pub_date_match = re.search(pub_date_pattern, content)
                pub_date = pub_date_match.group(1).strip() if pub_date_match else "Publication date not found"

                # Add the citation (ID and publication date) to the list
                citation_data.append(f"ID: {file_name} Published Date: {pub_date}")
        else:
            citation_data.append(f"{file_name} not found.")
    
    print("==== Returning relevant chunks ====")
    return relevant_chunks, results, citation_data




def generate_response(question, relevant_chunks, citation_data):
    context = "\n\n".join(relevant_chunks)
    
    # Convert citation_data list to a formatted string
    citation_str = "\n".join(citation_data)

    # Construct the prompt correctly
    prompt = (
        "Du bist ein Assistent für Frage-Antwort-Aufgaben. Verwende die folgenden abgerufenen Kontextstücke, um die Frage zu beantworten. "
        "Wenn du die Antwort nicht weißt, sage, dass du es nicht weißt. "
        "Verwende maximal drei Sätze und halte die Antwort kurz. "
        "Falls die Frage des Nutzers nicht die Tectake Arena, oder s.Oliver Arena oder Carl-Diem-Halle, betrifft, bitte sie, bei relevanten Themen zu bleiben, anstatt die Frage zu beantworten.\n\n"
        "Falls sie relevant ist, gib am Ende deiner Antwort die folgende Quellenangabe aus:\n\n"
        f"{citation_str}\n\n"
        "Context:\n" + context + "\n\n"
        "Question:\n" + question
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
            returned = query_documents(user_question)
            relevant_chunks = returned[0]
            citation_data = returned[2]  # Get the citation data

            # Generate the response with citations
            response = generate_response(user_question, relevant_chunks, citation_data)
            
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
    #st.sidebar.info("Gesamtzahl der indizierten Nachrichtenartikel: 528")
    st.sidebar.info(f"Gesamtzahl der indizierten chunks: {collection.count()}")

if __name__ == "__main__":
    main()