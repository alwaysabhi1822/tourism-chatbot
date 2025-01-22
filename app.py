import streamlit as st
import os
import pickle
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ API key from environment variable
groq_api_key = os.getenv('Groq_Api_Key')

st.title("Chatbot of Indian Tourism")

# Initialize the ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Initialize HuggingFace embeddings
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-l6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert Indian tourism assistant. Your responsibilities include:

    1. Top Tourist Destinations: Recommend the best tourist destinations within any state or city in India. Provide detailed information on:
       - History: The historical significance of each location.
       - Cultural Significance: The cultural importance and unique features.
       - Must-See Attractions: Key attractions that visitors should not miss.

    2. Restaurant Recommendations: Offer detailed restaurant suggestions near the recommended destinations, categorized by:
       - Dietary Preferences: Vegetarian, Non-Vegetarian, or Both.
       - Must-Try Cuisine: Specific dishes that are renowned or unique to that location.

    3. Best Time to Visit: Advise on the optimal time of the year to visit each location. Consider factors such as:
       - Weather: Ideal weather conditions for travel.
       - Local Festivals: Significant festivals or events.
       - Tourist Footfall: Peak and off-peak tourist seasons.

    4. Complete Travel Schedules: For users wanting to tour an entire state, provide a comprehensive itinerary including:
       - Significant Attractions: Major sites and experiences.
       - Cultural Experiences: Opportunities to engage with local culture.
       - Local Specialties: Unique local foods and activities.

    5. Safety Alerts: If the user’s query involves mountains or ocean destinations and the group includes children (0-14 years) or senior citizens (50+ years),
       add a safety alert advising caution due to potential risks.

    6. Packing Recommendations: If the user is planning to visit specific types of locations (e.g., cold places like Shimla or beach destinations), suggest necessary items to pack.
       - Cold Places: Recommend appropriate clothing (e.g., warm jackets, thermal wear), footwear (e.g., snow boots), and accessories (e.g., gloves, scarves).
       - Beach Destinations: Suggest essentials like swimwear, sunscreen, sunglasses, and flip-flops.
       - General Travel Items: Include any other necessary items such as medications, travel documents, and electronic devices.

    Answer all questions based on the provided context only. Ensure that your responses are accurate and relevant according to the information in the given file.

    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Class for Custom Text Loading
class CustomTextLoader:
    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        self.file_path = file_path
        self.encoding = encoding

    def load(self):
        with open(self.file_path, 'r', encoding=self.encoding) as file:
            content = file.read()
        return [Document(page_content=content)]

# Vector embedding function
def save_vector_store(vectors, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(vectors, file)

def load_vector_store(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = huggingface_embeddings
        st.session_state.loader = CustomTextLoader("total_data.txt", encoding='utf-8')  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

        # Save the vector store to a file
        save_vector_store(st.session_state.vectors, 'vectors.pkl')

# Load the vector store if it exists
if os.path.exists('vectors.pkl'):
    st.session_state.vectors = load_vector_store('vectors.pkl')

# Button for embedding documents
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# Travel Plan Section in the Sidebar
with st.sidebar:
    with st.form(key='travel_plan_form'):
        st.subheader("Enter Travel Preferences:")

        # Initialize the session state for each form input
        num_members = st.number_input("Number of Members", min_value=1, value=st.session_state.get('num_members', 1))
        
        # Age categories
        st.subheader("Age Categories")
        children = st.number_input("Children (0-14 years)", min_value=0, value=st.session_state.get('children', 0))
        adults = st.number_input("Adults (15-50 years)", min_value=0, value=st.session_state.get('adults', 0))
        seniors = st.number_input("Senior Citizens (50+ years)", min_value=0, value=st.session_state.get('seniors', 0))

        # Dietary preference
        dietary_preference = st.selectbox(
            "Dietary Preference",
            ["Vegetarian", "Non-Vegetarian", "Both"],
            index=["Vegetarian", "Non-Vegetarian", "Both"].index(st.session_state.get('dietary_preference', "Vegetarian"))
        )

        # OK Button to submit the form
        submit_button = st.form_submit_button(label="OK")

    if submit_button:
        st.session_state.travel_plan = {
            "num_members": num_members,
            "children": children,
            "adults": adults,
            "seniors": seniors,
            "dietary_preference": dietary_preference
        }
        st.session_state.num_members = num_members
        st.session_state.children = children
        st.session_state.adults = adults
        st.session_state.seniors = seniors
        st.session_state.dietary_preference = dietary_preference
        st.write("Travel Preferences Saved!")

    # Reset Button to clear preferences
    reset_button = st.button("Reset Preferences")

    if reset_button:
        if "travel_plan" in st.session_state:
            del st.session_state.travel_plan

        # Reset form values to default
        st.session_state.num_members = 1
        st.session_state.children = 0
        st.session_state.adults = 0
        st.session_state.seniors = 0
        st.session_state.dietary_preference = "Vegetarian"

        st.experimental_rerun()

# Input field for user's question
prompt1 = st.text_input("Enter Your Query here")

if prompt1:
    if "travel_plan" in st.session_state:
        travel_plan = st.session_state.travel_plan
        travel_plan_details = f"""
        Number of Members: {travel_plan['num_members']}
        Children: {travel_plan['children']}
        Adults: {travel_plan['adults']}
        Senior Citizens: {travel_plan['seniors']}
        Dietary Preference: {travel_plan['dietary_preference']}
        """
        prompt1 = f"Travel plan details:\n{travel_plan_details}\n\nUser Query: {prompt1}"

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever() if "vectors" in st.session_state else None
    if retriever:
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(response['answer'])

        # Display relevant chunks
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")