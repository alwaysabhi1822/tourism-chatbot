# Indian Tourism Chatbot



Explore India's rich culture, history, and travel destinations with ease.

#Overview

The Indian Tourism Chatbot is a Streamlit-based AI-powered application designed to assist users in exploring Indian tourist destinations and planning trips. It provides detailed recommendations on places to visit, local cuisines, and optimal travel times, and also generates personalized travel plans based on user preferences.

#Features
Destination Insights:

Discover historical and cultural details about tourist spots in India.
Get tailored recommendations based on user preferences.
Customized Travel Plans:

Input group size, age categories, and dietary preferences.
Receive safety alerts for children and senior citizens traveling to risky destinations.
Restaurant Suggestions:

Filter options based on vegetarian, non-vegetarian, or mixed preferences.
Interactive UI:

Embed documents for efficient data retrieval.
Reset preferences to start fresh.
Packing Advice:

Recommendations for travel essentials based on destination type (e.g., beach or mountain).

Tech Stack

Programming Language: Python

Libraries:
Streamlit for the web interface.
LangChain for conversational AI.
Hugging Face for embedding generation.
FAISS for vector storage and retrieval.


Steps to Set Up and Use the Application
Clone or Download the Repository:

Download or clone the repository to your local machine.
Install Dependencies:

Ensure you have Python 3.8+ installed on your machine.
Navigate to the project directory in your terminal.
Install the required dependencies by running:
bash
Copy
Edit
pip install -r requirements.txt
Set Up Environment Variables:

Create a .env file in the project directory if it doesn’t already exist.
Add your GROQ API key to the .env file in the following format:
makefile
Copy
Edit
Groq_Api_Key=your_api_key_here
Prepare the Data:

Place the total_data.txt file in the root directory of the project. This file should contain the data you want the chatbot to use for answering queries.
Generate the vectors.pkl File:

Start the Streamlit application by running:

bash
Copy
Edit
streamlit run <filename>.py
(Replace <filename>.py with the name of your Python file, e.g., main.py.)

In the Streamlit interface:

Click on the "Documents Embedding" button.
This will process the data in total_data.txt, create document embeddings, and save them into a vectors.pkl file in the root directory.
Ask Your Query:

After generating the vectors.pkl file, you can input your queries in the "Enter Your Query here" field.
If applicable, set travel preferences in the sidebar and click OK to save them. Preferences will be included in the context for generating responses.
Reset Preferences (Optional):

Use the "Reset Preferences" button in the sidebar to clear the saved preferences.
