from flask import Blueprint, request, jsonify
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable LangSmith Tracing (optional, helpful for debugging)
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Initialize the Blueprint
chatbot_bp = Blueprint("chatbot", __name__)

# Load the Ollama LLM (LLaMA 3.2 or any other local model)
llm = Ollama(model="llama3.2")

# Memory to store last 5 interactions
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# Custom Prompt Template for educational chatbot
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful and friendly AI tutor named EduBot. "
     "You assist students with concept explanations, study help, and doubts related to quizzes or lessons. "
     "Be clear, simple, and use examples if needed. "
     "Do not give direct answers to active quizzes unless asked to explain the concept behind the question."),
    ("system", "{history}"),  # 🧠 Include memory's history here
    ("user", "{question}")    # 👤 User's new question
])

# Create the conversation chain
chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,              
    input_key="question"        
)

# Chatbot API Endpoint
@chatbot_bp.route("/api/chatbot", methods=["POST"])
def ai_chatbot():
    try:
        data = request.get_json(force=True, silent=True)

        # Input validation
        if not data or "query" not in data:
            return jsonify({"error": "Invalid JSON input. Expected {'query': 'your message'}"}), 400

        query = data["query"].strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        # Run conversation chain
        response = chain.run(question=query)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
