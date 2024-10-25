from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pydantic models
class Question(BaseModel):
    question: str

class Conversation(BaseModel):
    conversation_id: str
    messages: List[Dict[str, str]]

# Store conversations in memory (consider using a database for production)
conversations: Dict[str, List[Dict[str, str]]] = {}

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0.7,
    model_name="mixtral-8x7b-32768",
    api_key=os.getenv("GROQ_API_KEY")
)
output_parser = StrOutputParser()

def create_prompt_with_history(conversation_history: List[Dict[str, str]], new_question: str) -> ChatPromptTemplate:
    messages = [
        ("system", "You are a helpful assistant. Please respond to the user queries while maintaining context of the conversation.")
    ]
    
    # Add conversation history to the prompt
    for message in conversation_history:
        if message["role"] == "user":
            messages.append(("user", message["content"]))
        else:
            messages.append(("assistant", message["content"]))
    
    # Add the new question
    messages.append(("user", new_question))
    
    return ChatPromptTemplate.from_messages(messages)

def update_conversation_history(conversation_id: str, user_message: str, assistant_response: str):
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add new messages
    conversations[conversation_id].extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ])
    
    # Keep only the last 3 message pairs (6 messages total)
    if len(conversations[conversation_id]) > 6:
        conversations[conversation_id] = conversations[conversation_id][-6:]

@app.get("/")
async def read_root():
    return {"message": "Welcome to LangChain API with Groq"}

@app.post("/ask/{conversation_id}")
async def get_answer(conversation_id: str, question: Question):
    try:
        # Get conversation history
        conversation_history = conversations.get(conversation_id, [])
        
        # Create prompt with history
        prompt = create_prompt_with_history(conversation_history, question.question)
        
        # Create chain with the new prompt
        chain = prompt | llm | output_parser
        
        # Get response
        response = chain.invoke({})
        
        # Update conversation history
        update_conversation_history(conversation_id, question.question, response)
        
        return {
            "answer": response,
            "conversation_history": conversations[conversation_id]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]
    }

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": f"Conversation {conversation_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Conversation not found")
