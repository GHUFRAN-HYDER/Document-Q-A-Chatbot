import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Document Q&A Chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    response: str

# Global variables
embeddings = None
vectorstore = None
retriever = None
rag_chain = None
llm = None

def initialize_base_components():
    """Initialize base components that don't require documents"""
    global embeddings, llm
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def initialize_document_components(docs):
    """Initialize components that require documents and create RAG chain"""
    global vectorstore, retriever, rag_chain

        # Initialize vector store
    if os.path.exists("data/vectorstore"):
        vectorstore = FAISS.load_local("data/vectorstore", embeddings,
            allow_dangerous_deserialization=True  # Only use if you trust the source)
        )
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Initialize chains
    system_prompt = """
    Answer questions based on the provided context. If the information isn't 
    available in the context, clearly state that you don't have enough information.
    Always provide concise and accurate responses.Formatting of the response should depend on the response type.
    
    Context:
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def process_pdf(file_path: str):
    """Process PDF file, update vector store and return number of chunks"""
    try:
        # Load and split PDF
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
        )
        docs = text_splitter.split_documents(raw_docs)
        
        # Initialize or update vector store
        global vectorstore
        if vectorstore is None:
            # First document, initialize all components
            initialize_document_components(docs)
        else:
            # Add documents to existing vector store
            vectorstore.add_documents(docs)
                # Save updated vector store
        os.makedirs("data/vectorstore", exist_ok=True)
        vectorstore.save_local("data/vectorstore")

        return len(docs)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize base components on startup"""
    initialize_base_components()

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a PDF file
    
    Args:
        file: PDF file to upload
        
    Returns:
        dict: Message indicating number of chunks processed
        
    Raises:
        HTTPException: If file is not PDF or processing fails
    """

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        os.makedirs("temp", exist_ok=True)
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF
        num_chunks = process_pdf(file_path)
        
        # Clean up/ remove the temporary file saved
        os.remove(file_path)
        
        return {"message": f"Successfully processed {num_chunks} chunks from {file.filename}"}
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat interactions"""
    if vectorstore is None:
        return ChatResponse(
            response="Please upload a document first before asking questions."
        )
    
    try:
        # Convert chat history to the format expected by the chain
        formatted_history = [
            (msg.role, msg.content) for msg in request.chat_history
        ]
        
        # Get response from chain
        response = rag_chain.invoke({
            "input": request.message,
            "chat_history": formatted_history
        })
        
        return ChatResponse(
            response=response["answer"]
        )
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)