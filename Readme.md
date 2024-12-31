# Document Q&A Chatbot

A FastAPI-based chatbot that allows users to upload PDF documents and ask questions about their content using RAG (Retrieval-Augmented Generation),Embeeddings, vector daabase and LLM.

## Features

- PDF document upload and processing
- Interactive chat interface
- Real-time question answering based on document content
- Responsive web UI 
- Vector store persistence for document embeddings
- Chat history management

## Technology Stack

- **Backend**: FastAPI, Python
- **AI/ML**: LangChain, OpenAI gpt-4o-mini, HuggingFace Embeddings
- **Vector Store**: FAISS
- **Frontend**: HTML, JavaScript, TailwindCSS
- **Deployment**: Docker

## Setup Instructions

### Prerequisites

- Docker installed on your system
- OpenAI API key

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Document-Q-A-Chatbot.git
   cd Document-Q-A-Chatbot
   ```

2. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   uvicorn chatbot-app:app --reload
   ```

  OR
  ```bash
   python chatbot-app.py
   ```

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t document-qa-chatbot .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 --env-file .env document-qa-chatbot
   ```

Access the application at `http://localhost:8000`

## API Documentation

### Endpoints

#### GET /
- Returns the main HTML interface
- Response: HTML file

#### POST /upload
- Uploads and processes a PDF file
- Request: Multipart form data with PDF file
- Response:
  ```json
  {
    "message": "Successfully processed X chunks from filename.pdf"
  }
  ```

#### POST /chat
- Handles chat interactions with the document
- Request body:
  ```json
  {
    "message": "string",
    "chat_history": [
      {
        "role": "string",
        "content": "string"
      }
    ]
  }
  ```
- Response:
  ```json
  {
    "response": "string"
  }
  ```

## Project Structure

```
document-qa-chatbot/
├── chatbot-app.py       # Main FastAPI application
├── static/             
│   └── index.html      # Frontend interface
├── data/
│   └── vectorstore/    # Persistent vector storage
├── temp/               # Temporary file storage
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Development Notes

- The application uses FAISS for efficient similarity search of document chunks
- Vector store is persisted between runs for better performance so that embeddings not to be generated on each interaction.
- Frontend includes message and responsive design
- Error handling and logging are implemented throughout the application


