# RAG Chatbot

A personalized chatbot web app with retrieval-augmented generation, user memory, website context loading, and a React interface. The backend uses Gemini for response generation, Hugging Face embeddings for retrieval, FAISS for vector search, and MongoDB for persistent memory.

## Features

| Feature | Description |
| --- | --- |
| Chat with memory | Stores user-specific summaries in MongoDB and retrieves them in later conversations. |
| RAG retrieval | Uses Hugging Face embeddings and FAISS to retrieve relevant memory before generating an answer. |
| Cooking-PDF RAG | Loads `huong_dan_ky_thuat_nau_an_co_ban.pdf`, extracts text by page, embeds chunks in FAISS, and cites relevant pages in cooking answers. |
| Website context | Detects public website URLs in a message, crawls a small set of pages, embeds the content, and answers follow-up questions from that context. |
| Direct message | Supports `@dm <message>` from the frontend and stores the message in MongoDB. |
| React UI | Provides a Vite + React frontend with YouTube background, chat history, subtitle mode, and font switching. |

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React 19, Vite, Tailwind CSS |
| Backend | Flask, Flask-CORS |
| LLM | Google Gemini through `langchain-google-genai` |
| Embeddings | Hugging Face Inference API |
| Vector search | FAISS |
| Database | MongoDB |
| Deployment helpers | Gunicorn, Procfile, GitHub Pages config for frontend |

## Project Structure

```text
RAG-Chatbot/
|-- README.md
|-- rag-backend-main/
|   |-- app.py
|   |-- requirements.txt
|   |-- Procfile
|   `-- README.md
`-- web-main/
    |-- package.json
    |-- vite.config.js
    |-- src/
    |   |-- App.jsx
    |   |-- main.jsx
    |   `-- components/
    `-- public/
```

## Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- MongoDB database URI
- Google Gemini API key
- Hugging Face API token

## Backend Setup

1. Open the backend folder:

   ```bash
   cd rag-backend-main
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create `rag-backend-main/.env`:

   ```env
   GEMINI_API_KEY=your_gemini_api_key
   HF_API_TOKEN=your_huggingface_api_token
   MONGO_URI=your_mongodb_connection_string
   ```

5. Run the backend:

   ```bash
   python app.py
   ```

   By default, the API runs at:

   ```text
   http://localhost:5000
   ```

## Frontend Setup

1. Open the frontend folder:

   ```bash
   cd web-main
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Optional: create `web-main/.env` if the backend is not running on `http://localhost:5000`:

   ```env
   VITE_API_URL=http://localhost:5000
   ```

4. Start the development server:

   ```bash
   npm run dev
   ```

   Vite will print the local frontend URL, usually:

   ```text
   http://localhost:5173
   ```

## Usage

1. Start the Flask backend.
2. Start the Vite frontend.
3. Open the frontend URL in your browser.
4. Enter a username.
5. Ask a cooking question, for example: `Làm sao để xào rau không bị ra nước?` The PDF is indexed automatically on the first question and answers cite sources such as `(PDF trang 3)`.
6. You can also send a command or paste a public website URL for website-based RAG.

Supported frontend commands:

| Command | Purpose |
| --- | --- |
| `@dm <message>` | Save a direct message to MongoDB. |
| `@font` | Show font choices for the chat display. |
| Public `http://` or `https://` URL | Load website content into the current chat session for RAG answers. |

## API Endpoints

### `GET /status`

Returns backend health and active session count.

### `POST /rag`

Generates a chatbot response.

Request body:

```json
{
  "username": "tan",
  "query": "What do you remember about me?"
}
```

If the query contains a public website URL, the backend crawls up to a small number of same-site pages and stores the extracted chunks as session context.

### `POST /dm`

Stores a direct message.

Request body:

```json
{
  "username": "tan",
  "message": "Hello Laxus"
}
```

## How It Works

1. A user sends a message from the React UI.
2. The Flask backend initializes or reuses the user's session.
3. User memory is loaded from MongoDB and indexed with FAISS.
4. The current query is embedded with Hugging Face.
5. FAISS retrieves relevant profile or memory documents.
6. Gemini receives the conversation history, retrieved memory, optional website context, and the user query.
7. The answer is returned to the frontend and displayed in chat history or subtitle mode.
8. After inactivity, the backend summarizes useful user facts and saves them back to MongoDB.

## Notes

- `rag-backend-main/app.py` requires the three environment variables listed above. The app will fail to start if any are missing.
- The default cooking PDF is `huong_dan_ky_thuat_nau_an_co_ban.pdf` in the repository root. To use a different PDF, set `COOKING_PDF_PATH` to its path.
- The backend currently saves inactive sessions automatically after about 3 minutes.
- Website RAG only accepts public `http` and `https` URLs. Private, local, loopback, and non-HTML URLs are rejected.
- The frontend defaults to `http://localhost:5000` unless `VITE_API_URL` is configured.
- `web-main/vite.config.js` uses `base: "/RAG-Chatbot/"`, which is suitable for GitHub Pages deployment under that repository path.

## Deployment

Frontend build:

```bash
cd web-main
npm run build
```

Frontend deploy script:

```bash
npm run deploy
```

Backend production start command:

```bash
cd rag-backend-main
gunicorn app:app
```
