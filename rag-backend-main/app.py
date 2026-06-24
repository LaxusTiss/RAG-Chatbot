from flask import Flask, request, jsonify, session
from flask_cors import CORS
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from huggingface_hub import InferenceClient
from pymongo import MongoClient
from datetime import datetime, timedelta
import threading
import atexit
import re
from dotenv import load_dotenv
import psutil
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.request import Request, urlopen
import ipaddress
import socket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def print_memory(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"[{stage}] RAM usage: {mem:.2f} MB")

# Lấy biến môi trường
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
HF_API_TOKEN = os.environ["HF_API_TOKEN"]
MONGO_URI = os.environ["MONGO_URI"]

# Khởi tạo Flask app và bật CORS
app = Flask(__name__)
print_memory("App started")
app.secret_key = os.urandom(24)
CORS(app)

# Kết nối MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["Laxus_DB"]
documents_collection = db["documents"]
direct_messages_collection = db["direct_messages"]

# Khởi tạo model và client
genai_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
client = InferenceClient(api_key=HF_API_TOKEN)

# Dữ liệu mẫu
doc_texts = [
"your name, who, called = Laxus TT",  
"age, how old, birth year = 21",  
"where, location, country = Da Lat, Lam Dong, Vietnam",  
"hobby, interests, like to do = sport, code, talk, photograph",
"sport = football, pickleball, badminton ",  
"job, work, profession, study = CS",  
"university, school, education = HCMUIT",  
"favorite food, like to eat = Com tam, Mother's meal",  
"favorite color, color you like = orange ",  
"language, speak, talk = English, Vietnamese",  
"pet, animal, have pet = dont have",  
"music, favorite song, like to listen = Thoi em dung di, Tron tim, 7 years, Bad liar",  
"book, favorite book, like to read = The Story Of A Seagull And The Cat Who Taught Her To Fly Book by Luis Sepúlveda",  
"movie, film, favorite movie, favortie show = High Kick , Running man 7012",  
"anime, cartoon, favorite anime = Fairy Tail, Doraemon, ",  
"goal, dream, future plan = live wholehearted",  
"relationship, girlfriend, love life = error found",  
"programming, coding, language you use = python, C++",  
"ai, machine learning, neural network = learning",  
"exercise, workout, fitness = sport",  
"game, video game, play = LOL, PUBG",  
"idol, favorite singer = Cristiano Ronaldo, myself"

]

# Lưu trữ phiên trò chuyện tạm thời (dùng username làm key)
sessions = {}
session_event = threading.Event()
state_lock = threading.Lock()

# Biến toàn cục cho FAISS
doc_index = None
doc_embeddings = None
doc_texts_current = None

MAX_SITE_PAGES = 5
MAX_SITE_CHUNKS = 24
MAX_PAGE_BYTES = 1_000_000
CHUNK_WORDS = 180
CHUNK_OVERLAP = 40
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)

# Hàm lấy embeddings
def get_embeddings(texts):
    try:
        result = client.feature_extraction(
            texts,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        return np.array(result)
    except Exception as e:
        raise Exception(f"Hugging Face API error: {str(e)}")

class WebPageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self.text_parts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "section", "article"}:
            self.text_parts.append("\n")

    def handle_data(self, data):
        clean = " ".join(data.split())
        if not clean:
            return
        if self._in_title:
            self.title = f"{self.title} {clean}".strip()
        if self._skip_depth == 0:
            self.text_parts.append(clean)

    def text(self):
        return re.sub(r"\n\s*\n+", "\n", " ".join(self.text_parts)).strip()

def is_public_http_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        for result in socket.getaddrinfo(parsed.hostname, None):
            ip = ipaddress.ip_address(result[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except Exception:
        return False
    return True

def normalize_url(url):
    clean_url, _ = urldefrag(url.strip())
    return clean_url.rstrip("/")

def fetch_page(url):
    if not is_public_http_url(url):
        raise Exception("Only public http/https website URLs are supported.")
    req = Request(
        url,
        headers={
            "User-Agent": "RAG-Chatbot/1.0 (+https://laxustiss.github.io/RAG-Chatbot/)"
        },
    )
    with urlopen(req, timeout=10) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            raise Exception(f"Unsupported content type: {content_type or 'unknown'}")
        raw = response.read(MAX_PAGE_BYTES + 1)
    if len(raw) > MAX_PAGE_BYTES:
        raw = raw[:MAX_PAGE_BYTES]
    html = raw.decode("utf-8", errors="ignore")
    parser = WebPageParser()
    parser.feed(html)
    return {
        "url": url,
        "title": parser.title or urlparse(url).netloc,
        "text": parser.text(),
        "links": parser.links,
    }

def same_site_link(base_url, href):
    absolute = normalize_url(urljoin(base_url, href))
    base = urlparse(base_url)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc != base.netloc:
        return None
    return absolute

def crawl_site(start_url, max_pages=MAX_SITE_PAGES):
    start_url = normalize_url(start_url)
    queue = [start_url]
    seen = set()
    pages = []

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            page = fetch_page(url)
        except Exception as exc:
            if not pages:
                raise exc
            continue
        if len(page["text"].split()) >= 40:
            pages.append(page)
        for href in page["links"][:80]:
            next_url = same_site_link(start_url, href)
            if next_url and next_url not in seen and next_url not in queue:
                queue.append(next_url)

    if not pages:
        raise Exception("I could not extract enough readable text from that website.")
    return pages

def chunk_text(text, words_per_chunk=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    step = max(1, words_per_chunk - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + words_per_chunk]).strip()
        if len(chunk.split()) >= 35:
            chunks.append(chunk)
    return chunks

def build_website_context(url):
    pages = crawl_site(url)
    chunks = []
    for page in pages:
        for index, chunk in enumerate(chunk_text(page["text"])):
            chunks.append({
                "text": chunk,
                "source_url": page["url"],
                "title": page["title"],
                "chunk_index": index,
            })
            if len(chunks) >= MAX_SITE_CHUNKS:
                break
        if len(chunks) >= MAX_SITE_CHUNKS:
            break

    embeddings = get_embeddings([chunk["text"] for chunk in chunks]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return {
        "root_url": normalize_url(url),
        "pages": len(pages),
        "chunks": chunks,
        "index": index,
    }

def retrieve_website_docs(username, query, top_k=4):
    with state_lock:
        site_context = sessions.get(username, {}).get("site_context")
    if not site_context:
        return []
    query_embedding = get_embeddings([query]).astype("float32")
    distances, indices = site_context["index"].search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(site_context["chunks"]):
            results.append(site_context["chunks"][idx])
    return results

# Khởi tạo FAISS index
def initialize_index(username=None):
    global doc_index, doc_embeddings, doc_texts_current
    all_texts = doc_texts.copy()
    all_embeddings = []

    if documents_collection.count_documents({}) > 0:
        stored_docs = list(documents_collection.find({}))
        if stored_docs:
            all_embeddings = [np.array(doc["embedding"]) for doc in stored_docs]
            all_texts = [doc["text"] for doc in stored_docs]
    
    if len(all_embeddings) != len(doc_texts):
        documents_collection.drop()
        all_embeddings = []
        for text in doc_texts:
            embedding = get_embeddings([text])[0]
            embedding = embedding.astype("float32")
            documents_collection.insert_one({
                "text": text,
                "embedding": embedding.tolist()
            })
            all_embeddings.append(embedding)

    if username:
        user_collection = db[username]
        user_docs = list(user_collection.find({}))
        for doc in user_docs:
            summary_text = doc.get("summary_sentence")
            embedding = doc.get("embedding")
            if summary_text and embedding:
                all_texts.append(summary_text)
                all_embeddings.append(np.array(embedding, dtype="float32"))

    doc_texts_current = all_texts
    doc_embeddings = np.array(all_embeddings).astype("float32")

    dimension = doc_embeddings.shape[1]
    doc_index = faiss.IndexFlatL2(dimension)
    doc_index.add(doc_embeddings)
    print_memory("After FAISS build")

# Hàm tóm tắt và lưu DB
def summarize_and_store(username):
    global doc_index, doc_embeddings, doc_texts_current
    with state_lock:
        if username not in sessions:
            return
        convo = list(sessions[username]["convo"])
    user_collection = db[username]
    
    conversation_text = "\n".join([
        m['parts'][0]['text'] for m in convo 
        if m['role'] == 'user' and not re.search(r'\b(what|how|where|when|why|who|which)\b|\?', m['parts'][0]['text'], re.IGNORECASE)
    ])

    prompt = f"""
    "Summarize the information about user in a '{username}, attribute : value' format in a single paragraph separated with a semicolon ';' . If it is {username} question, ignore and don't summarize it. Capture separated key facts. Follow the format, do not use asterisks, all in lowercase"
    
    Example: {username},height = 20 ; {username}, age, how old, birth year = 20

    Conversation history:
    {conversation_text}
    """

    summary = genai_model.invoke(prompt).content
    print(f"summary {username}: {summary}")
    summary_sentences = [s.strip() for s in summary.split(";") if s.strip()]
    for sentence in summary_sentences:
        embedding = get_embeddings([sentence])[0]
        embedding = embedding.astype("float32") 
        user_collection.insert_one({
            "summary_sentence": sentence,
            "embedding": embedding.tolist(),
            "timestamp": datetime.now()
        })

    with state_lock:
        sessions.pop(username, None)
        if not sessions:
            doc_index = None
            doc_embeddings = None
            doc_texts_current = None
            session_event.clear()

# Hàm lưu tất cả session khi server dừng
def save_all_sessions():
    for username in list(sessions.keys()):
        summarize_and_store(username)

atexit.register(save_all_sessions)

# Check timeout
def check_timeout():
    while True:
        with state_lock:
            has_sessions = bool(sessions)
            session_items = [
                (username, data["last_active"])
                for username, data in sessions.items()
            ]

        if has_sessions:
            now = datetime.now()
            
            for username, last_active in session_items:
                print(f"{last_active} \n {now}")
                if now - last_active > timedelta(minutes=3):
                    summarize_and_store(username)
            threading.Event().wait(60)
        else:
            session_event.wait()

threading.Thread(target=check_timeout, daemon=True).start()

# Hàm truy xuất tài liệu
def retrieve_docs(query, top_k=1):
    with state_lock:
        current_index = doc_index
        current_texts = doc_texts_current
    if current_index is None:
        return ["No FAISS index available, waiting for a new session!"]
    query_embedding = get_embeddings([query])
    distances, indices = current_index.search(query_embedding, top_k)
    return [current_texts[idx] for idx in indices[0]]

# Hàm sinh câu trả lời
def generate_response(username, query):
    with state_lock:
        if username not in sessions:
            return "Session expired, please enter your name again!"
        convo = sessions[username]["convo"]
    retrieved_docs = retrieve_docs(query, top_k=3)
    profile_context = "\n".join(retrieved_docs)
    website_docs = retrieve_website_docs(username, query, top_k=4)
    website_context = "\n".join([
        f"[Source {index + 1}] {doc['title']} - {doc['source_url']}\n{doc['text']}"
        for index, doc in enumerate(website_docs)
    ])
    
    convo.append({"role": "user", "parts": [{"text": query}]})
    
    prompt = f"Below is the conversation history between the {username} and Laxus TT (you):\n"
    for message in convo:
        role = message["role"]
        text = message["parts"][0]["text"]
        if role == "user":
            prompt += f"{username}: {text}\n"
        elif role == "assistant":
            prompt += f"Laxus TT: {text}\n"
        elif role == "system":
            prompt += f"system: {text}\n"
    prompt += (
        f"Profile memory about Laxus TT and {username}. Ignore if irrelevant: {profile_context}\n"
        f"Website context selected by the user. If you use it, mention the source URL naturally in your answer:\n{website_context or 'No website context loaded.'}\n"
        f"Now, respond to the {username}'s question: {query}"
    )

    response = genai_model.invoke(prompt).content
    with state_lock:
        if username in sessions:
            convo.append({"role": "assistant", "parts": [{"text": response}]})
            sessions[username]["last_active"] = datetime.now()
    return response

# API endpoint POST /rag
@app.route('/rag', methods=['POST'])
def rag_endpoint():
    global doc_index
    data = request.get_json(silent=True) or {}
    username = data.get('username', '')
    query = data.get('query', '')
    if not username or not query:
        return jsonify({"error": "Username và query là bắt buộc!"}), 400
    
    with state_lock:
        is_new_session = username not in sessions

    if is_new_session:
        initialize_index(username)
        with state_lock:
            sessions[username] = {
                "convo": [{"role": "system", "parts": [{"text": "You're playing as Laxus TT, a gentle boy, a chaotic friend, narcissistic, playful and humourous. Keep responses humanlike, short and on point! Do not use asterisks! Do not list up, only talk about one thing at a time. Do not answer summarization requests. Capitalize to emphasize! Answer in the language that users are using!"}]}],
                "last_active": datetime.now(),
                "username": username
            }
            session_event.set()

    try:
        urls = URL_PATTERN.findall(query)
        if urls:
            site_url = urls[0].rstrip(".,)")
            site_context = build_website_context(site_url)
            with state_lock:
                if username in sessions:
                    sessions[username]["site_context"] = site_context
                    sessions[username]["last_active"] = datetime.now()
                    sessions[username]["convo"].append({"role": "user", "parts": [{"text": query}]})
                    reply = (
                        f"I scanned {site_context['pages']} page(s) from {site_context['root_url']} "
                        f"and saved {len(site_context['chunks'])} chunks as website context. "
                        "Ask me about that site now, and I will answer with source links."
                    )
                    sessions[username]["convo"].append({"role": "assistant", "parts": [{"text": reply}]})
            return jsonify({
                "query": query,
                "response": reply,
                "site": {
                    "root_url": site_context["root_url"],
                    "pages": site_context["pages"],
                    "chunks": len(site_context["chunks"]),
                },
                "session_id": username
            })

        response = generate_response(username, query)
        return jsonify({
            "query": query,
            "response": response,
            "retrieved_docs": retrieve_docs(query, top_k=1),
            "website_sources": [
                {"title": doc["title"], "url": doc["source_url"]}
                for doc in retrieve_website_docs(username, query, top_k=2)
            ],
            "session_id": username
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint POST /dm
@app.route('/dm', methods=['POST'])
def dm_endpoint():
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    message = data.get('message', '').strip()

    if not username or not message:
        return jsonify({"error": "Username and message are required!"}), 400

    if len(message) > 2000:
        return jsonify({"error": "Direct message is too long. Keep it under 2000 characters."}), 400

    direct_messages_collection.insert_one({
        "username": username,
        "message": message,
        "created_at": datetime.now(),
        "status": "unread",
        "source": "web"
    })

    return jsonify({
        "ok": True,
        "response": "Direct message sent to Laxus. I will keep it safe in the inbox."
    })

# API endpoint GET /status
@app.route('/status', methods=['GET'])
def status_endpoint():
    return jsonify({
        "status": "Server is running",
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "active_sessions": len(sessions)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    
