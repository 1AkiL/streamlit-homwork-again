"""
Trustworthy AI Explainer - Streamlit Dashboard
Module 15 Team Project Template

This template provides a working Streamlit application with:
- Multi-page dashboard
- LLM chatbot with memory
- Explainability integration
- User feedback system
- State management and caching
- Performance monitoring

Team: Customize this template for your specific use case
"""

import streamlit as st
import os
import time
import sqlite3
from contextlib import closing
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import httpx
import hashlib
import hmac
from io import BytesIO
import json
import re
import unicodedata
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationSummaryBufferMemory
from pydantic import Field

# ============================================================================
# PAGE CONFIGURATION (Must be first Streamlit command)
# ============================================================================

st.set_page_config(
    page_title="Trustworthy AI Explainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MOCK LLM (Replace with actual LLM)
# ============================================================================

#class MockLLM:
#    """Mock LLM for testing - replace with actual implementation"""
#    def generate(self, prompt: str, history: List = None, temperature: float = 0.7) -> str:
#        time.sleep(0.5)  # Simulate API latency
#        return f"Response to: '{prompt[:50]}...'\n\nThis is a mock response. Integrate your actual LLM here."
#from dotenv import load_dotenv
#load_dotenv("API_KEY.env")  # local only; on Streamlit Cloud the key comes from st.secrets
# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

# Model deployed on the Inference Endpoint (shown in the sidebar; the Default engine
# doesn't use it in requests). Change it to your Qwen3-8B fine-tune's repo if you use one.
HF_MODEL = "OLizardi/qwen3-8b-ccna-qlora"

#========================================
# RAG functions
#========================================
BASE_DIR = Path(__file__).parent
VECTORSTORE_DIR = BASE_DIR / "course_vectorstore"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "ccna_" + EMBEDDING_MODEL.split("/")[-1].replace("-", "_").lower()
DOCUMENTS = [
    {
        'id':'ccna1_mod1',
        'course':'CCNA1',
        'title':'Módulo 1 - Las redes en la actualidad',
        'file_path':r'documents/CCNA 1 Versión 7_ Módulo 1 - Las redes en la actualidad.pdf',
        'text':'',
    },
    {
        'id':'ccna1_mod2',
        'course':'CCNA1',
        'title':'Módulo 2 - Configuración básica de switches y terminales',
        'file_path':r'documents/CCNA 1 Versión 7_ Módulo 2 - Configuración básica de switches y terminales.pdf',
        'text':'',
    },
    {
        'id':'ccna1_mod3',
        'course':'CCNA1',
        'title':'Módulo 3 - Protocolos y modelos',
        'file_path':r'documents/CCNA 1 Versión 7_ Módulo 3 - Protocolos y modelos.pdf',
        'text':'',
    },
    {
        'id':'ccna1_mod4',
        'course':'CCNA1',
        'title':'Módulo 4 - Capa física',
        'file_path':r'documents/CCNA 1 Versión 7_ Módulo 4 - Capa física.pdf',
        'text':'',
    },
    {
        'id':'ccna2_mod2',
        'course':'CCNA2',
        'title':'Módulo 2 - Conceptos de switching',
        'file_path':r'documents/CCNA 2 Versión 7_ Módulo 2 - Conceptos de switching.pdf',
        'text':'',
    },
    {
        'id':'ccna2_mod3',
        'course':'CCNA2',
        'title':'Módulo 3 - VLANs',
        'file_path':r'documents/CCNA 2 Versión 7_ Módulo 3 - VLANs.pdf',
        'text':'',
    },
    {
        'id':'ccna2_mod7',
        'course':'CCNA2',
        'title':'Módulo 7 - DHCPv4',
        'file_path':r'documents/CCNA 2 Versión 7_ Módulo 7 - DHCPv4.pdf',
        'text':'',
    },
    {
        'id':'ccna3_mod4',
        'course':'CCNA3',
        'title':'Módulo 4 - Conceptos de ACL',
        'file_path':r'documents/CCNA 3 Versión 7_ Módulo 4 - Conceptos de ACL.pdf',
        'text':'',
    },
    {
        'id':'ccna3_mod5',
        'course':'CCNA3',
        'title':'Módulo 5 - Configuración de ACL para IPv4',
        'file_path':r'documents/CCNA 3 Versión 7_ Módulo 5 - Configuración de ACL para IPv4.pdf',
        'text':'',
    },
]


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Skipping unreadable file {pdf_path}: {e}")
        return ""


def preprocess_course_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)  # control chars except \t \n \r
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse big gaps
    return text.strip()


def make_splitter() -> RecursiveCharacterTextSplitter:
# Same chunking for the course PDFs and for user-added material.
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def build_course_chunks() -> List[Document]:
# Read every PDF and split it into chunks with metadata.
    splitter = make_splitter()
    docs = []
    for doc in DOCUMENTS:
        text = preprocess_course_text(extract_text_from_pdf(BASE_DIR / doc['file_path']))
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(Document(
                page_content=chunk,
                metadata={'source_id': doc['id'], 'course': doc['course'],
                          'title': doc['title'], 'chunk_index': i},
            ))
    return docs


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Indexing course documents (first run only)...")
def load_vectorstore() -> Chroma:

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=load_embedding_model(),
        persist_directory=str(VECTORSTORE_DIR),
    )
    if store.get(limit=1)["ids"]:      # already indexed -> reuse
        return store

    docs = build_course_chunks()
    if not docs:
        raise RuntimeError("No text could be extracted from the PDFs in documents/.")
    ids = [f"{d.metadata['source_id']}-{d.metadata['chunk_index']}" for d in docs]
    store.add_documents(docs, ids=ids)
    return store


# Minimum cosine similarity for a chunk to be given to the model.
# Measured on this index: on-topic questions score 0.60-0.86 for the best chunk,
RELEVANCE_CUTOFF_DEFAULT = 0.45


def retrieve_context(question: str, k: int = 3,
                     min_similarity: float = RELEVANCE_CUTOFF_DEFAULT,
                     previous_question: Optional[str] = None) -> List[tuple]:
    store = load_vectorstore()
    queries = [(question, False)]
    if previous_question and previous_question.strip():
        queries.append((f"{previous_question.strip()} {question}", True))

    best = {}  # (source_id, chunk_index) -> (doc, similarity, via_history)
    for query, via_history in queries:
        for doc, distance in store.similarity_search_with_score(query, k=k):
            sim = 1.0 - distance / 2.0
            key = (doc.metadata.get('source_id'), doc.metadata.get('chunk_index'))
            if key not in best or sim > best[key][1]:
                best[key] = (doc, sim, via_history)

    ranked = sorted(best.values(), key=lambda item: item[1], reverse=True)
    return [item for item in ranked if item[1] >= min_similarity][:k]


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['title']}]\n{d.page_content}" for d in docs)


#========================================
# User-added material (Knowledge Base page)
#========================================
USER_COURSE = "Usuario"          # 'course' label for user-added material


def extract_text_from_upload(uploaded_file) -> str:
# Text from an uploaded PDF, TXT or Markdown file.
    data = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


def add_user_document(title: str, text: str) -> int:
# Chunk, embed and store user text in the shared index; returns the number of excerpts.
# The ID comes from a hash of the text, so adding the same content twice overwrites it.
    text = preprocess_course_text(text)
    if not text:
        return 0
    source_id = "user_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    chunks = make_splitter().split_text(text)
    docs = [Document(page_content=chunk,
                     metadata={'source_id': source_id, 'course': USER_COURSE,
                               'title': title, 'chunk_index': i})
            for i, chunk in enumerate(chunks)]
    load_vectorstore().add_documents(docs, ids=[f"{source_id}-{i}" for i in range(len(docs))])
    return len(docs)


def list_user_documents() -> List[Dict]:
# One entry per user-added document, with its excerpt count.
    data = load_vectorstore().get(where={"course": USER_COURSE}, include=["metadatas"])
    docs = {}
    for m in data["metadatas"]:
        entry = docs.setdefault(m["source_id"], {"source_id": m["source_id"],
                                                 "title": m["title"], "chunks": 0})
        entry["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["title"].lower())


def delete_user_document(source_id: str):
    load_vectorstore().delete(where={"source_id": source_id})


def check_admin_password(password: str) -> bool:
# Checks against ADMIN_PASSWORD from secrets
    expected = _get_setting("ADMIN_PASSWORD")
    if not expected or not password:
        return False
    return hmac.compare_digest(password.encode("utf-8"), str(expected).encode("utf-8"))



#================================
# LLM function
#================================
@st.cache_resource
def load_llm(hf_token: str):
    """
    Load the HTTP client for the endpoint (cached as resource, one client per token).
    Use @st.cache_resource for expensive objects like models and connections.
    """
    return httpx.Client(
        headers={"Authorization": f"Bearer {hf_token}"},
        timeout=httpx.Timeout(120.0, connect=10.0),  # generation on the GPU can take a while
    )

    #return MockLLM()


# Qwen3 thinks out loud (<think>...</think>) by default. An empty think block at the start
# of the assistant turn is what Qwen's chat template uses for enable_thinking=False.
QWEN_NO_THINK = "<think>\n\n</think>\n\n"


def build_qwen_prompt(system_prompt: str, messages: List[Dict],
                      context: str = "") -> str:
    # Format the chat in Qwen3's ChatML format:
    # <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    # Qwen has a real system role, so the system prompt (and memory summary) go there.
    # Retrieved course context is added only to the newest question.
    parts = []
    if system_prompt.strip():
        parts.append(f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n")
    last_user_idx = max((i for i, m in enumerate(messages) if m['role'] == 'user'), default=-1)
    for i, m in enumerate(messages):
        role = 'user' if m['role'] == 'user' else 'assistant'
        content = m['content']
        if i == last_user_idx and context:
            content = (f"Usa el siguiente material del curso para responder.\n\n"
                       f"Contexto:\n{context}\n\nPregunta: {content}")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n" + QWEN_NO_THINK)
    return "".join(parts)


def clean_model_output(text: str) -> str:
    # Remove any reasoning Qwen still produced and cut at the end-of-turn token,
    # in case the endpoint didn't stop there itself.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:                      # reasoning without an opening tag
        text = text.split("</think>", 1)[1]
    for stop in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.split(stop, 1)[0]
    return text.strip()


def query_endpoint(client: httpx.Client, endpoint_url: str, prompt: str,
                   temperature: float, max_new_tokens: int) -> str:
    url = endpoint_url.rstrip("/")
    if url.endswith("/v1"):  # a /v1 suffix only applies to vLLM/TGI endpoints
        url = url[:-3]

    parameters = {
        "max_new_tokens": max_new_tokens,
        "return_full_text": False,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        # Qwen3's recommended sampling for non-thinking mode
        parameters.update({"temperature": temperature, "top_p": 0.8, "top_k": 20})

    r = client.post(url, json={"inputs": prompt, "parameters": parameters})
    r.raise_for_status()
    data = r.json()

    item = data[0] if isinstance(data, list) else data
    text = item.get("generated_text", "")
    if text.startswith(prompt):  # in case the engine ignored return_full_text
        text = text[len(prompt):]
    return clean_model_output(text)


#================================
# Conversation memory (summary + recent turns)
#================================
class InferenceEndpointLLM(LLM):
#   LangChain LLM wrapper around the Default-engine endpoint, used by the summary memory.
    hf_token: str = Field(default="", repr=False)   # repr=False keeps the token out of logs
    endpoint_url: str = ""
    max_new_tokens: int = 256

    @property
    def _llm_type(self) -> str:
        return "hf_inference_endpoint_default"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager=None, **kwargs) -> str:
        client = load_llm(self.hf_token)
        qwen_prompt = build_qwen_prompt("", [{'role': 'user', 'content': prompt}])
        # Low temperature instead of greedy: Qwen advises against greedy decoding (repetition)
        return query_endpoint(client, self.endpoint_url, qwen_prompt,
                              temperature=0.3, max_new_tokens=self.max_new_tokens)

    def get_num_tokens(self, text: str) -> int:
        # Rough estimate (~4 characters per token); avoids downloading a tokenizer
        return max(1, len(text) // 4)


SUMMARY_PROMPT_ES = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template=(
        "Resume progresivamente la conversación entre un estudiante y un asistente de CCNA. "
        "Agrega las líneas nuevas al resumen actual y devuelve un resumen nuevo, breve "
        "(máximo 5 oraciones), en español. Conserva los temas de CCNA tratados, las preguntas "
        "del estudiante y los datos clave de las respuestas.\n\n"
        "Resumen actual:\n{summary}\n\n"
        "Líneas nuevas de la conversación:\n{new_lines}\n\n"
        "Resumen nuevo:"
    ),
)

# Recent turns are kept word-for-word up to this many (estimated) tokens;
# older turns are folded into the running summary by the model.
# Qwen3-8B has a 32K-token context, so ~3-4 recent exchanges fit before a summary is made.
MEMORY_TOKEN_LIMIT = 2500


def get_memory() -> ConversationSummaryBufferMemory:
#   The session's summary memory, created on first use.
    llm = InferenceEndpointLLM(hf_token=get_api_key() or "",
                               endpoint_url=get_endpoint_url() or "")
    if st.session_state.get('memory') is None:
        st.session_state.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=MEMORY_TOKEN_LIMIT,
            prompt=SUMMARY_PROMPT_ES,
            human_prefix="Estudiante",
            ai_prefix="Asistente",
        )
    else:
        st.session_state.memory.llm = llm   # the token or endpoint may have changed
    return st.session_state.memory


def memory_messages(memory: ConversationSummaryBufferMemory) -> List[Dict]:
#   Recent turns kept word-for-word, as {'role', 'content'} dicts.
    return [{'role': 'user' if m.type == 'human' else 'assistant', 'content': m.content}
            for m in memory.chat_memory.messages]


def _get_setting(name: str) -> Optional[str]:
#Checks streamlit secrets (Cloud or .streamlit/secrets.toml), then the environment.
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets file when running locally
        pass
    return os.getenv(name)


def get_api_key() -> Optional[str]:
    if st.session_state.get('hf_token'):
        return st.session_state.hf_token
    return _get_setting("HF_TOKEN")


def get_endpoint_url() -> Optional[str]:
    """Inference Endpoint URL, e.g. https://xxxx.us-east-1.aws.endpoints.huggingface.cloud"""
    return _get_setting("HF_ENDPOINT_URL")

DB_PATH = Path(__file__).parent / "feedback.db"


def get_db_connection() -> sqlite3.Connection:
    """Open the feedback database, creating the table on first use."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            message   TEXT,
            response  TEXT,
            rating    TEXT,
            comment   TEXT,
            sources   TEXT
        )
    """)
    # Databases created before sources were tracked: add the column, keep existing rows
    columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    if 'sources' not in columns:
        conn.execute("ALTER TABLE feedback ADD COLUMN sources TEXT")
        conn.commit()
    return conn


def _source_modules(sources_json: Optional[str]) -> str:
    if not sources_json:
        return ""
    try:
        sources = json.loads(sources_json)
    except (TypeError, ValueError):
        return ""
    return "; ".join(dict.fromkeys(f"{s.get('title', '?')} ({s.get('course', '')})"
                                   for s in sources))


@st.cache_data(ttl=3600)
def load_feedback_data() -> pd.DataFrame:
    #Loads feedback from SQLite db.
    with closing(get_db_connection()) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, message, response, rating, comment, sources "
            "FROM feedback ORDER BY timestamp DESC",
            conn,
        )
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['source_modules'] = df['sources'].apply(_source_modules)
    return df

def _make_relevance_fn(response: str):
    try:
        vectorizer = TfidfVectorizer(stop_words="english").fit([response])
    except ValueError:  # response contains only stop words
        vectorizer = TfidfVectorizer().fit([response])
    response_vec = vectorizer.transform([response])

    def predict(texts: List[str]) -> np.ndarray:
        sims = cosine_similarity(vectorizer.transform(texts), response_vec).ravel()
        sims = np.clip(sims, 0.0, 1.0)
        return np.column_stack([1.0 - sims, sims])

    return predict


@st.cache_data(show_spinner="Computing LIME explanation...",max_entries=200)
def compute_explanation(input_text: str, response: str,
                        num_features: int = 5, num_samples: int = 500) -> Dict:
    base = {
        'input_tokens': len(input_text.split()),
        'response_tokens': len(response.split()),
    }

    if not input_text.strip() or not response.strip():
        return {**base, 'confidence': 0.0, 'fidelity': 0.0,
                'top_features': [], 'feature_weights': [],
                'explanation': 'Not enough text to explain.'}

    try:
        predict_fn = _make_relevance_fn(response)
    except ValueError:  # no usable vocabulary in the response
        return {**base, 'confidence': 0.0, 'fidelity': 0.0,
                'top_features': [], 'feature_weights': [],
                'explanation': 'Response has no usable words to explain against.'}

    explainer = LimeTextExplainer(class_names=['unrelated', 'related'], random_state=42)
    exp = explainer.explain_instance(
        input_text,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=(1,),
    )

    weights = exp.as_list(label=1)  # [(word, weight), ...]
    confidence = float(predict_fn([input_text])[0, 1])

    return {
        **base,
        'confidence': confidence,           # relevance score of the full input
        'fidelity': float(exp.score),       # R² of LIME's local surrogate model
        'top_features': [f"{w} ({v:+.3f})" for w, v in weights],
        'feature_weights': weights,
        'explanation': (
            "LIME perturbed the input by removing words and measured how relevance "
            "to the response changed. Positive weights mark words that increase relevance."
        ),
    }

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    
    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Feedback database (in-memory for demo)
    if 'feedback_db' not in st.session_state:
        st.session_state.feedback_db = []
    
    # User preferences
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {
            'temperature': 0.7,
            'max_tokens': 500,
            'relevance_cutoff': RELEVANCE_CUTOFF_DEFAULT,
            'system_prompt': """You are an AI assistant that explains concepts from CCNA
            """
        }
    
    # Current explanation
    if 'current_explanation' not in st.session_state:
        st.session_state.current_explanation = None
    
    # Performance metrics
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'total_messages': 0,
            'avg_response_time': 0,
            'total_feedback': 0
        }

    # Per-message response times (shown on the Monitoring page)
    if 'response_log' not in st.session_state:
        st.session_state.response_log = []

    # Conversation memory (ConversationSummaryBufferMemory), created on first message
    if 'memory' not in st.session_state:
        st.session_state.memory = None
    if 'memory_warning' not in st.session_state:
        st.session_state.memory_warning = None

    # Knowledge Base page: unlocked with ADMIN_PASSWORD for this session only
    if 'kb_unlocked' not in st.session_state:
        st.session_state.kb_unlocked = False
    if 'kb_failed_attempts' not in st.session_state:
        st.session_state.kb_failed_attempts = 0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_response(message: str, temperature: float = 0.7) -> tuple:
    start_time = time.perf_counter()

    # Load cached model
    client = load_llm(get_api_key())

    # Retrieve the most relevant course chunks for this question; chunks below the
    # relevance cutoff are dropped, so off-topic questions get no course context
    # The previous question (if any) is used to resolve follow-ups like "¿Y cómo se configura?".
    # st.session_state.messages ends with the current question, so it is the second-to-last one.
    cutoff = st.session_state.preferences.get('relevance_cutoff', RELEVANCE_CUTOFF_DEFAULT)
    user_questions = [m['content'] for m in st.session_state.messages if m['role'] == 'user']
    previous_question = user_questions[-2] if len(user_questions) >= 2 else None
    scored_sources = retrieve_context(message, min_similarity=cutoff,
                                      previous_question=previous_question)
    sources = [doc for doc, _, _ in scored_sources]
    retrieval_done = time.perf_counter()

    # Build the conversation from memory instead of the full transcript:
    # system prompt + summary of older turns, then the recent turns word-for-word,
    # then the current question (with course context).
    memory = get_memory()
    system_prompt = st.session_state.preferences['system_prompt'].strip()
    if memory.moving_summary_buffer:
        system_prompt += ("\n\nResumen de la conversación anterior:\n"
                          f"{memory.moving_summary_buffer.strip()}")
    prompt_messages = memory_messages(memory) + [{'role': 'user', 'content': message}]
    prompt = build_qwen_prompt(system_prompt, prompt_messages,
                               context=format_context(sources))

    response = query_endpoint(client, get_endpoint_url(), prompt, temperature,
                              st.session_state.preferences['max_tokens'])
    model_done = time.perf_counter()

    # Compute explanation
    explanation = compute_explanation(message, response)
    explanation_done = time.perf_counter()

    # Save the turn to memory. When the recent turns exceed MEMORY_TOKEN_LIMIT,
    # the oldest ones are summarized by the model (one extra endpoint call).
    st.session_state.memory_warning = None
    try:
        memory.save_context({"input": message}, {"output": response})
    except httpx.HTTPError as e:
        st.session_state.memory_warning = (
            f"The conversation summary could not be updated ({type(e).__name__}); "
            "the oldest turns may be missing from memory.")
    end_time = time.perf_counter()

    timings = {
        'retrieval': retrieval_done - start_time,
        'model': model_done - retrieval_done,
        'explanation': explanation_done - model_done,
        'memory': end_time - explanation_done,
        'total': end_time - start_time,
    }
    response_time = timings['total']

    st.session_state.response_log.append({
        'timestamp': datetime.now(),
        'message': message,
        **timings,
    })

    # Update metrics
    st.session_state.metrics['total_messages'] += 1
    st.session_state.metrics['avg_response_time'] = (
        (st.session_state.metrics['avg_response_time'] * 
         (st.session_state.metrics['total_messages'] - 1) + response_time) /
        st.session_state.metrics['total_messages']
    )
    
    return response, explanation, timings, sources_to_dicts(scored_sources)


def sources_to_dicts(scored_docs: List[tuple]) -> List[Dict]:
    """Plain dicts (safe to keep in session_state) for each retrieved (chunk, similarity, via_history)."""
    return [{
        'title': d.metadata.get('title', 'Unknown'),
        'course': d.metadata.get('course', ''),
        'chunk_index': d.metadata.get('chunk_index'),
        'similarity': round(float(sim), 3),
        'via_history': bool(via_history),
        'excerpt': d.page_content,
    } for d, sim, via_history in scored_docs]


def render_sources(sources: List[Dict], with_excerpts: bool = True,
                   cutoff: Optional[float] = None):
    if not sources:
        if cutoff is not None:
            st.caption(f"📚 No course material matched this question "
                       f"(similarity cutoff {cutoff:.2f}), so it was answered without context.")
        return
    titles = list(dict.fromkeys(f"{s['title']} ({s['course']})" for s in sources))
    st.caption("📚 Sources: " + " · ".join(titles))
    if with_excerpts:
        with st.expander(f"View the {len(sources)} excerpts given to the model"):
            for n, s in enumerate(sources, 1):
                similarity = f" · similarity {s['similarity']:.2f}" if 'similarity' in s else ""
                history = " · matched using the previous question" if s.get('via_history') else ""
                st.markdown(f"**{n}. {s['title']}** — {s['course']}, "
                            f"excerpt #{s['chunk_index']}{similarity}{history}")
                excerpt = s['excerpt'] if len(s['excerpt']) <= 600 else s['excerpt'][:600] + "…"
                st.text(excerpt)


def format_timings(timings: Dict) -> str:
    memory = (f" · memory {timings['memory']:.2f} s"
              if timings.get('memory', 0) >= 0.01 else "")   # only shown when a summary was made
    return (f"⏱️ {timings['total']:.2f} s "
            f"(retrieval {timings.get('retrieval', 0):.2f} s · "
            f"model {timings['model']:.2f} s · explanation {timings['explanation']:.2f} s{memory})")

def save_feedback(message: str, response: str, rating: str, comment: str,
                  sources: Optional[List[Dict]] = None):
    feedback_entry = {
        'timestamp': datetime.now(),
        'message': message,
        'response': response,
        'rating': rating,
        'comment': comment,
        'sources': sources or [],
    }

    with closing(get_db_connection()) as conn, conn:  # inner `conn` commits
        conn.execute(
            "INSERT INTO feedback (timestamp, message, response, rating, comment, sources) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (feedback_entry['timestamp'].isoformat(timespec="seconds"),
             message, response, rating, comment,
             json.dumps(feedback_entry['sources'], ensure_ascii=False)),
        )

    # Keep the per-session list for the Monitoring debug panel
    st.session_state.feedback_db.append(feedback_entry)
    st.session_state.metrics['total_feedback'] += 1

    # Clear cache so the dashboard shows the new entry
    load_feedback_data.clear()

# ============================================================================
# PAGE: CHAT INTERFACE
# ============================================================================
#La página era originalmente en inglés, la traducción puede ser algo literal o extraña

def page_chat():
    """Main chat interface page."""
    st.title("🤖 AI Chat with Explainability")
    st.markdown("Haz preguntas y recibe respuestas..")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.preferences['temperature'],
            step=0.1,
            help="Valores más altos hacen las respuestas más aleatorias"
        )
        st.session_state.preferences['temperature'] = temperature
        
        max_tokens = st.slider(
            "Tokens maximos",
            min_value=50,
            max_value=2000,
            value=st.session_state.preferences['max_tokens'],
            step=50
        )
        st.session_state.preferences['max_tokens'] = max_tokens

        relevance_cutoff = st.slider(
            "Recorte de relevancia",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.preferences.get('relevance_cutoff',
                                                         RELEVANCE_CUTOFF_DEFAULT)),
            step=0.05,
            help="Similitud minima (0-1). ",
        )
        st.session_state.preferences['relevance_cutoff'] = relevance_cutoff
        
        with st.expander("Prompt de sistema"):
            system_prompt = st.text_area(
                "System Instructions",
                value=st.session_state.preferences['system_prompt'],
                height=100
            )
            st.session_state.preferences['system_prompt'] = system_prompt
        
        st.divider()
        
        if st.button("🗑️ Limpiar historial de chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.current_explanation = None
            st.session_state.memory = None          # forget the summary too
            st.session_state.memory_warning = None
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Conversación")
        
        # Display chat history
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and "sources" in msg:
                        render_sources(msg["sources"], cutoff=msg.get("cutoff"))
                    if msg.get("timings"):
                        st.caption(format_timings(msg["timings"]))

        # Chat input
        if prompt := st.chat_input("Escribe tu mensaje aquí.",
                                   disabled=not (get_api_key() and get_endpoint_url())):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # Generate response
            try:
                with st.spinner("Thinking..."):
                    response, explanation, timings, sources = generate_response(
                        prompt,
                        temperature
                    )
            except httpx.HTTPStatusError as e:
                st.session_state.messages.pop()   # drop the unanswered message
                status = e.response.status_code
                if status in (401, 403):
                    st.error("El token de hugging face fue rechazado, revisa el token en al sidebar.")
                elif status == 503:
                    st.warning("El inference endpoint esta iniciando (scaled to zero). "
                               "Vuelve a intentarlo en un momento.")
                else:
                    st.error(f"El inference endpoint dio error {status}: "
                             f"{e.response.text[:300]}")
                st.stop()
            except httpx.RequestError as e:
                st.session_state.messages.pop()   # drop the unanswered message
                st.error(f"No se pudo alcanzar al inference endpoint: {type(e).__name__}: {e}")
                st.stop()
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": response,
                                              "timings": timings, "sources": sources,
                                              "cutoff": relevance_cutoff})
            st.session_state.current_explanation = {
                'input': prompt,
                'output': response,
                'details': explanation,
                'response_time': timings['total'],
                'timings': timings,
                'sources': sources,
                'cutoff': relevance_cutoff,
            }

            # Display assistant message
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
                    render_sources(sources, cutoff=relevance_cutoff)
                    st.caption(format_timings(timings))
            
            st.rerun()
    
    with col2:
        st.subheader("Explainability")
        
        if st.session_state.current_explanation:
            exp = st.session_state.current_explanation
            
            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confianza", f"{exp['details']['confidence']:.2f}")
            with col_b:
                st.metric("Tiempo de respuesta", f"{exp['response_time']:.2f}s")
            if exp.get('timings'):
                st.caption(f"Model: {exp['timings']['model']:.2f} s · "
                           f"Explanation: {exp['timings']['explanation']:.2f} s")

            st.divider()
            
            # Explanation details
            st.markdown("**Factores clave:**")
            for feature in exp['details']['top_features']:
                st.markdown(f"- {feature}")

            if 'sources' in exp:
                st.markdown("**Fuentes utilizadas:**")
                for s in exp['sources']:
                    similarity = f", similarity {s['similarity']:.2f}" if 'similarity' in s else ""
                    st.markdown(f"- {s['title']} ({s['course']}, "
                                f"excerpt #{s['chunk_index']}{similarity})")
                if not exp['sources']:
                    cutoff = exp.get('cutoff', RELEVANCE_CUTOFF_DEFAULT)
                    st.markdown(f"- None: no excerpt reached the {cutoff:.2f} cutoff")

            # What the model remembers of the conversation
            memory = st.session_state.get('memory')
            if memory is not None:
                with st.expander("🧠 Memoria de conversación"):
                    if memory.moving_summary_buffer:
                        st.markdown("**Resumen de turnos anteriores:**")
                        st.write(memory.moving_summary_buffer)
                    else:
                        st.caption("Sin resumen, la conversación cabe en memoria.")
                    recent = memory.chat_memory.messages
                    used = memory.llm.get_num_tokens(" ".join(m.content for m in recent)) if recent else 0
                    st.caption(f"{len(recent)} recent messages kept word-for-word "
                               f"(~{used} / {MEMORY_TOKEN_LIMIT} tokens). Older turns are summarized.")
            if st.session_state.get('memory_warning'):
                st.warning(st.session_state.memory_warning)

            st.divider()

            # Feedback section
            st.markdown("**Proveer feedback:**")
            
            rating = st.radio(
                "Califica la respuesta:",
                options=["👍 Buena", "👎 Mala"],
                key=f"rating_{len(st.session_state.messages)}"
            )
            
            comment = st.text_area(
                "Comentario opcional:",
                placeholder="Qué fue bueno y que puede mejorar?",
                key=f"comment_{len(st.session_state.messages)}"
            )
            
            if st.button("Subir feedback", use_container_width=True):
                save_feedback(
                    exp['input'],
                    exp['output'],
                    rating,
                    comment,
                    sources=exp.get('sources'),
                )
                st.success("✅ Feedback guardado!")
        else:
            st.info("Envía un mensaje para obtener explicación y enviar feedback.")

# ============================================================================
# PAGE: EXPLAINABILITY ANALYSIS
# ============================================================================

def page_explainability():
    """Detailed explainability analysis page."""
    st.title("🔍 Analisis de explicabilidad")
    st.markdown("Un vistazo en las decisiones del modelo y patrones de comportamiento.")
    
    if not st.session_state.messages:
        st.info("Sin conversaciones.")
        return
    
    # Get recent conversations
    conversations = []
    for i in range(0, len(st.session_state.messages), 2):
        if i+1 < len(st.session_state.messages):
            conversations.append({
                'user': st.session_state.messages[i]['content'],
                'assistant': st.session_state.messages[i+1]['content'],
                'sources': st.session_state.messages[i+1].get('sources', []),
                'cutoff': st.session_state.messages[i+1].get('cutoff'),
            })
    
    if not conversations:
        st.warning("Sin conversaciones completas.")
        return
    
    # Select conversation to analyze
    selected_idx = st.selectbox(
        "Selecciona una conversación para analizar:",
        range(len(conversations)),
        format_func=lambda i: f"Conv {i+1}: {conversations[i]['user'][:50]}..."
    )
    
    selected_conv = conversations[selected_idx]
    
    # Display conversation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Entrada de usuario")
        st.markdown(f"```\n{selected_conv['user']}\n```")
        
        st.subheader("Respuesta de modelo")
        st.markdown(f"```\n{selected_conv['assistant']}\n```")

        st.subheader("Fuentes")
        if selected_conv['sources'] or selected_conv['cutoff'] is not None:
            render_sources(selected_conv['sources'], cutoff=selected_conv['cutoff'])
        else:
            st.caption("Ninguna fuente fue observada.")

    with col2:
        st.subheader("Explicabilidad")
        
        # Compute explanation
        exp = compute_explanation(selected_conv['user'], selected_conv['assistant'])
        
        # Display metrics
        st.metric("Tokens de entrada", exp['input_tokens'])
        st.metric("Tokens de respuesta", exp['response_tokens'])
        st.metric("Confianza", f"{exp['confidence']:.2%}")
        
        st.divider()
        
        st.markdown("**Top Features:**")
        for i, feature in enumerate(exp['top_features'], 1):
            st.markdown(f"{i}. {feature}")
    
    # Visualization section
    st.divider()
    st.subheader("Visualización de importancia de features")
    
    # Mock feature importance data
    features = ['Tamaño de entrada', 'Complejidad', 'Relevancia al contexto', 'Paridad semántica', 'Reconocimiento de patrones']
    importance = [0.25, 0.20, 0.30, 0.15, 0.10]
    
    fig = go.Figure(data=[
        go.Bar(x=features, y=importance, marker_color='lightblue')
    ])
    fig.update_layout(
        title="Importancia de features para la respuesta seleccionada",
        xaxis_title="Features",
        yaxis_title="Puntuación de importancia",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
# ============================================================================
# PAGE: KNOWLEDGE BASE (user-added RAG material, password protected)
# ============================================================================

MAX_PASSWORD_ATTEMPTS = 5


def page_knowledge_base():
    st.title("📂 Base de conocimientos")
    st.markdown("El material añadido aqui es buscado por el modelo y citado como fuente.")

    admin_configured = bool(_get_setting("ADMIN_PASSWORD"))
    unlocked = admin_configured and st.session_state.get('kb_unlocked', False)

    if not admin_configured:
        st.warning("Añadir o remover materiales esta deshabilitado: configura ADMIN_PASSWORD "
                   "en secrets.")
    elif not unlocked:
        attempts = st.session_state.get('kb_failed_attempts', 0)
        if attempts >= MAX_PASSWORD_ATTEMPTS:
            st.error("Muchas contraseñas incorrectas, recarga la página.")
        else:
            with st.form("kb_login"):
                password = st.text_input("Contraseña de admin", type="password")
                if st.form_submit_button("Desbloquear"):
                    if check_admin_password(password):
                        st.session_state.kb_unlocked = True
                        st.session_state.kb_failed_attempts = 0
                        st.rerun()
                    else:
                        time.sleep(1)   # slows down guessing
                        st.session_state.kb_failed_attempts = attempts + 1
                        st.error(f"Wrong password ({attempts + 1}/{MAX_PASSWORD_ATTEMPTS}).")
    else:
        col_info, col_lock = st.columns([5, 1])
        col_info.success("Desbloqueado: Puedes añadir o remover material.")
        if col_lock.button("🔒 Bloquear"):
            st.session_state.kb_unlocked = False
            st.rerun()

        tab_upload, tab_paste = st.tabs(["Subir archivos", "Copiar texto"])

        with tab_upload:
            files = st.file_uploader("PDF, TXT or Markdown",
                                     type=["pdf", "txt", "md"], accept_multiple_files=True)
            if files and st.button("Añadir a la base de conocimientos", type="primary"):
                for f in files:
                    try:
                        with st.spinner(f"Indexing {f.name}..."):
                            n = add_user_document(Path(f.name).stem, extract_text_from_upload(f))
                    except Exception as e:
                        st.error(f"{f.name}: could not be read ({type(e).__name__}).")
                        continue
                    if n:
                        st.success(f"{f.name}: {n} excerpts added.")
                    else:
                        st.warning(f"{f.name}: no text found (scanned PDFs need OCR first).")

        with tab_paste:
            with st.form("paste_text", clear_on_submit=True):
                title = st.text_input("Title", placeholder="e.g. Notas de clase - OSPF")
                text = st.text_area("Text", height=220)
                if st.form_submit_button("Añadir a la base de conocimientos", type="primary"):
                    if not title.strip() or not text.strip():
                        st.warning("Enter both a title and some text.")
                    else:
                        with st.spinner("Indexing..."):
                            n = add_user_document(title.strip(), text)
                        st.success(f"'{title.strip()}': {n} excerpts added.")

    st.divider()
    st.subheader("Documentos añadidos")
    user_docs = list_user_documents()
    if not user_docs:
        st.info("Nada añadido aún.")
    for d in user_docs:
        col_a, col_b = st.columns([5, 1])
        col_a.markdown(f"**{d['title']}** — {d['chunks']} excerpts")
        if unlocked and col_b.button("Delete", key=f"del_{d['source_id']}"):
            delete_user_document(d['source_id'])
            st.rerun()

# ============================================================================
# PAGE: FEEDBACK DASHBOARD
# ============================================================================

def page_feedback():
    """User feedback and quality monitoring page."""
    st.title("📊 Dashboard Feedback")
    st.markdown("Monitorea el feedback de usuarios y la calidad de respuestas.")
    
    # Load feedback data
    feedback_df = load_feedback_data()
    
    if feedback_df.empty:
        st.info("No se ha recolectado feedback aún.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Feedback total", len(feedback_df))
    
    with col2:
        positive = len(feedback_df[feedback_df['rating'].str.contains('👍')])
        st.metric("Positivo", positive)
    
    with col3:
        negative = len(feedback_df[feedback_df['rating'].str.contains('👎')])
        st.metric("Negativo", negative)
    
    with col4:
        if len(feedback_df) > 0:
            satisfaction = (positive / len(feedback_df)) * 100
            st.metric("Satisfacción", f"{satisfaction:.1f}%")
    
    st.divider()
    
    # Feedback visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de feedback")
        
        rating_counts = feedback_df['rating'].value_counts()
        fig = px.pie(
            values=rating_counts.values,
            names=rating_counts.index,
            title="Feedback positivo vs negativo"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feedback con el tiempo")
        
        feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
        daily_counts = feedback_df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Feedback Count by Day",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Feedback by source module: which course material leads to helpful / unhelpful answers
    st.subheader("Feedback por módulo fuente")
    with_sources = feedback_df[feedback_df['source_modules'] != ""]
    if with_sources.empty:
        st.info("Ningun feedback con fuentes guardadas.")
    else:
        by_module = (
            with_sources.assign(module=with_sources['source_modules'].str.split("; "))
            .explode('module')
            .assign(result=lambda d: d['rating'].map(
                lambda r: 'Helpful' if '👍' in r else 'Not helpful'))
            .groupby(['module', 'result']).size().reset_index(name='count')
        )
        fig = px.bar(
            by_module,
            x='count',
            y='module',
            color='result',
            orientation='h',
            barmode='stack',
            color_discrete_map={'Helpful': '#2ca02c', 'Not helpful': '#d62728'},
            labels={'count': 'Feedback count', 'module': '', 'result': ''},
            title="Ratings of answers that used each module",
        )
        fig.update_layout(height=max(250, 60 * by_module['module'].nunique()),
                          yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{len(with_sources)} of {len(feedback_df)} feedback entries have recorded sources. "
                   "An answer can use several modules, so it is counted once per module.")

    st.divider()

    # Recent feedback
    st.subheader("Feedback reciente")

    # Display feedback table
    display_df = feedback_df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['message'] = display_df['message'].str[:50] + '...'
    display_df['response'] = display_df['response'].str[:50] + '...'

    st.dataframe(
        display_df[['timestamp', 'message', 'response', 'rating', 'comment', 'source_modules']],
        use_container_width=True,
        hide_index=True
    )
    
    # Export feedback
    st.divider()
    if st.button("📥 Exportar datos de feedback"):
        csv = feedback_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"feedback_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============================================================================
# PAGE: MONITORING
# ============================================================================

def page_monitoring():
    """System monitoring and performance metrics page."""
    st.title("📈 Monitoreo del sistema")
    st.markdown("Rastrea el rendimiento de la aplicación y métricas de uso.")
    
    # Metrics overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Mensajes totales",
            st.session_state.metrics['total_messages']
        )
    
    with col2:
        st.metric(
            "Tiempo de respuesta promedio",
            f"{st.session_state.metrics['avg_response_time']:.2f}s"
        )
    
    with col3:
        st.metric(
            "Feedback total",
            st.session_state.metrics['total_feedback']
        )

    st.divider()

    # Response times
    st.subheader("Tiempos de respuesta")
    log = st.session_state.response_log
    if not log:
        st.info("Sin respuestas, envía un mensaje al chat para guardar el tiempo de respuesta.")
    else:
        times_df = pd.DataFrame(log)
        times_df.insert(0, 'response', range(1, len(times_df) + 1))

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Last", f"{times_df['total'].iloc[-1]:.2f}s")
        col_b.metric("Fastest", f"{times_df['total'].min():.2f}s")
        col_c.metric("Slowest", f"{times_df['total'].max():.2f}s")
        col_d.metric("Avg model time", f"{times_df['model'].mean():.2f}s")

        fig = px.bar(
            times_df,
            x='response',
            y=[part for part in ['retrieval', 'model', 'explanation', 'memory']
               if part in times_df.columns],
            labels={'response': 'Response #', 'value': 'Seconds', 'variable': 'Part'},
            title="Tiempo de respuesta (Retrieval, Llamada del modelo, Explicación LIME, Resumen de memoria)",
        )
        fig.update_layout(height=350, xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")

        with st.expander("Response time log"):
            show_df = times_df.copy()
            show_df['timestamp'] = show_df['timestamp'].dt.strftime('%H:%M:%S')
            show_df['message'] = show_df['message'].str[:60]
            st.dataframe(
                show_df[['response', 'timestamp', 'message', 'model', 'explanation', 'total']]
                    .round({'model': 2, 'explanation': 2, 'total': 2}),
                width="stretch",
                hide_index=True,
            )

    st.divider()

    # Cache status
    st.subheader("Estatus del cache")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cache del modelo**")
        st.success("✅ Modelo cargado y en cache")
        st.markdown("**Datos de cache de feedback**")
        st.info("ℹ️ TTL: 1 hora")
    
    with col2:
        if st.button("🔄 Limpiar cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared!")
            st.rerun()
    
    st.divider()
    
    # Session state inspection
    with st.expander("🔍 Session State (Debug)"):
        st.json({
            'messages_count': len(st.session_state.messages),
            'feedback_count': len(st.session_state.feedback_db),
            'preferences': st.session_state.preferences,
            'metrics': st.session_state.metrics,
            'memory_summary': (st.session_state.memory.moving_summary_buffer
                               if st.session_state.get('memory') else None),
        })
    
    st.divider()
    
    # System recommendations
    st.subheader("Optimization Recommendations")
    
    if st.session_state.metrics['avg_response_time'] > 2.0:
        st.warning("⚠️ Average response time is high. Consider implementing streaming responses.")
    else:
        st.success("✅ Response times are within acceptable range.")
    
    if st.session_state.metrics['total_feedback'] < st.session_state.metrics['total_messages'] * 0.3:
        st.info("ℹ️ Feedback rate is low. Consider making feedback more prominent in the UI.")
    else:
        st.success("✅ Good feedback engagement rate.")

# ============================================================================
# PAGE: DOCUMENTATION
# ============================================================================

def page_documentation():
    """Documentation and team information page."""
    st.title("📚 Documentation")
    
    tab1, tab2, tab3 = st.tabs(["About", "Technical", "Team"])
    
    with tab1:
        st.markdown("""
        ## About This Application
        
        The Trustworthy AI Explainer Dashboard is a comprehensive interface for interacting
        with LLM applications while maintaining transparency and accountability.
        
        ### Features
        
        - **Interactive Chat**: Conversational AI with memory
        - **Explainability**: Real-time analysis of model decisions
        - **Feedback System**: User rating and comment collection
        - **Monitoring**: Performance metrics and usage analytics
        - **Multi-page Dashboard**: Organized interface for different tasks
        
        ### How to Use
        
        1. **Chat**: Go to the Chat page to interact with the AI
        2. **Review**: Check explanations for each response
        3. **Feedback**: Rate responses to help improve the system
        4. **Analyze**: Use the Explainability page for deep analysis
        5. **Monitor**: Track system performance on the Monitoring page
        6. **Add material**: On the Knowledge Base page, admins (ADMIN_PASSWORD) can upload
           PDF/TXT/Markdown files or paste text for the assistant to search
        """)
    
    with tab2:
        st.markdown("""
        ## Technical Documentation
        
        ### Architecture
        
        ```
        Frontend: Streamlit Multi-page App
        Backend: Hugging Face Inference Endpoint (Default engine, Qwen3-8B)
        Explainability: SHAP/LIME
        State: st.session_state
        Caching: @st.cache_resource, @st.cache_data
        ```
        
        ### State Management
        
        This application uses `st.session_state` to persist:
        - Conversation history
        - User preferences
        - Feedback data
        - Performance metrics
        
        ### Caching Strategy
        
        - `@st.cache_resource`: Model loading (expensive, not serialized)
        - `@st.cache_data`: Data loading (serializable, with TTL)
        
        ### Deployment
        
        ```bash
        # Local
        streamlit run streamlit_app_template.py
        
        # Streamlit Community Cloud
        1. Push to GitHub
        2. Connect repo in Streamlit Cloud
        3. Deploy
        
        # Docker
        docker build -t trustworthy-ai-dashboard .
        docker run -p 8501:8501 trustworthy-ai-dashboard
        ```
        
        ### Requirements
        
        ```
        streamlit
        pandas
        plotly
        httpx  # calls the Inference Endpoint
        numpy
        ```
        """)
    
    with tab3:
        st.markdown("""
        ## Team Information
        
        **Team Name**: [Your Team Name]
        
        **Team Members**:
        - Member 1: Oscar Eduardo Lizardi López
        - Member 2: [Role]
        - Member 3: [Role]
        - Member 4: [Role]
        
        **Module**: 15 - App Prototyping with Streamlit
        
        **Project**: Trustworthy AI Explainer Dashboard
        
        ### Team Contributions
        
        - **Gradio Prototype**: [Team Member]
        - **Streamlit Dashboard**: [Team Member]
        - **Explainability Integration**: [Team Member]
        - **Deployment**: [Team Member]
        
        ### Contact
        
        For questions or support, contact: [team@example.com]
        """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🤖 IA con conocimientos de redes")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["💬 Chat", "🔍 Explicación", "📂 Base de conocimientos", "📊 Feedback",
             "📈 Monitoreo", "📚 Documentación"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.caption(f"Session: {len(st.session_state.messages)} messages")

        st.markdown("---")
        st.subheader("🔑 Token de Hugging Face")
        st.text_input(
            "Token de acceso",
            type="password",
            key="hf_token",
            placeholder="hf_...",
            help="Usado solo en esta sesión, no guardado en disco.",
        )
        if st.session_state.hf_token:
            st.success("Usando el token insertado.")
        elif get_api_key():
            st.info("Usando el token configurado.")
        else:
            st.warning("Inserta un token para usar el chat.")
        if not get_endpoint_url():
            st.warning("HF_ENDPOINT_URL no esta configurado en secrets.")
        st.caption(f"Modelo: `{HF_MODEL}`")
    
    # Route to selected page
    if page == "💬 Chat":
        page_chat()
    elif page == "🔍 Explicación":
        page_explainability()
    elif page == "📂 Base de conocimientos":
        page_knowledge_base()
    elif page == "📊 Feedback":
        page_feedback()
    elif page == "📈 Monitoreo":
        page_monitoring()
    elif page == "📚 Documentación":
        page_documentation()

if __name__ == "__main__":
    main()

