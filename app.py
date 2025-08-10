import streamlit as st
import sys
import subprocess

# Ensure aifc is installed (required for speech_recognition)
# aifc is part of Python's standard library — no install needed
import aifc


# Now import other dependencies
import speech_recognition as sr
import PyPDF2
import docx2txt
import google.generativeai as genai
import os
import io
import tempfile
from audio_recorder_streamlit import audio_recorder
import base64
from datetime import datetime

# Configuration for Gemini API - Hardcoded (NOT RECOMMENDED FOR PUBLIC REPOS)
GEMINI_API_KEY = "AIzaSyDg8vSbT_wueoiGwi_0W9UjJLkPNjhLHwY"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"temperature": 0.7, "max_output_tokens": 2048},
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
)

# ========== TEXT EXTRACTION ==========
@st.cache_data(show_spinner=False)
def extract_text(uploaded_file):
    try:
        if uploaded_file.name.endswith('.pdf'):
            with io.BytesIO(uploaded_file.read()) as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
                
        elif uploaded_file.name.endswith('.docx'):
            with io.BytesIO(uploaded_file.read()) as f:
                return docx2txt.process(f).strip()
                
        else:
            return "Unsupported file type. Please upload PDF or DOCX."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# ========== AUDIO TRANSCRIPTION ==========
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand audio. Please speak clearly."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service: {str(e)}"
    except Exception as e:
        return f"Audio processing error: {str(e)}"

# ========== GEMINI RESPONSE ==========
def generate_response(question, context_text):
    try:
        # Format prompt for better context understanding
        prompt = f"Document content:\n{context_text}\n\n---\n\nQuestion: {question}\n\nAnswer:"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ========== STREAMLIT UI ==========
st.set_page_config(
    page_title="DocVoice Pro - Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/light theme support
st.markdown("""
<style>
    :root {
        --primary: #1f618d;
        --secondary: #2980b9;
        --background: #121212;
        --surface: #1e1e1e;
        --on-primary: #ffffff;
        --on-secondary: #ffffff;
        --on-background: #e0e0e0;
        --on-surface: #f5f5f5;
        --success: #27ae60;
        --warning: #f39c12;
        --error: #e74c3c;
        --user-msg-bg: #2c3e50;
        --bot-msg-bg: #34495e;
    }
    
    [data-theme="light"] {
        --background: #f5f7fa;
        --surface: #ffffff;
        --on-background: #333333;
        --on-surface: #444444;
        --user-msg-bg: #d1ecf1;
        --bot-msg-bg: #e8f4f8;
    }
    
    body {
        background-color: var(--background);
        color: var(--on-background);
        transition: background-color 0.3s, color 0.3s;
    }
    
    .header {
        color: var(--primary);
        text-align: center;
        padding: 1rem;
        border-bottom: 2px solid var(--secondary);
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, var(--surface) 0%, rgba(30,30,30,0.9) 100%);
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .sidebar {
        background-color: var(--surface);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        background-color: var(--surface);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        height: 65vh;
        overflow-y: auto;
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .user-msg {
        background-color: var(--user-msg-bg);
        color: var(--on-surface);
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 0 15px;
        margin: 1rem 0;
        max-width: 85%;
        margin-left: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .bot-msg {
        background-color: var(--bot-msg-bg);
        color: var(--on-surface);
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 15px 0;
        margin: 1rem 0;
        max-width: 85%;
        margin-right: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .user-msg::before, .bot-msg::before {
        content: "";
        position: absolute;
        width: 0;
        height: 0;
        border-style: solid;
    }
    
    .user-msg::before {
        border-width: 0 0 15px 15px;
        border-color: transparent transparent var(--user-msg-bg) transparent;
        right: -10px;
        top: 0;
    }
    
    .bot-msg::before {
        border-width: 0 15px 15px 0;
        border-color: transparent var(--bot-msg-bg) transparent transparent;
        left: -10px;
        top: 0;
    }
    
    .record-btn {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: var(--on-primary);
        border-radius: 30px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
        background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
    }
    
    .stTextInput>div>div>input {
        background-color: var(--surface);
        color: var(--on-surface);
        border-radius: 30px;
        padding: 0.75rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stRadio>div {
        background-color: var(--surface);
        padding: 1rem;
        border-radius: 10px;
    }
    
    .stFileUploader>div>div>div>div {
        background-color: var(--surface);
        border: 1px dashed rgba(255,255,255,0.2);
        border-radius: 10px;
    }
    
    .stSpinner>div>div {
        border-color: var(--primary) transparent transparent transparent;
    }
    
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-weight: bold;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
    }
    
    .bot-avatar {
        background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
        color: white;
    }
    
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    
    .message-content {
        line-height: 1.6;
    }
    
    .timestamp {
        font-size: 0.75rem;
        opacity: 0.7;
        text-align: right;
        margin-top: 5px;
    }
    
    .logo {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .logo h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .logo p {
        color: var(--secondary);
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'doc_text' not in st.session_state:
    st.session_state.doc_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'audio_recorded' not in st.session_state:
    st.session_state.audio_recorded = False
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'reset_input' not in st.session_state:
    st.session_state.reset_input = False

# App header
st.markdown("""
<div class="logo">
    <h1>📄 DocVoice Pro</h1>
    <p>Chat with your documents using voice commands</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for document upload
with st.sidebar:
    st.markdown('<div class="sidebar">', unsafe_allow_html=True)
    
    st.subheader("📂 Step 1: Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=['pdf', 'docx'], label_visibility="collapsed")
    
    if uploaded_file:
        with st.spinner("🔍 Extracting text..."):
            st.session_state.doc_text = extract_text(uploaded_file)
        st.success("✅ Document processed successfully!")
        st.info(f"📄 Extracted {len(st.session_state.doc_text)} characters")
    
    st.divider()
    st.subheader("🎤 Step 2: Ask Questions")
    input_method = st.radio("Select input method:", ["Record Audio", "Upload Audio", "Type Text"])
    
    question_text = ""
    audio_uploaded = False
    
    if st.session_state.reset_input:
        question_text = ""
        audio_file = None
        audio_bytes = None
        st.session_state.reset_input = False
        st.session_state.audio_recorded = False
        st.rerun()
    
    if input_method == "Record Audio":
        st.info("🎙️ Click the mic and speak clearly")
        audio_bytes = audio_recorder(
            pause_threshold=3.0,
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#6c757d",
            icon_size="2x"
        )
        
        if audio_bytes and not st.session_state.audio_recorded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
                
            with st.spinner("🔊 Transcribing audio..."):
                question_text = transcribe_audio(tmp_path)
                if question_text.startswith("Could not understand audio") or question_text.startswith("Error"):
                    st.error(question_text)
                else:
                    st.session_state.audio_recorded = True
            os.unlink(tmp_path)
            
    elif input_method == "Upload Audio":
        audio_file = st.file_uploader("Upload audio file (WAV format only)", type=['wav'])
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
                
            with st.spinner("🔊 Transcribing audio..."):
                question_text = transcribe_audio(tmp_path)
                if question_text.startswith("Could not understand audio") or question_text.startswith("Error"):
                    st.error(question_text)
            os.unlink(tmp_path)
            
    else:  # Type Text
        question_text = st.text_input("✏️ Enter your question:", label_visibility="collapsed", key="text_input")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main chat area
st.subheader("💬 Chat with Your Document")
chat_container = st.container()

if st.session_state.doc_text == "":
    st.warning("ℹ️ Please upload a document in the sidebar to get started")
else:
    # Display chat history
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for role, msg, timestamp in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"""
                <div class="user-msg">
                    <div class="message-header">
                        <div class="avatar user-avatar">👤</div>
                        <div><strong>You</strong></div>
                    </div>
                    <div class="message-content">{msg}</div>
                    <div class="timestamp">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-msg">
                    <div class="message-header">
                        <div class="avatar bot-avatar">🤖</div>
                        <div><strong>Assistant</strong></div>
                    </div>
                    <div class="message-content">{msg}</div>
                    <div class="timestamp">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process question
    if question_text and question_text != st.session_state.last_question:
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        st.session_state.chat_history.append(("user", question_text, timestamp))
        st.session_state.last_question = question_text
        
        with st.spinner("🧠 Generating response..."):
            response = generate_response(question_text, st.session_state.doc_text)
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append(("bot", response, timestamp))
        
        # Set flag to reset inputs
        st.session_state.reset_input = True
        st.rerun()

# Instructions and theme toggle
st.sidebar.divider()
st.sidebar.info("""
**📌 How to use:**
1. Upload a PDF/DOCX document
2. Choose input method (voice or text)
3. Ask questions about your document
4. View conversation history

**💡 Tips:**
- Speak clearly when using voice input
- Keep audio recordings under 30 seconds
- Use headphones for better voice recognition
""")

# Theme toggle
st.sidebar.divider()
theme = st.sidebar.radio("🎨 Theme:", ["Light", "Dark"], index=1)
if theme == "Light":
    st.markdown('<style>[data-theme="light"] {}</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>[data-theme="dark"] {}</style>', unsafe_allow_html=True)

# Clear chat button
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.last_question = ""
    st.rerun()

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px; color: #777; font-size: 0.8rem;">
    Powered by Gemini AI • v1.3
</div>
""", unsafe_allow_html=True)

