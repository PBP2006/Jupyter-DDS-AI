# ============================================================================
# RUN STREAMLIT APP FROM JUPYTER NOTEBOOK
# Paste this entire code into ONE Jupyter Notebook cell and run it
# ============================================================================

# STEP 1: Install required packages (run once)
!pip install streamlit google-generativeai pandas plotly

# STEP 2: Create the Streamlit app file
app_code = '''
import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import json
import re
from datetime import datetime

st.set_page_config(
    page_title="Sentiment AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .positive-badge {
        background-color: #28a745;
        color: white;
    }
    .negative-badge {
        background-color: #dc3545;
        color: white;
    }
    .neutral-badge {
        background-color: #ffc107;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

API_KEY = "AIzaSyAyJEaDQDKiF7jKDzkERfPXqs8ajuvP53s"

@st.cache_resource
def initialize_llm():
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(
            "gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        return model
    except Exception as e:
        st.error(f"Error: {e}")
        return None

SYSTEM_PROMPT = """You are an expert sentiment analysis AI assistant. 

When analyzing sentiment, provide:
- Sentiment: Positive, Negative, or Neutral
- Confidence: 0-1
- Reasoning: Clear explanation
- Key phrases and emotions

Be conversational and helpful!"""

def chat_with_llm(model, user_message, chat_history):
    conversation = f"{SYSTEM_PROMPT}\\n\\n"
    
    for msg in chat_history[-5:]:
        role = msg["role"].capitalize()
        content = msg["content"]
        conversation += f"{role}: {content}\\n"
    
    conversation += f"\\nUser: {user_message}\\nAssistant:"
    
    try:
        response = model.generate_content(conversation)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def extract_sentiment(response_text):
    sentiment = "Neutral"
    confidence = 0.7
    
    lower_text = response_text.lower()
    if "positive" in lower_text and "sentiment" in lower_text:
        sentiment = "Positive"
        confidence = 0.85
    elif "negative" in lower_text and "sentiment" in lower_text:
        sentiment = "Negative"
        confidence = 0.85
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "reasoning": response_text
    }

def main():
    st.markdown("""
        <div class="chat-header">
            <h1>🤖 Sentiment AI Assistant</h1>
            <p>Your Intelligent Companion for Sentiment Analysis</p>
            <p style="font-size: 14px;">Powered by Google Gemini LLM</p>
        </div>
    """, unsafe_allow_html=True)
    
    model = initialize_llm()
    if model is None:
        st.error("Failed to initialize. Check API key.")
        return
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Hi! I am your Sentiment AI Assistant. I can analyze emotions in text. Try:\\n\\n- Analyze this: [your text]\\n- What is sentiment analysis?\\n- Explain my analysis\\n\\nWhat would you like to do?"
        }]
    
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    
    with st.sidebar:
        st.markdown("## 🎯 Quick Actions")
        
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "👋 Ready for a new conversation!"
            }]
            st.rerun()
        
        st.markdown("---")
        st.markdown("## 💡 Examples")
        
        examples = [
            "Analyze: This is amazing!",
            "Analyze: I am disappointed",
            "What is sentiment analysis?",
            "How do you detect emotions?"
        ]
        
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.pending_input = ex
                st.rerun()
        
        st.markdown("---")
        st.markdown("## 📊 Stats")
        
        if st.session_state.analysis_history:
            total = len(st.session_state.analysis_history)
            positive = sum(1 for a in st.session_state.analysis_history if a.get("sentiment") == "Positive")
            negative = sum(1 for a in st.session_state.analysis_history if a.get("sentiment") == "Negative")
            neutral = total - positive - negative
            
            st.metric("Total", total)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("😊", positive)
            with col2:
                st.metric("😞", negative)
            with col3:
                st.metric("😐", neutral)
            
            if st.button("📈 Chart", use_container_width=True):
                fig = px.pie(
                    names=["Positive", "Negative", "Neutral"],
                    values=[positive, negative, neutral],
                    color_discrete_map={
                        "Positive": "#28a745",
                        "Negative": "#dc3545",
                        "Neutral": "#ffc107"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analyses yet!")
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.info("LLM-powered sentiment analysis chatbot. Ask questions, analyze text, learn about emotions!")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sentiment_data" in message:
                data = message["sentiment_data"]
                sentiment = data.get("sentiment", "Unknown")
                confidence = data.get("confidence", 0)
                
                badge_class = {
                    "Positive": "positive-badge",
                    "Negative": "negative-badge",
                    "Neutral": "neutral-badge"
                }.get(sentiment, "neutral-badge")
                
                st.markdown(f"""
                    <div class="sentiment-badge {badge_class}">
                        {sentiment} ({confidence:.0%})
                    </div>
                """, unsafe_allow_html=True)
    
    prompt = None
    if "pending_input" in st.session_state:
        prompt = st.session_state.pending_input
        del st.session_state.pending_input
    else:
        prompt = st.chat_input("💬 Type your message or paste text to analyze...")
    
    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = chat_with_llm(model, prompt, st.session_state.messages)
            
            st.markdown(response)
            
            sentiment_data = extract_sentiment(response)
            
            message_data = {
                "role": "assistant",
                "content": response,
                "sentiment_data": sentiment_data
            }
            
            st.session_state.analysis_history.append(sentiment_data)
            st.session_state.messages.append(message_data)
            
            sentiment = sentiment_data.get("sentiment", "Unknown")
            confidence = sentiment_data.get("confidence", 0)
            
            badge_class = {
                "Positive": "positive-badge",
                "Negative": "negative-badge",
                "Neutral": "neutral-badge"
            }.get(sentiment, "neutral-badge")
            
            st.markdown(f"""
                <div class="sentiment-badge {badge_class}">
                    {sentiment} ({confidence:.0%})
                </div>
            """, unsafe_allow_html=True)
        
        st.rerun()
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Chat", use_container_width=True):
            export = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "stats": st.session_state.analysis_history
            }
            st.download_button(
                "💾 Download",
                json.dumps(export, indent=2),
                f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    
    with col2:
        if st.button("📊 Analytics", use_container_width=True):
            if st.session_state.analysis_history:
                df = pd.DataFrame(st.session_state.analysis_history)
                st.dataframe(df, use_container_width=True)
    
    with col3:
        if st.button("❓ Help", use_container_width=True):
            st.info("""
            **How to use:**
            - Paste text to analyze
            - Ask questions
            - Get explanations
            - Compare sentiments
            """)
    
    st.markdown("""
        <p style="text-align: center; color: white; background: rgba(0,0,0,0.3); 
        padding: 10px; border-radius: 5px;">
            🤖 Powered by Google Gemini | DDS Academy 2025
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
'''

# STEP 3: Save the app to a file
with open('sentiment_app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print("✅ Streamlit app file created: sentiment_app.py")
print("=" * 70)

# STEP 4: Run Streamlit (this will open in a new browser tab)
print("🚀 Starting Streamlit app...")
print("=" * 70)
print("⚠️ IMPORTANT:")
print("1. A new browser tab will open automatically")
print("2. The app will run at: http://localhost:8501")
print("3. To STOP the app: Press Ctrl+C in the terminal")
print("=" * 70)

# Run the Streamlit app
!streamlit run sentiment_app.py