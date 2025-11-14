import streamlit as st
import openai
import base64
import random
from datetime import datetime
import io
import tempfile
import os
import speech_recognition as sr
from gtts import gTTS
import threading
import queue
import time
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# Initialize API Key
# ---------------------------
openai.api_key = st.secrets["OPENAI_KEY"]

# ---------------------------
# Exercise Library
# ---------------------------
EXERCISE_LIBRARY = {
    "cardio": {
        "beginner": [
            {"name": "Brisk Walking", "duration": 30, "calories": 150},
            {"name": "Cycling", "duration": 25, "calories": 175},
            {"name": "Jump Rope", "duration": 15, "calories": 200}
        ],
        "intermediate": [
            {"name": "Running", "duration": 30, "calories": 300},
            {"name": "Swimming", "duration": 30, "calories": 250},
            {"name": "HIIT", "duration": 20, "calories": 280}
        ],
        "advanced": [
            {"name": "Sprint Intervals", "duration": 20, "calories": 320},
            {"name": "Hill Sprints", "duration": 25, "calories": 350},
            {"name": "Advanced HIIT", "duration": 25, "calories": 380}
        ]
    },
    "strength": {
        "beginner": [
            {"name": "Push-ups", "sets": 3, "reps": 10, "calories": 70},
            {"name": "Bodyweight Squats", "sets": 3, "reps": 12, "calories": 60},
            {"name": "Plank", "sets": 3, "reps": 30, "calories": 50}
        ],
        "intermediate": [
            {"name": "Bench Press", "sets": 4, "reps": 8, "calories": 110},
            {"name": "Dumbbell Rows", "sets": 4, "reps": 10, "calories": 90},
            {"name": "Lunges", "sets": 3, "reps": 12, "calories": 100}
        ],
        "advanced": [
            {"name": "Heavy Squats", "sets": 5, "reps": 5, "calories": 180},
            {"name": "Deadlifts", "sets": 4, "reps": 6, "calories": 200},
            {"name": "Pull-ups", "sets": 4, "reps": 8, "calories": 120}
        ]
    },
    "flexibility": {
        "all": [
            {"name": "Yoga Flow", "duration": 20, "calories": 90},
            {"name": "Stretching Routine", "duration": 15, "calories": 60},
            {"name": "Mobility Drills", "duration": 10, "calories": 40}
        ]
    }
}

# ---------------------------
# Enhanced Audio Functions for Voice-to-Voice Chat
# ---------------------------

def record_audio(duration=5):
    """Record audio from microphone using speech_recognition"""
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info(f"🎤 Recording for {duration} seconds... Speak now!")
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=1)
            # Record audio
            audio = r.listen(source, timeout=duration, phrase_time_limit=duration)
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            with open(tmp_file.name, "wb") as f:
                f.write(audio.get_wav_data())
            return tmp_file.name
    except sr.WaitTimeoutError:
        st.error("⏰ Recording timeout. Please try again.")
        return None
    except Exception as e:
        st.error(f"🎤 Microphone error: {e}")
        return None

def transcribe_audio(audio_file_path):
    """Transcribe audio using OpenAI Whisper"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        return f"Error transcribing audio: {e}"

def get_voice_response(user_message, conversation_history=[]):
    """Get AI response for voice queries with conversation context"""
    try:
        system_prompt = """You are a friendly, encouraging fitness AI coach. Provide helpful, 
        professional advice about exercise, nutrition, and wellness. Keep responses clear, 
        actionable, and conversational. Be supportive and motivating."""
        
        # Convert conversation history from tuple format to OpenAI message format
        messages = [{"role": "system", "content": system_prompt}]
        
        # Convert the conversation history - handle both 2-tuple and 3-tuple formats
        for message_data in conversation_history[-6:]:  # Last 3 exchanges
            if len(message_data) == 3:
                speaker, msg, _ = message_data
            elif len(message_data) == 2:
                speaker, msg = message_data
            else:
                continue  # Skip invalid formats
                
            role = "user" if speaker == "user" else "assistant"
            messages.append({"role": role, "content": msg})
        
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

def get_ai_response(user_message, context="", conversation_history=[]):
    """Get AI response for text queries with specific context"""
    try:
        if context == "form_analysis":
            system_prompt = """You are an expert fitness coach specializing in exercise form and technique. 
            Provide detailed, technical feedback on exercise form, common mistakes, and corrections.
            Be specific about biomechanics and safety considerations."""
        elif context == "food_analysis":
            system_prompt = """You are a nutritionist and dietitian expert. Provide accurate nutritional 
            analysis, calorie estimates, and health recommendations. Focus on evidence-based nutrition advice."""
        elif context == "posture_analysis":
            system_prompt = """You are a physical therapist and posture specialist. Analyze posture issues, 
            muscle imbalances, and provide corrective exercises and daily habit recommendations."""
        else:
            system_prompt = """You are a friendly, encouraging fitness AI coach. Provide helpful, 
            professional advice about exercise, nutrition, and wellness."""
        
        # Convert conversation history from tuple format to OpenAI message format
        messages = [{"role": "system", "content": system_prompt}]
        
        # Convert the conversation history from (speaker, message) format to {"role": role, "content": content}
        for speaker, message in conversation_history[-6:]:  # Keep last 6 messages for context
            role = "user" if speaker == "user" else "assistant"
            messages.append({"role": role, "content": message})
        
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

def text_to_speech_openai(text, voice="alloy"):
    """Convert text to speech using OpenAI TTS"""
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            response.stream_to_file(temp_file.name)
            return temp_file.name
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

def text_to_speech_gtts(text):
    """Convert text to speech using gTTS (free fallback)"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            return temp_file.name
    except Exception as e:
        st.error(f"Error with gTTS: {e}")
        return None

# ---------------------------
# Visualization Functions
# ---------------------------

def create_thermometer(value, max_value, title, color_scale):
    """Create a thermometer-style gauge chart - Compact version"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'color': 'white', 'size': 14}},
        number = {'font': {'color': 'white', 'size': 20}},
        gauge = {
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "white", 
                    'tickfont': {'color': 'white', 'size': 10}},
            'bar': {'color': color_scale},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_value*0.6], 'color': 'rgba(255, 107, 107, 0.3)'},
                {'range': [max_value*0.6, max_value*0.8], 'color': 'rgba(254, 202, 87, 0.3)'},
                {'range': [max_value*0.8, max_value], 'color': 'rgba(78, 205, 196, 0.3)'}],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        height=250,  # Reduced height
        margin=dict(l=10, r=10, t=40, b=10)  # Reduced margins
    )
    
    return fig

def create_fitness_level_gauge(level):
    """Create a gauge for fitness level - Compact version"""
    level_values = {"beginner": 33, "intermediate": 66, "advanced": 100}
    value = level_values.get(level, 0)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fitness Level", 'font': {'color': 'white', 'size': 14}},
        number = {'suffix': "%", 'font': {'color': 'white', 'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white", 
                    'tickfont': {'color': 'white', 'size': 10}},
            'bar': {'color': "#4ecdc4"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(255, 107, 107, 0.3)'},
                {'range': [33, 66], 'color': 'rgba(254, 202, 87, 0.3)'},
                {'range': [66, 100], 'color': 'rgba(78, 205, 196, 0.3)'}],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        height=250,  # Reduced height
        margin=dict(l=10, r=10, t=40, b=10)  # Reduced margins
    )
    
    return fig

def create_goal_progress(goal):
    """Create goal progress visualization - Compact version"""
    goal_colors = {
        "weight_loss": "#ff6b6b",
        "muscle_gain": "#4ecdc4", 
        "maintenance": "#feca57"
    }
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 50,  # Default progress
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Goal Progress", 'font': {'color': 'white', 'size': 14}},
        number = {'suffix': "%", 'font': {'color': 'white', 'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white", 
                    'tickfont': {'color': 'white', 'size': 10}},
            'bar': {'color': goal_colors.get(goal, "#4ecdc4")},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        height=250,  # Reduced height
        margin=dict(l=10, r=10, t=40, b=10)  # Reduced margins
    )
    
    return fig

# ---------------------------
# Enhanced Helper Functions
# ---------------------------

def calculate_calorie_needs(age, gender, weight, height, activity_level, goal):
    """Calculate daily calorie needs based on user profile"""
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    multipliers = {
        "sedentary": 1.2, 
        "light": 1.375, 
        "moderate": 1.55, 
        "active": 1.725, 
        "very_active": 1.9
    }
    
    maintenance = bmr * multipliers[activity_level]
    
    if goal == "weight_loss":
        return maintenance - 500
    elif goal == "muscle_gain":
        return maintenance + 300
    return maintenance

def generate_workout(fitness_level, day_number, workout_type=None):
    """Generate workout based on fitness level and day"""
    if workout_type is None:
        types = ["cardio", "strength", "flexibility", "rest"]
        workout_type = types[day_number % 4]
    
    if workout_type == "rest":
        return {
            "type": "rest", 
            "name": "Active Recovery", 
            "description": "Light walking or stretching", 
            "calories": 50
        }
    
    if workout_type == "flexibility":
        exercise = random.choice(EXERCISE_LIBRARY["flexibility"]["all"])
        exercise["type"] = "flexibility"
    else:
        exercise = random.choice(EXERCISE_LIBRARY[workout_type][fitness_level])
        exercise["type"] = workout_type
    
    return exercise

def generate_weekly_plan(fitness_level):
    """Generate a weekly workout plan"""
    plan = []
    for day in range(7):
        workout = generate_workout(fitness_level, day)
        plan.append(workout)
    return plan

def analyze_image_prompt(image_file, prompt_text):
    """Send image + prompt to OpenAI GPT for multimodal analysis"""
    try:
        # Convert image to base64
        base64_image = base64.b64encode(image_file.getvalue()).decode('utf-8')
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {e}"

# ---------------------------
# Streamlit UI Configuration
# ---------------------------
st.set_page_config(
    page_title="🏋️ Multimodal Fitness AI Coach", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Dark Theme and Beautiful Buttons
st.markdown("""
<style>
    /* Main Dark Theme */
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    .stApp {
        background-color: #000000;
    }
    .css-1d391kg, .css-1lcbmhc, .css-1outwn7 {
        background-color: #000000;
    }
    
    /* Gradient Buttons - From Uiverse.io by M4rco592 */
    .stButton > button {
        position: relative;
        padding: 16px 32px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        background: transparent;
        border: none;
        cursor: pointer;
        border-radius: 50px;
        overflow: hidden;
        transition: transform 0.2s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.03);
    }

    .stButton > button::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(
            from 0deg,
            #ff6b6b,
            #4ecdc4,
            #45b7d1,
            #96ceb4,
            #feca57,
            #ff9ff3,
            #ff6b6b
        );
        z-index: -2;
        filter: blur(10px);
        transform: rotate(0deg);
        transition: transform 1.5s ease-in-out;
    }

    .stButton > button:hover::before {
        transform: rotate(180deg);
    }

    .stButton > button::after {
        content: "";
        position: absolute;
        inset: 3px;
        background: black;
        border-radius: 47px;
        z-index: -1;
        filter: blur(5px);
    }

    .gradient-text {
        color: transparent;
        background: conic-gradient(
            from 0deg,
            #ff6b6b,
            #4ecdc4,
            #45b7d1,
            #96ceb4,
            #feca57,
            #ff9ff3,
            #ff6b6b
        );
        background-clip: text;
        -webkit-background-clip: text;
        filter: hue-rotate(0deg);
    }

    .stButton > button:hover .gradient-text {
        animation: hue-rotating 2s linear infinite;
    }

    .stButton > button:active {
        transform: scale(0.99);
    }

    @keyframes hue-rotating {
        to {
            filter: hue-rotate(360deg);
        }
    }

    /* Headers and Text */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .tab-header {
        font-size: 2rem;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    
    /* Cards and Containers */
    .metric-card {
        background: rgba(30, 30, 30, 0.8);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .workout-card {
        background: rgba(30, 30, 30, 0.8);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .workout-card:hover {
        transform: translateY(-5px);
        border-color: #4ecdc4;
    }
    
    /* Analysis Results */
    .analysis-result {
        padding: 2rem;
        background: rgba(30, 30, 30, 0.8);
        border-radius: 20px;
        border-left: 5px solid #4ecdc4;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Upload Sections */
    .upload-section {
        border: 3px dashed #4ecdc4;
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        background: rgba(30, 30, 30, 0.5);
        backdrop-filter: blur(10px);
    }
    
    /* Voice Chat Bubbles */
    .voice-chat-bubble {
        padding: 1.5rem 2rem;
        border-radius: 25px;
        margin: 1rem 0;
        max-width: 80%;
        backdrop-filter: blur(10px);
    }
    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    .assistant-bubble {
        background: rgba(78, 205, 196, 0.2);
        color: white;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        border: 1px solid rgba(78, 205, 196, 0.3);
    }

    /* Text Chat Bubbles */
    .text-chat-bubble {
        padding: 1rem 1.5rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        backdrop-filter: blur(10px);
    }
    .text-user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    .text-assistant-bubble {
        background: rgba(255, 107, 107, 0.2);
        color: white;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        border: 1px solid rgba(255, 107, 107, 0.3);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #000000, #1a1a1a);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000000;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #4ecdc4;
        color: black;
        font-weight: bold;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    }

    /* Chat Input */
    .chat-input-container {
        background: rgba(30, 30, 30, 0.8);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<h1 class="main-header">🏋️ Multimodal Fitness AI Coach</h1>', unsafe_allow_html=True)

# ---------------------------
# Sidebar: User Profile
# ---------------------------
with st.sidebar:
    st.markdown("## 👤 Your Profile")
    st.markdown("---")
    
    with st.form("user_profile"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            weight = st.number_input("Weight (kg)", min_value=40, max_value=200, value=70)
            fitness_level = st.selectbox("Fitness Level", ["beginner", "intermediate", "advanced"])
        
        with col2:
            gender = st.selectbox("Gender", ["male", "female"])
            height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170)
            activity_level = st.selectbox("Activity Level", ["sedentary", "light", "moderate", "active", "very_active"])
        
        goal = st.selectbox("Goal", ["weight_loss", "muscle_gain", "maintenance"])
        
        submitted = st.form_submit_button("🚀 Generate My Plan")
        
        if submitted:
            daily_calories = calculate_calorie_needs(age, gender, weight, height, activity_level, goal)
            st.session_state.user_data = {
                "age": age, 
                "gender": gender, 
                "weight": weight, 
                "height": height,
                "fitness_level": fitness_level, 
                "activity_level": activity_level,
                "goal": goal, 
                "daily_calories": daily_calories
            }
            st.session_state.plan_generated = True
            st.session_state.weekly_plan = generate_weekly_plan(fitness_level)
            st.success("🎉 Your personalized fitness plan has been generated!")

# ---------------------------
# Initialize Session State for All Chats
# ---------------------------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "form_chat_history" not in st.session_state:
    st.session_state.form_chat_history = []

if "food_chat_history" not in st.session_state:
    st.session_state.food_chat_history = []

if "posture_chat_history" not in st.session_state:
    st.session_state.posture_chat_history = []

if "voice_option" not in st.session_state:
    st.session_state.voice_option = "alloy"

if "use_openai_tts" not in st.session_state:
    st.session_state.use_openai_tts = True

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Dashboard"

# ---------------------------
# Main Tabs with Navigation
# ---------------------------
def navigate_to_tab(tab_name):
    st.session_state.current_tab = tab_name

tabs = st.tabs(["🏠 Dashboard", "📷 Exercise Analysis", "🍎 Smart Food Nutrition Recognition", "📊 Posture Check", "🎯 Workout Plan", "🎤 Voice Coach"])

# ---------------------------
# Tab 1: Enhanced Dashboard with Thermometers
# ---------------------------
with tabs[0]:
    st.markdown('<h3 class="tab-header">🎯 Your Fitness Dashboard</h3>', unsafe_allow_html=True)
    
    if st.session_state.get('plan_generated', False):
        user = st.session_state.user_data
        
        # Image Placeholder
        st.markdown("### 📸 Your Progress Photo")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="background: rgba(30, 30, 30, 0.8); padding: 3rem; border-radius: 20px; 
                        border: 2px dashed #4ecdc4; text-align: center; margin: 1rem 0;">
                <h4 style="color: #4ecdc4; margin-bottom: 1rem;">🖼️ Progress Photo</h4>
                <p style="color: #888;">Upload your progress photo to track visual changes</p>
                <p style="color: #666; font-size: 0.9rem;">Click here to upload an image</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Thermometer Visualizations - All in one row
        st.markdown("### 📊 Your Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Daily Calories Thermometer
            fig_calories = create_thermometer(
                user['daily_calories'], 
                3000, 
                "Daily Calories", 
                "#ff6b6b"
            )
            st.plotly_chart(fig_calories, use_container_width=True)
        
        with col2:
            # Fitness Level Gauge
            fig_fitness = create_fitness_level_gauge(user['fitness_level'])
            st.plotly_chart(fig_fitness, use_container_width=True)
        
        with col3:
            # BMI Thermometer
            bmi = user["weight"] / ((user["height"] / 100) ** 2)
            fig_bmi = create_thermometer(bmi, 40, "BMI", "#4ecdc4")
            st.plotly_chart(fig_bmi, use_container_width=True)
        
        with col4:
            # Goal Progress
            fig_goal = create_goal_progress(user['goal'])
            st.plotly_chart(fig_goal, use_container_width=True)
        
        # Quick Actions with Navigation
        st.markdown("### ⚡ Quick Actions")
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            if st.button("📸 Analyze Ex  mnercise", key="nav_form"):
                navigate_to_tab("Exercise Analysis")
                st.rerun()
        
        with quick_col2:
            if st.button("🍎 Smart Food Nutrition Recognition", key="nav_food"):
                navigate_to_tab("Smart Food Nutrition Recognition")
                st.rerun()
        
        with quick_col3:
            if st.button("🎤 Ask AI Coach", key="nav_voice"):
                navigate_to_tab("Voice Coach")
                st.rerun()
        
        # Progress Summary
        st.markdown("### 📈 Progress Summary")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.markdown(f'<div class="metric-card"><h3>🔥</h3><h2>{user["daily_calories"]:.0f}</h2><p>Daily Calories</p></div>', unsafe_allow_html=True)
        
        with summary_col2:
            st.markdown(f'<div class="metric-card"><h3>💪</h3><h2>{user["fitness_level"].title()}</h2><p>Fitness Level</p></div>', unsafe_allow_html=True)
        
        with summary_col3:
            goal_display = user["goal"].replace("_", " ").title()
            st.markdown(f'<div class="metric-card"><h3>🎯</h3><h2>{goal_display}</h2><p>Goal</p></div>', unsafe_allow_html=True)
        
        with summary_col4:
            st.markdown(f'<div class="metric-card"><h3>⚖️</h3><h2>{bmi:.1f}</h2><p>BMI</p></div>', unsafe_allow_html=True)
    
    else:
        st.info("👈 Please fill out your profile in the sidebar to generate your personalized fitness plan!")
        st.image("fitness.jpeg", caption="", use_column_width=True)
        
        # Placeholder thermometers in one row
        st.markdown("### 📊 Your Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fig_placeholder1 = create_thermometer(0, 3000, "Daily Calories", "#666")
            st.plotly_chart(fig_placeholder1, use_container_width=True)
        
        with col2:
            fig_placeholder2 = create_fitness_level_gauge("beginner")
            st.plotly_chart(fig_placeholder2, use_container_width=True)
        
        with col3:
            fig_placeholder3 = create_thermometer(0, 40, "BMI", "#666")
            st.plotly_chart(fig_placeholder3, use_container_width=True)
        
        with col4:
            fig_placeholder4 = create_goal_progress("maintenance")
            st.plotly_chart(fig_placeholder4, use_container_width=True)

# ---------------------------
# Tab 2: Exercise Form Analysis with Chat
# ---------------------------
with tabs[1]:
    st.markdown('<h3 class="tab-header">📷 AI Exercise Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🏋️ Exercise Selection")
        exercise_name = st.selectbox(
            "Choose Exercise", 
            ["Squat", "Deadlift", "Bench Press", "Push-up", "Pull-up", "Lunge", "Shoulder Press"],
            key="form_exercise"
        )
        
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Upload your exercise photo", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of you performing the exercise",
            key="form_upload"
        )
    
    with col2:
        if uploaded_file:
            st.image(uploaded_file, caption="Your Exercise Analysis", use_column_width=True)
            
            if st.button("🔍 Analyze My Exercise", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your Exercise..."):
                    prompt_text = f"""
                    Analyze this person performing {exercise_name}. Provide detailed feedback on:
                    1. Posture and alignment
                    2. Common mistakes to watch for
                    3. Specific corrections needed
                    4. Safety tips
                    
                    Be constructive and specific in your feedback.
                    """
                    result = analyze_image_prompt(uploaded_file, prompt_text)
                
                st.markdown("### 💡 Exercise Analysis Results")
                st.markdown(f'<div class="analysis-result">{result}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="upload-section">'
                       '<h3>📸 Upload Your Exercise Photo</h3>'
                       '<p>Get AI-powered feedback on your exercise analysis</p>'
                       '</div>', unsafe_allow_html=True)
    
    # Form Analysis Chat
    st.markdown("### 💬 Ask About Exercise Analysis")
    
    # Display chat history
    if not st.session_state.form_chat_history:
        st.info("💡 Ask me anything about exercise analysis, technique, or corrections!")
    else:
        for i, (speaker, message) in enumerate(st.session_state.form_chat_history):
            if speaker == "user":
                st.markdown(f'<div class="text-chat-bubble text-user-bubble">'
                           f'<strong>You:</strong> {message}'
                           '</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="text-chat-bubble text-assistant-bubble">'
                           f'<strong>Form Coach:</strong> {message}'
                           '</div>', unsafe_allow_html=True)
    
    # Chat input
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    
    with col1:
        form_question = st.text_input(
            "Ask about exercise analysis...",
            placeholder="E.g., How can I improve my squat depth?",
            key="form_question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("Send", key="form_send", use_container_width=True):
            if form_question:
                # Add user message to chat history
                st.session_state.form_chat_history.append(("user", form_question))
                
                # Get AI response
                with st.spinner("Form coach is thinking..."):
                    ai_response = get_ai_response(form_question, "form_analysis", st.session_state.form_chat_history)
                
                # Add AI response to chat history
                st.session_state.form_chat_history.append(("assistant", ai_response))
                
                # Clear input and rerun
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick form questions
    st.markdown("#### 💡 Quick Questions")
    quick_q_col1, quick_q_col2 = st.columns(2)
    
    form_quick_questions = [
        "How do I maintain proper form during weightlifts?",
        "What are common squat mistakes?",
        "How can I improve my bench press technique?",
        "What's the proper push-up form?",
        "How to avoid back pain during exercises?",
        "What's the correct lunge technique?"
    ]
    
    for i, question in enumerate(form_quick_questions):
        col = quick_q_col1 if i % 2 == 0 else quick_q_col2
        with col:
            if st.button(f"💬 {question}", key=f"form_q_{i}", use_container_width=True):
                st.session_state.form_chat_history.append(("user", question))
                with st.spinner("Form coach is thinking..."):
                    ai_response = get_ai_response(question, "form_analysis", st.session_state.form_chat_history)
                st.session_state.form_chat_history.append(("assistant", ai_response))
                st.rerun()

# ---------------------------
# Tab 3: Food Analysis with Chat
# ---------------------------
with tabs[2]:
    st.markdown('<h3 class="tab-header">🍎 Smart Food Nutrition Recognition</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📤 Upload Food Photo")
        food_image = st.file_uploader(
            "Upload your meal photo", 
            type=["jpg", "jpeg", "png"], 
            key="food_uploader",
            help="Upload a clear photo of your food or meal"
        )
    
    with col2:
        if food_image:
            st.image(food_image, caption="Your Meal", use_column_width=True)
            
            if st.button("🔍 Analyze This Meal", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing nutrition content..."):
                    prompt_text = """
                    Analyze this food image and provide:
                    1. Food items identification
                    2. Estimated calorie count
                    3. Macronutrient breakdown (carbs, protein, fats)
                    4. Health rating (1-10) with explanation
                    5. Nutritional benefits and considerations
                    
                    Be realistic and helpful in your analysis.
                    """
                    result = analyze_image_prompt(food_image, prompt_text)
                
                st.markdown("### 📊 Nutrition Analysis")
                st.markdown(f'<div class="analysis-result">{result}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="upload-section">'
                       '<h3>📸 Upload Your Food Photo</h3>'
                       '<p>Get instant nutritional analysis of your meals</p>'
                       '</div>', unsafe_allow_html=True)
    
    # Food Analysis Chat
    st.markdown("### 💬 Ask About Nutrition")
    
    # Display chat history
    if not st.session_state.food_chat_history:
        st.info("💡 Ask me anything about nutrition, diet, or healthy eating!")
    else:
        for i, (speaker, message) in enumerate(st.session_state.food_chat_history):
            if speaker == "user":
                st.markdown(f'<div class="text-chat-bubble text-user-bubble">'
                           f'<strong>You:</strong> {message}'
                           '</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="text-chat-bubble text-assistant-bubble">'
                           f'<strong>Nutritionist:</strong> {message}'
                           '</div>', unsafe_allow_html=True)
    
    # Chat input
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    
    with col1:
        food_question = st.text_input(
            "Ask about nutrition...",
            placeholder="E.g., What should I eat before workout?",
            key="food_question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("Send", key="food_send", use_container_width=True):
            if food_question:
                # Add user message to chat history
                st.session_state.food_chat_history.append(("user", food_question))
                
                # Get AI response
                with st.spinner("Nutritionist is thinking..."):
                    ai_response = get_ai_response(food_question, "food_analysis", st.session_state.food_chat_history)
                
                # Add AI response to chat history
                st.session_state.food_chat_history.append(("assistant", ai_response))
                
                # Clear input and rerun
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick food questions
    st.markdown("#### 💡 Quick Questions")
    food_q_col1, food_q_col2 = st.columns(2)
    
    food_quick_questions = [
        "What are good protein sources for vegetarians?",
        "How many calories should I eat daily?",
        "What's a healthy breakfast option?",
        "How to read nutrition labels?",
        "Best post-workout meals?",
        "Healthy snacks for weight loss?"
    ]
    
    for i, question in enumerate(food_quick_questions):
        col = food_q_col1 if i % 2 == 0 else food_q_col2
        with col:
            if st.button(f"💬 {question}", key=f"food_q_{i}", use_container_width=True):
                st.session_state.food_chat_history.append(("user", question))
                with st.spinner("Nutritionist is thinking..."):
                    ai_response = get_ai_response(question, "food_analysis", st.session_state.food_chat_history)
                st.session_state.food_chat_history.append(("assistant", ai_response))
                st.rerun()

# ---------------------------
# Tab 4: Posture Analysis with Chat
# ---------------------------
with tabs[3]:
    st.markdown('<h3 class="tab-header">📊 Posture Check</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📤 Upload Full-Body Photo")
        posture_image = st.file_uploader(
            "Upload full-body posture photo", 
            type=["jpg", "jpeg", "png"], 
            key="posture_uploader",
            help="Upload a clear full-body photo in standing position"
        )
    
    with col2:
        if posture_image:
            st.image(posture_image, caption="Posture Analysis", use_column_width=True)
            
            if st.button("🔍 Analyze My Posture", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing your posture..."):
                    prompt_text = """
                    Analyze this full-body image for posture assessment. Provide:
                    1. Overall posture alignment
                    2. Potential muscle imbalances
                    3. Spinal curvature observations
                    4. Recommended corrective exercises
                    5. Daily habits for improvement
                    
                    Focus on actionable advice and be encouraging.
                    """
                    result = analyze_image_prompt(posture_image, prompt_text)
                
                st.markdown("### 💡 Posture Analysis Results")
                st.markdown(f'<div class="analysis-result">{result}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="upload-section">'
                       '<h3>📸 Upload Full-Body Photo</h3>'
                       '<p>Get AI-powered posture analysis and improvement tips</p>'
                       '</div>', unsafe_allow_html=True)
    
    # Posture Analysis Chat
    st.markdown("### 💬 Ask About Posture")
    
    # Display chat history
    if not st.session_state.posture_chat_history:
        st.info("💡 Ask me anything about posture, alignment, or corrective exercises!")
    else:
        for i, (speaker, message) in enumerate(st.session_state.posture_chat_history):
            if speaker == "user":
                st.markdown(f'<div class="text-chat-bubble text-user-bubble">'
                           f'<strong>You:</strong> {message}'
                           '</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="text-chat-bubble text-assistant-bubble">'
                           f'<strong>Posture Specialist:</strong> {message}'
                           '</div>', unsafe_allow_html=True)
    
    # Chat input
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    
    with col1:
        posture_question = st.text_input(
            "Ask about posture...",
            placeholder="E.g., How can I fix rounded shoulders?",
            key="posture_question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("Send", key="posture_send", use_container_width=True):
            if posture_question:
                # Add user message to chat history
                st.session_state.posture_chat_history.append(("user", posture_question))
                
                # Get AI response
                with st.spinner("Posture specialist is thinking..."):
                    ai_response = get_ai_response(posture_question, "posture_analysis", st.session_state.posture_chat_history)
                
                # Add AI response to chat history
                st.session_state.posture_chat_history.append(("assistant", ai_response))
                
                # Clear input and rerun
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick posture questions
    st.markdown("#### 💡 Quick Questions")
    posture_q_col1, posture_q_col2 = st.columns(2)
    
    posture_quick_questions = [
        "How to improve sitting posture?",
        "Best exercises for forward head posture?",
        "How to fix rounded shoulders?",
        "What causes lower back pain?",
        "Exercises for better spinal alignment?",
        "How to check my own posture?"
    ]
    
    for i, question in enumerate(posture_quick_questions):
        col = posture_q_col1 if i % 2 == 0 else posture_q_col2
        with col:
            if st.button(f"💬 {question}", key=f"posture_q_{i}", use_container_width=True):
                st.session_state.posture_chat_history.append(("user", question))
                with st.spinner("Posture specialist is thinking..."):
                    ai_response = get_ai_response(question, "posture_analysis", st.session_state.posture_chat_history)
                st.session_state.posture_chat_history.append(("assistant", ai_response))
                st.rerun()

# ---------------------------
# Tab 5: Enhanced Workout Plan
# ---------------------------
with tabs[4]:
    st.markdown('<h3 class="tab-header">🎯 Your Workout Plan</h3>', unsafe_allow_html=True)
    
    if st.session_state.get('plan_generated', False):
        user = st.session_state.user_data
        
        # Day Selection and View Type
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            selected_day = st.slider("Select Day", 1, 30, 1)
            view_type = st.radio("View", ["Daily Workout", "Weekly Plan"])
        
        with col3:
            st.metric("Selected Day", f"Day {selected_day}")
            if st.button("🔄 Generate New Workout", use_container_width=True):
                st.session_state.weekly_plan = generate_weekly_plan(user['fitness_level'])
                st.rerun()
        
        if view_type == "Daily Workout":
            # Daily workout details
            workout = generate_workout(user['fitness_level'], selected_day - 1)
            
            if workout["type"] == "rest":
                st.markdown('<div class="workout-card" style="border-color: #feca57;">'
                           '<h2>🏖️ Active Recovery Day</h2>'
                           '<p style="font-size: 1.2rem; color: #feca57;">Today is your rest day - recovery is essential for growth!</p>'
                           '<div style="background: rgba(254, 202, 87, 0.1); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">'
                           '<h4>💫 Recommended Activities:</h4>'
                           '<ul style="color: white;">'
                           '<li>Light walking (20-30 minutes)</li>'
                           '<li>Gentle stretching or yoga</li>'
                           '<li>Foam rolling</li>'
                           '<li>Mobility exercises</li>'
                           '</ul>'
                           '<p style="color: #feca57; font-style: italic;">Remember: Recovery is when your body gets stronger! 💪</p>'
                           '</div>'
                           '</div>', unsafe_allow_html=True)
            else:
                # Workout type colors
                type_colors = {
                    "cardio": "#ff6b6b",
                    "strength": "#4ecdc4", 
                    "flexibility": "#45b7d1"
                }
                color = type_colors.get(workout["type"], "#4ecdc4")
                
                st.markdown(f'<div class="workout-card" style="border-color: {color};">'
                           f'<h2>📅 Day {selected_day}: {workout["name"]}</h2>'
                           f'<div style="display: inline-block; background: {color}20; color: {color}; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; margin: 0.5rem 0;">'
                           f'{workout["type"].title()} Workout'
                           '</div>'
                           '</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="workout-card">'
                               '<h3>🏋️ Exercise Details</h3>'
                               '<div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px;">'
                               f'<h4 style="color: #4ecdc4;">{workout["name"]}</h4>', unsafe_allow_html=True)
                    
                    if "sets" in workout:
                        st.markdown(f'<p>🔢 <strong>Sets:</strong> {workout["sets"]}</p>'
                                   f'<p>🔄 <strong>Reps:</strong> {workout["reps"]}</p>', unsafe_allow_html=True)
                    
                    if "duration" in workout:
                        st.markdown(f'<p>⏱️ <strong>Duration:</strong> {workout["duration"]} minutes</p>', unsafe_allow_html=True)
                    
                    st.markdown(f'<p>🔥 <strong>Calories Burned:</strong> {workout["calories"]} cal</p>'
                               '</div></div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="workout-card">'
                               '<h3>💡 Pro Tips</h3>'
                               '<div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px;">', unsafe_allow_html=True)
                    
                    tips = {
                        "cardio": "🏃‍♂️ Maintain steady breathing and focus on maintaining good form throughout your cardio session.",
                        "strength": "💪 Control the movement - don't rush. Focus on muscle-mind connection and proper form.",
                        "flexibility": "🌬️ Breathe deeply into each stretch. Don't force it - listen to your body."
                    }
                    
                    st.markdown(f'<p>{tips.get(workout["type"], "🎯 Listen to your body and maintain proper form throughout your workout.")}</p>'
                               '</div></div>', unsafe_allow_html=True)
                
                # Progress tracker
                st.markdown("### 📊 Workout Completion")
                completion = st.slider("How much did you complete?", 0, 100, 0, format="%d%%", key=f"day_{selected_day}")
                if completion == 100:
                    st.balloons()
                    st.success("🎉 Amazing! Workout completed! Your body thanks you!")
        
        else:  # Weekly Plan View
            st.markdown("### 📅 Your Weekly Workout Schedule")
            
            if 'weekly_plan' not in st.session_state:
                st.session_state.weekly_plan = generate_weekly_plan(user['fitness_level'])
            
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            for i, (day, workout) in enumerate(zip(days, st.session_state.weekly_plan)):
                type_colors = {
                    "cardio": "#ff6b6b",
                    "strength": "#4ecdc4", 
                    "flexibility": "#45b7d1",
                    "rest": "#feca57"
                }
                color = type_colors.get(workout["type"], "#4ecdc4")
                
                with st.expander(f"{day}: {workout['name']} • {workout['type'].title()}", expanded=i==0):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f'<div style="color: {color}; font-weight: bold; font-size: 1.1rem;">{workout["type"].title()} Workout</div>', unsafe_allow_html=True)
                        if "duration" in workout:
                            st.markdown(f'⏱️ **Duration:** {workout["duration"]} mins')
                        if "sets" in workout:
                            st.markdown(f'🔢 **Sets:** {workout["sets"]} | 🔄 **Reps:** {workout["reps"]}')
                        st.markdown(f'🔥 **Calories:** {workout["calories"]} cal')
                    
                    with col2:
                        completion = st.slider(f"Completion", 0, 100, 0, key=f"week_{i}")
                        if completion > 0:
                            st.progress(completion/100)
    
    else:
        st.info("👈 Please generate your fitness plan from the sidebar first!")
        st.markdown("""
        <div class="workout-card">
        <h3>🎯 What to Expect in Your Workout Plan:</h3>
        <ul style="color: white;">
        <li>🏋️ <strong>Personalized workouts</strong> based on your fitness level</li>
        <li>⚖️ <strong>Balanced routine</strong> with cardio, strength, and flexibility</li>
        <li>😴 <strong>Proper rest days</strong> for optimal recovery</li>
        <li>🔥 <strong>Calorie estimates</strong> for each workout</li>
        <li>📈 <strong>Progressive overload</strong> as you get stronger</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Tab 6: Enhanced Voice Coach
# ---------------------------
with tabs[5]:
    st.markdown('<h3 class="tab-header">🎤 Your AI Voice Coach</h3>', unsafe_allow_html=True)
    
    # Voice settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.voice_option = st.selectbox(
            "Choose AI Voice",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            help="Select the voice for AI responses",
            key="voice_select"
        )
    
    with col2:
        recording_duration = st.slider("Recording Duration (seconds)", 3, 10, 5)
    
    with col3:
        st.session_state.use_openai_tts = st.checkbox("Use OpenAI TTS (Higher Quality)", value=True)
    
    # Display conversation history
    st.markdown("### 💬 Conversation History")
    
    if not st.session_state.conversation_history:
        st.info("🎤 Start a conversation by recording your voice or using a sample question!")
    else:
        for i, (speaker, message, audio_path) in enumerate(st.session_state.conversation_history):
            if speaker == "user":
                st.markdown(f'<div class="voice-chat-bubble user-bubble">'
                           f'<strong>You:</strong> {message}'
                           '</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="voice-chat-bubble assistant-bubble">'
                           f'<strong>Coach:</strong> {message}'
                           '</div>', unsafe_allow_html=True)
                
                # Show audio player for assistant responses
                if audio_path and os.path.exists(audio_path):
                    st.audio(audio_path, format="audio/mp3")
    
    # Voice recording section
    st.markdown("### 🎙️ Live Voice Chat")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Start Recording", type="primary", use_container_width=True):
            with st.spinner("Recording in progress..."):
                audio_file = record_audio(recording_duration)
                
            if audio_file:
                # Transcribe audio
                with st.spinner("Transcribing your voice..."):
                    user_text = transcribe_audio(audio_file)
                
                if user_text and not user_text.startswith("Error"):
                    # Add to conversation history
                    st.session_state.conversation_history.append(("user", user_text, None))
                    
                    # Get AI response
                    with st.spinner("Coach is thinking..."):
                        ai_response = get_voice_response(user_text, st.session_state.conversation_history)
                    
                    # Generate TTS for AI response
                    with st.spinner("Generating voice response..."):
                        if st.session_state.use_openai_tts:
                            audio_path = text_to_speech_openai(ai_response, st.session_state.voice_option)
                        else:
                            audio_path = text_to_speech_gtts(ai_response)
                    
                    # Add AI response to conversation history
                    if audio_path:
                        st.session_state.conversation_history.append(("assistant", ai_response, audio_path))
                    else:
                        st.session_state.conversation_history.append(("assistant", ai_response, None))
                    
                    # Force refresh to show new messages
                    st.rerun()
                else:
                    st.error("Could not transcribe audio. Please try again.")
                
                # Clean up temporary audio file
                try:
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
                except:
                    pass
    
    with col2:
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Sample questions for quick start
    st.markdown("### 💡 Try These Sample Questions")
    sample_col1, sample_col2 = st.columns(2)
    
    sample_questions = [
        "How can I improve my squat form?",
        "What should I eat before my workout?",
        "Give me a 15-minute home workout",
        "How do I fix lower back pain during deadlifts?",
        "What's the best cardio for weight loss?",
        "How much protein do I need daily?"
    ]
    
    for i, question in enumerate(sample_questions):
        col = sample_col1 if i % 2 == 0 else sample_col2
        with col:
            if st.button(f"🎤 {question}", key=f"sample_{i}", use_container_width=True):
                # Add sample question to conversation (as 3-tuple with None for audio path)
                st.session_state.conversation_history.append(("user", question, None))
                
                # Get AI response
                with st.spinner("Coach is thinking..."):
                    ai_response = get_voice_response(question, st.session_state.conversation_history)
                
                # Generate TTS
                with st.spinner("Generating voice response..."):
                    if st.session_state.use_openai_tts:
                        audio_path = text_to_speech_openai(ai_response, st.session_state.voice_option)
                    else:
                        audio_path = text_to_speech_gtts(ai_response)
                
                # Add AI response to conversation history (as 3-tuple)
                if audio_path:
                    st.session_state.conversation_history.append(("assistant", ai_response, audio_path))
                else:
                    st.session_state.conversation_history.append(("assistant", ai_response, None))
                
                st.rerun()

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Soft Tech Group 2025 • Your AI Fitness Companion"
    "</div>",
    unsafe_allow_html=True
)

# Clean up old audio files
def cleanup_old_audio_files():
    """Clean up audio files older than 1 hour"""
    try:
        current_time = time.time()
        for _, _, audio_path in st.session_state.conversation_history:
            if audio_path and os.path.exists(audio_path):
                file_age = current_time - os.path.getctime(audio_path)
                if file_age > 3600:  # 1 hour
                    os.unlink(audio_path)
    except:
        pass

# Run cleanup
cleanup_old_audio_files()