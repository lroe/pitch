# -*- coding: utf-8 -*-
"""
pitchine.py - MODIFIED with robust 2-step transcription feature.
Backend now handles interim transcript requests and waits for a 'send_composed_text'
command before executing the full AI logic.
FIX: Always sends a response for interim transcripts to prevent frontend getting stuck.
MAJOR UPDATE: Implemented server-side Google OAuth2 flow to bypass mobile browser
restrictions on client-side redirects.
FINAL FIX: Added static file serving for the frontend.
"""

# 2. IMPORTS
import fastapi
import uvicorn
from fastapi import WebSocket, WebSocketDisconnect, File, UploadFile, Request
from fastapi.responses import RedirectResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.config import Config
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
import asyncio
import aiofiles
import os
import json
import google.generativeai as genai
import traceback
import re
from datetime import datetime
import shutil
import firebase_admin
from firebase_admin import credentials, firestore, auth

# 3. INITIAL SETUP
app = fastapi.FastAPI()

# Add session middleware for OAuth state
# IMPORTANT: You must set a secret key in your environment variables for this to work.
app.add_middleware(SessionMiddleware, secret_key=os.environ.get('SESSION_SECRET_KEY', 'a_default_secret_key_for_dev'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OAUTH2 SERVER-SIDE CONFIG ---
config = Config()
oauth = OAuth(config)

# You must get these from your Google Cloud Console -> APIs & Services -> Credentials -> OAuth 2.0 Client IDs
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name='google',
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        client_kwargs={
            'scope': 'openid email profile'
        }
    )
    print("Server-side Google OAuth configured.")
else:
    print("ERROR: GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET not found in environment. Server-side auth will fail.")


# --- FIREBASE ADMIN SDK INITIALIZATION ---
fb_db = None
try:
    # This path should be correct for the deployment environment
    SERVICE_ACCOUNT_KEY_PATH = 'pitchine-ed6c2-firebase-adminsdk-fbsvc-11654bf63e.json' [cite: 3]
    if os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH) [cite: 3]
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred) [cite: 3]
        fb_db = firestore.client() [cite: 3]
        print("Firebase Admin SDK initialized successfully.")
    else:
        print(f"ERROR: Firebase service account key not found at {SERVICE_ACCOUNT_KEY_PATH}. Firestore integration will be disabled.")
except Exception as e:
    print(f"ERROR: Could not initialize Firebase Admin SDK: {e}")


# --- GOOGLE CLOUD AUTHENTICATION & v2 CLIENT ---
PROJECT_ID = None
speech_client_v2 = None
try:
    # This path should be correct for the deployment environment
    SERVICE_ACCOUNT_FILE_GCP = 'semiotic-mender-461407-n2-9d029397fc74.json' [cite: 3]
    if os.path.exists(SERVICE_ACCOUNT_FILE_GCP):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE_GCP [cite: 3]
        with open(SERVICE_ACCOUNT_FILE_GCP, 'r') as f: [cite: 3]
            data = json.load(f) [cite: 3]
            PROJECT_ID = data.get('project_id') [cite: 3]
        if not PROJECT_ID: [cite: 3]
            raise ValueError("Google Cloud Project ID could not be determined from the service account file.")
        speech_client_v2 = SpeechClient(client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")) [cite: 3]
        print(f"Google Cloud Speech v2 client initialized for project: {PROJECT_ID}")
    else:
        print(f"ERROR: GCP service account key not found at {SERVICE_ACCOUNT_FILE_GCP}. Speech-to-Text will be disabled.")
except Exception as e:
    print(f"ERROR: Could not initialize Google Cloud Speech v2 client: {e}")
    speech_client_v2 = None

# --- Gemini API Configuration & Models ---
GEMINI_API_KEY = None
moderation_model = None
pitch_eval_model = None
analysis_model = None
try:
    # In a production environment like Render, secrets are set as environment variables.
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') [cite: 3]
    if GEMINI_API_KEY: [cite: 3]
        genai.configure(api_key=GEMINI_API_KEY) [cite: 3]
        moderation_model = genai.GenerativeModel("gemini-1.5-pro-latest") [cite: 3]
        pitch_eval_model = genai.GenerativeModel("gemini-1.5-pro-latest") [cite: 3]
        analysis_model = genai.GenerativeModel("gemini-1.5-pro-latest") [cite: 3]
        print("Gemini API key loaded from environment variable. All models initialized.")
    else:
        print("ERROR: GEMINI_API_KEY not found in environment variables. Gemini models will not be available.")
except Exception as e:
    print(f"ERROR: Could not get Gemini API key from environment or initialize models: {e}")

# --- Storage Setup ---
STORAGE_DIR = "pitch_history" [cite: 3]
AUDIO_REPLAY_DIR = os.path.join(STORAGE_DIR, "audio_replays") [cite: 3]
os.makedirs(STORAGE_DIR, exist_ok=True) [cite: 3]
os.makedirs(AUDIO_REPLAY_DIR, exist_ok=True) [cite: 3]


# --- Investor Persona Definitions ---
INVESTOR_PERSONAS={
    "Alex Chen": {
      "system_prompt": """You are Alex Chen, a detail-oriented and analytical early-stage investor. You come from an operations or finance background. You focus on the core mechanics of the business: go-to-market strategy, unit economics, and operational plans. Your goal is to test the founder's grasp of the tactical details required to build a real, sustainable company. You are skeptical of grand visions without a concrete plan to achieve them.

      **Termination Protocol:**
      - If a founder *persistently* avoids direct questions about their business model or operational plan after you have redirected them once, you may end the meeting.
      - Your reasoning should be professional and direct.
      - **Prefix your response with the special command: [TERMINATE_SESSION].**
      - Example: "[TERMINATE_SESSION] We seem to be going in circles, and I'm not getting the clarity I need on the core operational questions. I think it's best we stop here for now. I wish you the best."

      **Questioning Style (General):**
      - Be direct, concise, and focused on the 'how'.
      - Ask clear, specific questions about numbers, processes, and execution.
      - Your tone is professional and grounded, not aggressive. You're trying to understand if the business is viable.

      **Drill-Down on Vagueness Protocol:**
      - If a founder gives a generic or high-level answer to an operational question, your immediate follow-up must ask for a more specific, concrete detail.
      - Example: If the founder says they will acquire customers through "content marketing," your response should be: "Can you be more specific? What specific channels are you prioritizing for that content, and what is your target customer acquisition cost (CAC) for the first year?"

      **Challenge Evasion / Irrelevance Protocol:**
      - If a founder answers a specific tactical question with a broad, visionary statement, or otherwise dodges the question:
          - Politely but firmly steer the conversation back to the original point.
          - Example: "I appreciate the context on the long-term vision, but I want to make sure I understand the fundamentals first. Could we go back to the unit economics? How do you see the gross margin evolving as you scale?"

      **Your Task:**
      - Review the conversation history.
      - **1. Check for Termination:** Does the founder's response meet the criteria for the Termination Protocol? If so, end the session professionally.
      - **2. Address Last Exchange:**
          - Did the founder give a vague operational answer? Apply the "Drill-Down" protocol to get more detail.
          - Did they evade a specific question? Apply the "Challenge Evasion" protocol to guide them back on topic.
      - **3. New Question:** If the previous point was addressed, ask a new, insightful question about the business's operations, go-to-market, or financial model.
      - Your output should be a single, professional question or a termination statement.
      """
    },
    "Maria Santos": {
      "system_prompt": """You are Maria Santos, an investor who believes that the quality and drive of the founding team is the most important predictor of a startup's success. You focus on the 'who' and the 'why': the founder's personal connection to the problem, their resilience, and the dynamics of the team. Your goal is to understand the people behind the idea and gauge their ability to handle the immense pressure of building a company.

      **Termination Protocol:**
      - If a founder is persistently unwilling to discuss their personal motivation, their team, or how they handle conflict, you may end the meeting. This indicates a lack of transparency that you find concerning.
      - Your termination should be gentle but firm.
      - **Prefix your response with the special command: [TERMINATE_SESSION].**
      - Example: "[TERMINATE_SESSION] It seems like you're still figuring out some key aspects of the team's alignment. I think it might be a bit too early for us, but I appreciate you sharing your story with me today."

      **Questioning Style (General):**
      - Ask thoughtful, open-ended questions about their personal journey, team history, and how they handle adversity.
      - Your tone is empathetic and curious, but you listen carefully for inconsistencies or red flags.

      **Drill-Down on Vagueness Protocol:**
      - If a founder makes a generic statement about their team or culture, gently ask for a specific, real-world example.
      - Example: If the founder says "we have a great company culture," your response should be: "I'm glad to hear that. Could you tell me about a specific time a team member disagreed with a major decision you made? I'm curious to learn about your process for handling that."

      **Challenge Evasion / Irrelevance Protocol:**
      - If a founder dodges a question about a past failure or a team conflict:
          - Explain why the question is important to you and gently ask it again.
          - Example: "I understand that can be tough to talk about, but learning how founders navigate hardship is a key part of my process. Could you perhaps tell me what the biggest lesson was from that experience?"

      **Your Task:**
      - Review the conversation history.
      - **1. Check for Termination:** Does the founder's response meet the criteria for your Termination Protocol?
      - **2. Address Last Exchange:**
          - Was the founder's answer about their team or journey vague? Use the "Drill-Down" protocol to ask for a specific story.
          - Did they evade a question about conflict or motivation? Use the "Challenge Evasion" protocol to gently re-engage.
      - **3. New Question:** If the previous point was addressed, ask a new, insightful question about the founder's personal story, their relationship with their co-founders, or their vision for the team.
      - Your output should be a single, thoughtful question or a termination statement.
      """
    },
    "Ben Carter": {
      "system_prompt": """You are Ben Carter, a forward-looking, strategic investor. You focus on the big picture: market size, long-term vision, and a startup's defensible moat. You push founders to think bigger and challenge their assumptions about the market and their competition. Your goal is to determine if the idea has the potential to become a true category-defining company.

      **Termination Protocol:**
      - If it becomes clear after a few exchanges that the founder's ambition is fundamentally misaligned with the venture-scale returns your fund targets (e.g., they describe a small, niche business with no desire to scale), you may end the meeting.
      - Your termination should be professional and based on a lack of strategic fit.
      - **Prefix your response with the special command: [TERMINATE_SESSION].**
      - Example: "[TERMINATE_SESSION] I appreciate you walking me through the model. Based on our conversation, I'm struggling to see the path to the venture-scale outcome we typically look for. This doesn't seem to be the right fit for our fund, but I wish you the best of luck."

      **Questioning Style (General):**
      - Ask expansive, strategic questions. Focus on 'what if' and 'why now'.
      - Challenge the market size, competitive landscape, and long-term defensibility.
      - Your tone is intellectually curious and challenging, but respectful.

      **Drill-Down on Vagueness Protocol:**
      - If a founder makes a broad claim about their market or vision, ask for the rigorous thinking and data that backs it up.
      - Example: If the founder says "our market is massive," your response should be: "Can you help me understand the scale? Could you walk me through your bottoms-up TAM calculation and the key assumptions you're making?"

      **Challenge Evasion / Irrelevance Protocol:**
      - If a founder responds to a strategic question with a short-term, tactical answer:
          - Acknowledge their point but steer them back to the bigger picture.
          - Example: "That's an interesting feature for the next release, but I'm thinking further out. How do you plan to build a structural moat around this business that prevents a major incumbent from building the same thing in two years?"

      **Your Task:**
      - Review the conversation history.
      - **1. Check for Termination:** Is there a fundamental misalignment on scale and ambition?
      - **2. Address Last Exchange:**
          - Was the founder's answer about their market or vision too generic? Use the "Drill-Down" protocol to request their methodology.
          - Did they answer a strategic question with a tactical detail? Use the "Challenge Evasion" protocol to elevate the conversation.
      - **3. New Question:** If the previous point was addressed, ask a new, challenging question about the company's long-term vision, competitive strategy, or the fundamental reason for its existence.
      - Your output should be a single, strategic question or a termination statement.
      """
    }
} [cite: 3]


# --- KILL SWITCH EVALUATION FUNCTIONS ---
async def check_for_inappropriate_content(text: str, model: genai.GenerativeModel): [cite: 3]
    if not text.strip(): return True, "No content to moderate." [cite: 3]
    if not model: return True, "Moderation model not available." [cite: 3]
    prompt = f"""You are a strict moderator for a professional startup pitch meeting. Analyze the following founder's statement. Your only job is to determine if the statement is abusive, hateful, contains slurs, is sexually explicit, or is a clear attempt to troll. If the statement is acceptable for a professional (even if bad) pitch, respond with ONLY the word "SAFE". If the statement is unacceptable, respond with ONLY the word "UNSAFE". Founder's statement: "{text}". Your response:""" [cite: 3]
    try:
        response = await run_in_threadpool(model.generate_content, prompt) [cite: 3]
        decision = response.text.strip().upper() [cite: 3]
        if decision == "UNSAFE": [cite: 3]
            return False, "Inappropriate or abusive language detected." [cite: 3]
        return True, "Content is safe." [cite: 3]
    except Exception as e:
        return True, "Moderation check errored out." [cite: 3]

async def evaluate_pitch_opening(text: str, model: genai.GenerativeModel): [cite: 3]
    if not text.strip(): return 'TERMINATE', "Founder said nothing." [cite: 3]
    if not model: return 'PROCEED', "Pitch evaluation model not available." [cite: 3]
    prompt = f"""You are a seasoned early-stage investor. You have just heard a founder's opening statement. Your task is to decide if it's a waste of time. A pitch is a 'waste of time' if it's completely incoherent, uses excessive jargon with no substance, or fails to clearly state what the company actually does. If the pitch is merely unpolished but you can figure out what they do, it's acceptable. If the pitch is a waste of time, respond with ONLY the word 'TERMINATE'. Otherwise, respond with ONLY the word 'PROCEED'. Founder's opening statement: "{text}". Your response:""" [cite: 3]
    try:
        response = await run_in_threadpool(model.generate_content, prompt) [cite: 3]
        decision = response.text.strip().upper() [cite: 3]
        if decision == "TERMINATE": [cite: 3]
            return 'TERMINATE', "The opening pitch was unclear and a waste of time." [cite: 3]
        return 'PROCEED', "Pitch is clear enough." [cite: 3]
    except Exception as e:
        return 'PROCEED', "Evaluation check errored out." [cite: 3]


# --- PITCH ANALYSIS CLASS ---
class PitchAnalyzer: [cite: 3]
    def __init__(self, conversation_history): [cite: 3]
        self.conversation_history = conversation_history [cite: 3]
        self.analysis_results = {} [cite: 3]
        self.analysis_model = analysis_model [cite: 3]
    def _format_history_for_analysis(self): [cite: 3]
        return "\n".join(f"{entry.get('role', 'System')}: {entry.get('content', '')}" for entry in self.conversation_history) [cite: 3]
    async def _get_analysis_from_gemini(self, prompt, max_retries=2): [cite: 3]
        if not self.analysis_model: return "Error: Analysis model not available." [cite: 3]
        for attempt in range(max_retries): [cite: 3]
            try:
                response = await run_in_threadpool(self.analysis_model.generate_content, prompt) [cite: 3]
                return response.text.strip() [cite: 3]
            except Exception as e:
                if attempt >= max_retries - 1: return f"Error: Could not get a response from the analysis model. Details: {e}" [cite: 3]
        return "Error: Analysis failed after multiple retries." [cite: 3]
    def _parse_numerical_score(self, text_response, scale_max=5): [cite: 3]
        if text_response is None: return None [cite: 3]
        match = re.search(r'\b([1-5])\b', text_response) [cite: 3]
        if match: [cite: 3]
            try:
                score = int(match.group(1)); [cite: 3]
                if 1 <= score <= scale_max: return score [cite: 3]
            except (ValueError, IndexError): pass [cite: 3]
        return "N/A" [cite: 3]
    async def analyze_pitch(self): [cite: 3]
        full_conversation_text = self._format_history_for_analysis() [cite: 3]
        dad_prompt = f"Based on the following startup pitch conversation, assess if it sounds 'Default Alive' or 'Default Dead'. A '[System: ... hesitation]' note indicates the founder was unprepared for a question or was silent. This is a major negative signal. Factor this heavily into your assessment of their viability.\n\nConversation:\n{full_conversation_text}\n\nAssessment:" [cite: 3]
        dad_assessment = await self._get_analysis_from_gemini(dad_prompt) [cite: 3]
        self.analysis_results["default_alive_dead"] = dad_assessment [cite: 3]
        pillars_config = { [cite: 3]
            "Problem/Solution Fit": "Assess if a hair-on-fire problem was articulated and if the solution is obviously better for those users.",
            "Formidable Founders (Clarity & Conviction)": "Assess how 'formidable' the founders seem based on clarity, directness, and confidence. Critically, if you see a '[System: ... hesitation]' note, it means the founder froze under pressure. This should lead to a very low score for this pillar.",
            "Market / 'Why Now?'": "Evaluate the articulation of market size, opportunity, and the timeliness ('Why Now?') of the solution."
        }
        self.analysis_results["pillars"] = {} [cite: 3]
        for pillar_name, detail in pillars_config.items(): [cite: 3]
            score_prompt = f"""
**Primary Context**: An initial assessment of the pitch concluded the startup is '{dad_assessment}'.
**Your Task**: Based on this primary context AND the full conversation below, score the specific pillar '{pillar_name}' from 1 (Poor) to 5 (Excellent).
- **Pillar Detail**: {detail}
- **Crucial Instruction**: The final score MUST be consistent with the primary context. For example, a 'Default Dead' assessment means scores should be low (likely 1-2). A '[System: ... hesitation]' note, especially for the 'Formidable Founders' pillar, must result in a very low score.
- **Output Format**: Output *ONLY* the numerical score (1-5).
**Full Conversation:**
{full_conversation_text}
**Score for {pillar_name}:**""" [cite: 3]
            feedback_prompt = f"""
**Primary Context**: An initial assessment of the pitch concluded the startup is '{dad_assessment}'.
**Your Task**: Based on the primary context and the full conversation, provide specific, bullet-point feedback on '{pillar_name}'.
- **Instructions**: If the founder hesitated, call it out directly. Your feedback must align with the '{dad_assessment}' conclusion, explaining how this pillar contributed to it. Keep feedback concise.
- **Output Format**: Bullet points.
**Full Conversation:**
{full_conversation_text}
**Feedback for {pillar_name}:**""" [cite: 3]
            score_response = await self._get_analysis_from_gemini(score_prompt) [cite: 3]
            feedback_response = await self._get_analysis_from_gemini(feedback_prompt) [cite: 3]
            self.analysis_results["pillars"][pillar_name] = { [cite: 3]
                "score": self._parse_numerical_score(score_response),
                "feedback": [line.strip() for line in feedback_response.splitlines() if line.strip()]
            }
        brutal_prompt = f"""
**Primary Context**: An initial assessment of the pitch concluded the startup is '{dad_assessment}'.
**Your Task**: Based on this primary context and the full conversation, provide specific, brutally honest, and actionable feedback points.
- **Instructions**: Your feedback must explain *why* the pitch was deemed '{dad_assessment}'. If you see notes about hesitation, make that a primary point of feedback. Use direct language. No fluff.
- **Output Format**: Bullet points.
**Full Conversation:**
{full_conversation_text}
**Brutally Honest Feedback:**""" [cite: 3]
        self.analysis_results["brutal_feedback"] = [line.strip() for line in (await self._get_analysis_from_gemini(brutal_prompt)).splitlines() if line.strip()] [cite: 3]
        top_3_prompt = f"""
**Primary Context**: An initial assessment of the pitch concluded the startup is '{dad_assessment}'.
**Your Task**: Based on the primary context and the conversation, identify the top 3 most critical areas to improve.
- **Instructions**: If the assessment was 'Default Dead' or involved hesitation, your suggestions must directly address the root causes (e.g., 'Answering questions under pressure,' 'Articulating the core problem clearly').
**Full Conversation:**
{full_conversation_text}
**Top 3 Areas for Next Practice:**""" [cite: 3]
        self.analysis_results["top_3_areas"] = [line.strip() for line in (await self._get_analysis_from_gemini(top_3_prompt)).splitlines() if line.strip()] [cite: 3]
        return self.analysis_results [cite: 3]

# --- SESSION MANAGEMENT ---
def save_session_to_firestore(user_uid, session_id, report_data): [cite: 3]
    if not fb_db: [cite: 3]
        print("Firestore not initialized. Cannot save session.") [cite: 3]
        return [cite: 3]
    if not user_uid or not session_id: [cite: 3]
        print("Cannot save session, user_uid or session_id missing.") [cite: 3]
        return [cite: 3]
    try:
        session_ref = fb_db.collection('users').document(user_uid).collection('sessions').document(session_id) [cite: 3]
        report_data_with_timestamp = report_data.copy() [cite: 3]
        report_data_with_timestamp['timestamp'] = firestore.SERVER_TIMESTAMP [cite: 3]
        session_ref.set(report_data_with_timestamp) [cite: 3]
        print(f"Session for user {user_uid}, session {session_id} saved to Firestore.") [cite: 3]
    except Exception as e:
        print(f"Error saving session to Firestore: {e}") [cite: 3]

def get_history_from_firestore(user_uid): [cite: 3]
    if not fb_db: [cite: 3]
        print("Firestore not initialized. Cannot fetch history.") [cite: 3]
        return [] [cite: 3]
    if not user_uid: [cite: 3]
        return [] [cite: 3]
    try:
        sessions_ref = fb_db.collection('users').document(user_uid).collection('sessions').order_by('timestamp', direction=firestore.Query.DESCENDING).stream() [cite: 3]
        user_history = [] [cite: 3]
        for session_doc in sessions_ref: [cite: 3]
            doc_data = session_doc.to_dict() [cite: 3]
            if 'timestamp' in doc_data and hasattr(doc_data['timestamp'], 'isoformat'): [cite: 3]
                 doc_data['timestamp'] = doc_data['timestamp'].isoformat() [cite: 3]
            user_history.append(doc_data) [cite: 3]
        print(f"Found {len(user_history)} historical records for user {user_uid} from Firestore.") [cite: 3]
        return user_history [cite: 3]
    except Exception as e:
        print(f"Error reading history from Firestore: {e}") [cite: 3]
        return [] [cite: 3]

def save_session_to_local_file(user_identifier, report_data): [cite: 3]
    if not user_identifier: return [cite: 3]
    filename = "".join(c for c in user_identifier if c.isalnum() or c in ('_','-')).rstrip() [cite: 3]
    filepath = os.path.join(STORAGE_DIR, f"{filename}.jsonl") [cite: 3]
    try:
        session_record = {"timestamp": datetime.now().isoformat(), "report": report_data} [cite: 3]
        with open(filepath, 'a') as f: f.write(json.dumps(session_record) + '\n') [cite: 3]
    except Exception as e: print(f"Error saving session to local file: {e}") [cite: 3]

def get_history_from_local_file(user_identifier): [cite: 3]
    if not user_identifier: return [] [cite: 3]
    filename = "".join(c for c in user_identifier if c.isalnum() or c in ('_','-')).rstrip() [cite: 3]
    filepath = os.path.join(STORAGE_DIR, f"{filename}.jsonl") [cite: 3]
    if not os.path.exists(filepath): return [] [cite: 3]
    try:
        user_history = [] [cite: 3]
        with open(filepath, 'r') as f: [cite: 3]
            for line in f: [cite: 3]
                if line.strip(): user_history.append(json.loads(line)) [cite: 3]
        return user_history [cite: 3]
    except Exception as e: return [] [cite: 3]

class ConnectionManager: [cite: 3]
    def __init__(self): [cite: 3]
        self.active_connections: dict[WebSocket, dict] = {} [cite: 3]
        self.investor_names = list(INVESTOR_PERSONAS.keys()) [cite: 3]
    def initialize_investors(self): [cite: 3]
        if not GEMINI_API_KEY: return None [cite: 3]
        investor_chats = {} [cite: 3]
        for name, persona in INVESTOR_PERSONAS.items(): [cite: 3]
            if persona.get("system_prompt") == "...": continue [cite: 3]
            model_instance = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", system_instruction=persona["system_prompt"]) [cite: 3]
            investor_chats[name] = model_instance.start_chat(history=[]) [cite: 3]
        return investor_chats [cite: 3]
    async def connect(self, websocket: WebSocket, user_uid: str = None): [cite: 3]
        await websocket.accept() [cite: 3]
        self.active_connections[websocket] = { [cite: 3]
            "user_uid": user_uid,
            "investor_chats": self.initialize_investors(),
            "investor_turn_index": 0,
            "conversation_history": [],
            "last_investor_name": None,
            "startup_details": None,
            "initial_context_sent": False,
            "mode": "strict",
            "opening_evaluated": False,
            "current_session_id": f"session_{datetime.now().timestamp()}",
            "latest_audio_blob": None
        }
    def disconnect(self, websocket: WebSocket): [cite: 3]
        if websocket in self.active_connections: [cite: 3]
            del self.active_connections[websocket] [cite: 3]
    def get_connection_data(self, websocket: WebSocket): [cite: 3]
        return self.active_connections.get(websocket) [cite: 3]

    def reset_session_state(self, websocket: WebSocket): [cite: 3]
        conn_data = self.get_connection_data(websocket) [cite: 3]
        if conn_data: [cite: 3]
            conn_data["conversation_history"] = [] [cite: 3]
            conn_data["last_investor_name"] = None [cite: 3]
            conn_data["startup_details"] = None [cite: 3]
            conn_data["initial_context_sent"] = False [cite: 3]
            conn_data["opening_evaluated"] = False [cite: 3]
            conn_data["mode"] = "strict" [cite: 3]
            conn_data["investor_chats"] = self.initialize_investors() [cite: 3]
            conn_data["investor_turn_index"] = 0 [cite: 3]
            conn_data["latest_audio_blob"] = None [cite: 3]
            print(f"Server-side session state reset for websocket: {websocket.client}") [cite: 3]

    def get_next_investor(self, websocket: WebSocket): [cite: 3]
        conn_data = self.get_connection_data(websocket) [cite: 3]
        if not conn_data or not self.investor_names or not conn_data.get("investor_chats"): return None [cite: 3]
        valid_investor_names = [name for name in self.investor_names if name in conn_data["investor_chats"]] [cite: 3]
        if not valid_investor_names: return None [cite: 3]
        current_index = conn_data["investor_turn_index"] % len(valid_investor_names) [cite: 3]
        investor_name = valid_investor_names[current_index] [cite: 3]
        conn_data["investor_turn_index"] = (conn_data["investor_turn_index"] + 1) [cite: 3]
        conn_data["last_investor_name"] = investor_name [cite: 3]
        return investor_name [cite: 3]
manager = ConnectionManager()

# --- SERVER-SIDE AUTH ENDPOINTS ---
@app.get('/login/google')
async def login_via_google(request: Request):
    # The redirect_uri must match EXACTLY what is in your Google Cloud OAuth Client ID configuration
    redirect_uri = request.url_for('auth_via_google')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get('/auth/google')
async def auth_via_google(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if user_info and user_info.get('email_verified'):
            uid = user_info.get('sub') # Use Google's unique subject ID as the UID
            email = user_info.get('email')
            display_name = user_info.get('name')
            
            try:
                firebase_user = auth.get_user_by_email(email)
            except auth.UserNotFoundError:
                firebase_user = auth.create_user(uid=uid, email=email, display_name=display_name)
            
            custom_token = auth.create_custom_token(firebase_user.uid)
            
            # Redirect back to the root of the site, passing the custom token
            frontend_url = f"/?logintoken={custom_token.decode('utf-8')}"
            return RedirectResponse(url=frontend_url)
            
    except Exception as e:
        print(f"ERROR during Google auth callback: {e}")
        return RedirectResponse(url="/?error=auth_failed")
    
    return RedirectResponse(url="/?error=unknown")


# --- API ENDPOINTS ---
@app.post("/upload-audio/{session_id}")
async def upload_audio(session_id: str, file: UploadFile = File(...)): [cite: 3]
    safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ('_','-')).rstrip() [cite: 3]
    if not safe_session_id: [cite: 3]
        return {"error": "Invalid session ID"}, 400
    file_path = os.path.join(AUDIO_REPLAY_DIR, f"{safe_session_id}.webm") [cite: 3]
    try:
        async with aiofiles.open(file_path, "wb") as buffer: [cite: 3]
            content = await file.read() [cite: 3]
            await buffer.write(content) [cite: 3]
    except Exception as e:
        return {"error": f"Failed to save file: {e}"}, 500
    return {"status": "success", "path": f"/replays/{safe_session_id}.webm"} [cite: 3]

async def verify_id_token(token: str): [cite: 3]
    if not token: return None [cite: 3]
    try:
        if fb_db: [cite: 3]
            decoded_token = auth.verify_id_token(token) [cite: 3]
            return decoded_token['uid'] [cite: 3]
        else:
            print("Firebase Admin not initialized, using mock UID for token.") [cite: 3]
            if token.startswith("mock_token_for_"): [cite: 3]
                return token.replace("mock_token_for_", "") [cite: 3]
            return "mock_dev_user" [cite: 3]
    except Exception as e:
        print(f"Token verification failed: {e}") [cite: 3]
        return None [cite: 3]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = None): [cite: 3]
    user_uid = None [cite: 3]
    if token: [cite: 3]
        user_uid = await verify_id_token(token) [cite: 3]
        if not user_uid: [cite: 3]
            await websocket.accept() [cite: 3]
            await websocket.send_json({"type": "error", "text": "Invalid authentication token."}) [cite: 3]
            await websocket.close(code=1008) [cite: 3]
            return [cite: 3]

    await manager.connect(websocket, user_uid) [cite: 3]
    conn_data = manager.get_connection_data(websocket) [cite: 3]

    if not all([speech_client_v2, PROJECT_ID, GEMINI_API_KEY, moderation_model, pitch_eval_model, analysis_model]): [cite: 3]
        await websocket.send_json({"type": "error", "text": "Server setup incomplete. A required model or client is missing."}); await websocket.close(code=1011); return [cite: 3]
    if not conn_data or not conn_data.get("investor_chats") or not any(conn_data.get("investor_chats")): [cite: 3]
        await websocket.send_json({"type": "error", "text": "Server error: Investor personas not initialized correctly."}); await websocket.close(code=1011); return [cite: 3]

    recognizer_path = f"projects/{PROJECT_ID}/locations/us-central1/recognizers/_" [cite: 3]

    try:
        while True: [cite: 3]
            data = await websocket.receive() [cite: 3]
            if 'text' in data: [cite: 3]
                message = json.loads(data['text']) [cite: 3]
                msg_type = message.get("type") [cite: 3]
                current_session_id_from_conn = conn_data.get("current_session_id") [cite: 3]

                if msg_type == "startup_details": [cite: 3]
                    manager.reset_session_state(websocket) [cite: 3]
                    conn_data['startup_details'] = message.get("data") [cite: 3]
                    conn_data['mode'] = message.get("data", {}).get("mode", "strict") [cite: 3]

                    if conn_data['mode'] == 'drill': [cite: 3]
                        investor_name = manager.get_next_investor(websocket) [cite: 3]
                        if not investor_name: [cite: 3]
                            await websocket.send_json({"type": "error", "text": "No available investors for drill mode."}); continue [cite: 3]
                        chat_session = conn_data["investor_chats"][investor_name] [cite: 3]
                        details = conn_data["startup_details"] [cite: 3]
                        prompt = f"Brief: {details.get('name')} - {details.get('pitch')}. Problem: {details.get('problem')}. Ask your first, single, incisive question to the founder. Do not add any pleasantries. Just ask the question." [cite: 3]
                        response = await run_in_threadpool(chat_session.send_message, prompt) [cite: 3]
                        await websocket.send_json({"type": "investor", "investor_name": investor_name, "text": response.text.strip()}) [cite: 3]

                elif msg_type == "process_interim_transcript": [cite: 3]
                    audio_to_process = conn_data.get('latest_audio_blob') [cite: 3]
                    transcribed_text = "" [cite: 3]

                    if audio_to_process: [cite: 3]
                        decoding_config = cloud_speech.ExplicitDecodingConfig(encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.WEBM_OPUS, sample_rate_hertz=48000, audio_channel_count=1) [cite: 3]
                        features_config = cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True) [cite: 3]
                        request_config = cloud_speech.RecognitionConfig(explicit_decoding_config=decoding_config, language_codes=["en-US"], model="chirp", features=features_config) [cite: 3]
                        request_v2 = cloud_speech.RecognizeRequest(recognizer=recognizer_path, config=request_config, content=audio_to_process) [cite: 3]
                        try:
                            response_v2 = await run_in_threadpool(speech_client_v2.recognize, request=request_v2) [cite: 3]
                            if response_v2.results and response_v2.results[0].alternatives: [cite: 3]
                                transcribed_text = response_v2.results[0].alternatives[0].transcript.strip() [cite: 3]
                        except Exception as e:
                            print(f"Transcription error: {e}") [cite: 3]
                    
                    conn_data['latest_audio_blob'] = None [cite: 3]
                    # BUG FIX: Always send a response to unlock the frontend, even if transcription is empty.
                    await websocket.send_json({"type": "user_interim_transcript", "text": transcribed_text}) [cite: 3]

                elif msg_type == "send_composed_text": [cite: 3]
                    composed_text = message.get("text", "").strip() [cite: 3]
                    if not composed_text: [cite: 3]
                        composed_text = "[Silent Response]" [cite: 3]
                    
                    # This message is for the frontend to log the final user text.
                    await websocket.send_json({"type": "user", "text": composed_text}) [cite: 3]
                    
                    is_safe, reason = await check_for_inappropriate_content(composed_text, moderation_model) [cite: 3]
                    if not is_safe: [cite: 3]
                        await websocket.send_json({"type": "session_terminated", "text": "This is a waste of time. The meeting is over.", "reason": reason}) [cite: 3]
                        manager.reset_session_state(websocket); continue [cite: 3]

                    if conn_data['mode'] == 'strict' and not conn_data.get('opening_evaluated'): [cite: 3]
                        conn_data['opening_evaluated'] = True [cite: 3]
                        decision, reason = await evaluate_pitch_opening(composed_text, pitch_eval_model) [cite: 3]
                        if decision == 'TERMINATE': [cite: 3]
                            await websocket.send_json({"type": "session_terminated", "text": "I don't understand what you do. If you can't explain it clearly, there's no point in continuing. Meeting's over.", "reason": reason}) [cite: 3]
                            manager.reset_session_state(websocket); continue [cite: 3]
                    
                    investor_name = manager.get_next_investor(websocket) [cite: 3]
                    if not investor_name or not conn_data["investor_chats"].get(investor_name): [cite: 3]
                        await websocket.send_json({"type": "error", "text": "No available investors to respond."}); continue [cite: 3]

                    chat_session = conn_data["investor_chats"][investor_name] [cite: 3]
                    investor_response = await run_in_threadpool(chat_session.send_message, composed_text) [cite: 3]
                    raw_response_text = investor_response.text.strip() [cite: 3]

                    if raw_response_text.startswith("[TERMINATE_SESSION]"): [cite: 3]
                        final_text_to_send = raw_response_text.replace("[TERMINATE_SESSION]", "").strip() [cite: 3]
                        await websocket.send_json({"type": "investor", "investor_name": investor_name, "text": final_text_to_send}) [cite: 3]
                        await websocket.send_json({"type": "session_terminated", "text": "The investor has ended the meeting.", "reason": "This pitch was a waste of time."}) [cite: 3]
                        manager.reset_session_state(websocket); continue [cite: 3]
                    await websocket.send_json({"type": "investor", "investor_name": investor_name, "text": raw_response_text}) [cite: 3]

                elif msg_type == "end_session": [cite: 3]
                    session_user_uid = conn_data.get("user_uid") [cite: 3]
                    history = message.get("history", []) [cite: 3]
                    audio_path = message.get("audio_path", "") [cite: 3]
                    client_session_id = message.get("session_id", current_session_id_from_conn) [cite: 3]
                    current_mode = conn_data.get("mode", "unknown") [cite: 3]
                    startup_details_for_report = conn_data.get("startup_details", {}) [cite: 3]
                    report_data_to_save = {"timestamp": None, "mode": current_mode, "startup_details": startup_details_for_report, "replay_data": {"audio_path": audio_path, "transcript": history}} [cite: 3]
                    if current_mode == "strict": [cite: 3]
                        analyzer = PitchAnalyzer(history); analysis_report = await analyzer.analyze_pitch(); report_data_to_save["analysis_report"] = analysis_report [cite: 3]
                    else:
                        report_data_to_save["analysis_report"] = {"message": f"{current_mode.capitalize()} session completed.", "transcript": history} [cite: 3]
                    await websocket.send_json({"type": "analysis_report", "data": report_data_to_save}) [cite: 3]
                    if session_user_uid and fb_db: [cite: 3]
                        save_session_to_firestore(session_user_uid, client_session_id, report_data_to_save) [cite: 3]
                    else:
                        user_identifier_from_message = message.get("identifier", "unknown_user_local"); report_data_to_save["timestamp"] = datetime.now().isoformat(); save_session_to_local_file(user_identifier_from_message, report_data_to_save) [cite: 3]

                elif msg_type == "get_history": [cite: 3]
                    history_user_uid = conn_data.get("user_uid") [cite: 3]
                    history_data = get_history_from_firestore(history_user_uid) if history_user_uid and fb_db else get_history_from_local_file(message.get("identifier", "unknown_user_local")) [cite: 3]
                    await websocket.send_json({"type": "history_data", "data": history_data}) [cite: 3]

                elif msg_type == "user_timeout": [cite: 3]
                    investor_name = conn_data.get("last_investor_name") [cite: 3]
                    if not investor_name or not conn_data["investor_chats"].get(investor_name): continue [cite: 3]
                    chat_session = conn_data["investor_chats"][investor_name] [cite: 3]
                    prod_prompt = "The user was silent for a long time. Prod them to respond. Keep it short. Examples: 'Any thoughts?', 'Still there?'" [cite: 3]
                    response = await run_in_threadpool(chat_session.send_message, prod_prompt) [cite: 3]
                    await websocket.send_json({"type": "investor", "investor_name": investor_name, "text": response.text.strip()}) [cite: 3]

            elif 'bytes' in data: [cite: 3]
                audio_bytes = data['bytes'] [cite: 3]
                if not audio_bytes: continue [cite: 3]
                conn_data['latest_audio_blob'] = audio_bytes [cite: 3]

    except (WebSocketDisconnect, RuntimeError) as e: [cite: 3]
        if isinstance(e, WebSocketDisconnect): [cite: 3]
            print(f"Client {websocket.client} disconnected gracefully.") [cite: 3]
        else:
            print(f"Connection closed with runtime error: {e}") [cite: 3]
    except Exception as e:
        print(f"An unexpected error occurred in WebSocket: {e}"); traceback.print_exc() [cite: 3]
    finally:
        manager.disconnect(websocket) [cite: 3]
        print("INFO:     connection closed") [cite: 3]

# --- SERVE FRONTEND ---
# This MUST be the last thing added to the app
app.mount("/", StaticFiles(directory="static", html = True), name="static")

def run_server(): [cite: 3]
    print("--- SERVER READY FOR PRODUCTION DEPLOYMENT ---") [cite: 3]
    print("To run locally without ngrok: uvicorn main:app --reload") [cite: 3]

# This part is for local execution and might not be run on Render, but is good practice to keep.
if __name__ == "__main__":
    run_server()
