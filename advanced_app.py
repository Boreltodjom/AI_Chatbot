from flask import Flask, request, jsonify, render_template, session
import os, uuid, logging, requests, base64, io, re
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret")

# Flask app setup
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Conversation memory ---
conversations = {}

def get_conversation_history(session_id):
    if session_id not in conversations:
        conversations[session_id] = []
    return conversations[session_id]

def add_to_conversation(session_id, user_message, ai_response):
    if session_id not in conversations:
        conversations[session_id] = []
    conversations[session_id].append({
        'user': user_message,
        'assistant': ai_response,
        'timestamp': datetime.now().isoformat()
    })
    # keep last 8 exchanges
    if len(conversations[session_id]) > 8:
        conversations[session_id] = conversations[session_id][-8:]

# --- Format AI response (make more human-like) ---
def format_ai_response(text):
    # improve readability with line breaks & lists
    text = re.sub(r'(\d+\.\s*)', r'\n\1', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
    text = text.strip()
    return f"✨ {text}"

# --- Encode image for vision model ---
def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Image encoding error: {e}")
        return None

# --- Send to OpenRouter WITH CONTEXT ---
def ask_openrouter_with_context(session_id, prompt, image_path=None):
    if not OPENROUTER_API_KEY:
        return "⚠️ API key not configured."

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    
    try:
        # Get conversation history for this session
        history = get_conversation_history(session_id)
        
        if image_path:  # Vision model request
            base64_image = encode_image_to_base64(image_path)
            # For vision, we typically send only the current prompt and image.
            # Some models support context, but it's complex. We'll keep it simple for now.
            # If you need context with images, we'd need to describe the history in text.
            effective_prompt = prompt or "What’s in this image?"
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": effective_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
            model = "anthropic/claude-3-haiku" # Or another vision-capable model
        else:  # Text model request with context
            messages = []
            # Add previous conversation history to the prompt
            for turn in history:
                 # Only add if both user and assistant parts exist and are not the initial intro
                if turn.get('user') and turn.get('assistant') and not (turn['user'] is None and "Hello! I’m your AI Assistant" in turn['assistant']):
                    messages.append({"role": "user", "content": turn['user']})
                    messages.append({"role": "assistant", "content": turn['assistant']})
            
            # Add the current user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Use a model suitable for conversation
            model = "mistralai/mistral-7b-instruct" # Or "openchat/openchat-7b" etc.

        data = {"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 800}
        logging.info(f"Sending request to {model} with messages: {messages}") # Log for debugging

        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, json=data, timeout=45)
        logging.info(f"Received response status: {response.status_code}") # Log for debugging
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            logging.info(f"AI Response: {answer}") # Log for debugging
            return format_ai_response(answer)
        else:
            error_text = response.text
            logging.error(f"API error ({response.status_code}): {error_text}")
            return f"❌ API error ({response.status_code}): {error_text}"
    except Exception as e:
        logging.error(f"Request exception: {e}", exc_info=True)
        return f"❌ Error: {str(e)}"

# --- Routes ---
@app.route("/")
def home():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        # Do NOT send intro yet - wait for language selection in frontend
        # Or, if you prefer the old behavior, uncomment the lines below and comment out the new ones
        # intro = (
        #     "👋 Hello! I’m your AI Assistant.\n\n"
        #     "Here’s what I can do:\n"
        #     "1. 💬 Chat with you naturally\n"
        #     "2. 📷 Analyze uploaded images and answer questions\n"
        #     "3. 🎙️ Accept voice input (on supported browsers)\n\n"
        #     "Ask me anything to get started!"
        # )
        # conversations[session['session_id']] = [{
        #     'user': None,
        #     'assistant': intro,
        #     'timestamp': datetime.now().isoformat()
        # }]
        conversations[session['session_id']] = [] # Start with empty history for language selection
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    message = request.form.get("message", "").strip()
    has_image = 'image' in request.files and request.files['image'].filename

    if not message and not has_image:
        return jsonify({"response": "⚠️ Please enter a message or upload an image."})

    if has_image:
        file = request.files['image']
        if not file.mimetype.startswith('image/'):
            return jsonify({"response": "⚠️ Please upload a valid image."})
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        # For image analysis, context is less straightforward, so we pass the prompt directly
        ai_response = ask_openrouter_with_context(session_id, message, filepath)
        add_to_conversation(session_id, f"[Image] {message}", ai_response)
        os.remove(filepath)
        return jsonify({"response": ai_response})

    if message:
        ai_response = ask_openrouter_with_context(session_id, message)
        add_to_conversation(session_id, message, ai_response)
        return jsonify({"response": ai_response})

    return jsonify({"response": "⚠️ Unexpected error."})

@app.route("/clear", methods=["POST"])
def clear_conversation():
    session_id = session.get('session_id')
    if session_id and session_id in conversations:
        del conversations[session_id]
    return jsonify({"status": "cleared"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "api_configured": bool(OPENROUTER_API_KEY)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
