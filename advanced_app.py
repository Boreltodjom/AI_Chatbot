from flask import Flask, request, jsonify, render_template, session
import os
import uuid
import logging
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests
from PIL import Image
import base64
import io
from datetime import datetime

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory conversation storage (per session)
conversations = {}

def get_conversation_history(session_id):
    """Get conversation history for a session"""
    if session_id not in conversations:
        conversations[session_id] = []
    return conversations[session_id]

def add_to_conversation(session_id, user_message, ai_response):
    """Add exchange to conversation history"""
    if session_id not in conversations:
        conversations[session_id] = []
    
    conversations[session_id].append({
        'user': user_message,
        'assistant': ai_response,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 10 exchanges to avoid token limits
    if len(conversations[session_id]) > 10:
        conversations[session_id] = conversations[session_id][-10:]

def encode_image_to_base64(image_path):
    """Convert image to base64 for API"""
    try:
        # Open and potentially resize image to avoid API limits
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Resize if too large (max 2048px on longest side)
            max_size = 2048
            if max(img.width, img.height) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr = img_byte_arr.getvalue()
            
            return base64.b64encode(img_byte_arr).decode('utf-8')
    except Exception as e:
        logging.error(f"Image encoding error: {e}")
        return None

def build_context_messages(session_id, current_message):
    """Build message context with conversation history"""
    messages = [
        {
            "role": "system", 
            "content": f"""You are a helpful AI assistant with access to current information up to 2024. 
            Today's date is {datetime.now().strftime('%Y-%m-%d')}. 
            You have memory of previous conversations in this session. 
            Be accurate, helpful, and conversational. 
            When discussing people, use current/recent information when available."""
        }
    ]
    
    # Add conversation history
    history = get_conversation_history(session_id)
    for exchange in history[-5:]:  # Last 5 exchanges for context
        messages.append({"role": "user", "content": exchange['user']})
        messages.append({"role": "assistant", "content": exchange['assistant']})
    
    # Add current message
    messages.append({"role": "user", "content": current_message})
    
    return messages

def ask_openrouter_with_context(session_id, prompt, image_path=None):
    """Send request to OpenRouter API with conversation context"""
    if not OPENROUTER_API_KEY:
        return "API key not configured."
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        if image_path:
            # For images, use vision model with simpler context
            base64_image = encode_image_to_base64(image_path)
            if not base64_image:
                return "Sorry, I couldn't process the image."
            
            # Build context for vision model
            context = ""
            history = get_conversation_history(session_id)
            if history:
                recent_context = []
                for exchange in history[-3:]:  # Last 3 exchanges for image context
                    if 'image' not in exchange['user'].lower():  # Only include non-image exchanges
                        recent_context.append(f"User: {exchange['user']}\nAssistant: {exchange['assistant']}")
                
                if recent_context:
                    context = "Previous conversation context:\n" + "\n\n".join(recent_context) + "\n\nCurrent question: "
            
            full_prompt = context + prompt
            
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
            
            model = "anthropic/claude-3-haiku"
            logging.info(f"Using vision model with context: {full_prompt[:100]}...")
            
        else:
            # For text, use conversation history
            messages = build_context_messages(session_id, prompt)
            model = "mistralai/mistral-7b-instruct"
            logging.info(f"Using text model with {len(messages)} context messages")
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=45  # Longer timeout for vision processing
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            logging.info(f"Got response: {answer[:100]}...")
            return answer
        else:
            logging.error(f"API error: {response.status_code} - {response.text[:200]}")
            
            # Try fallback for images
            if image_path and response.status_code == 400:
                return ask_openrouter_image_fallback(session_id, prompt)
            
            return f"Sorry, I got an error from the AI service (Status: {response.status_code}). Please try again."
            
    except requests.exceptions.Timeout:
        logging.error("Request timeout")
        return "Sorry, the request timed out. Please try again with a smaller image or simpler question."
    except Exception as e:
        logging.error(f"Request exception: {e}")
        return f"Sorry, something went wrong: {str(e)}"

def ask_openrouter_image_fallback(session_id, prompt):
    """Fallback for when vision model fails"""
    try:
        # Use text model to give a helpful response about image analysis failure
        messages = build_context_messages(session_id, 
            f"The user uploaded an image and asked: '{prompt}'. I couldn't analyze the image due to technical issues. Please provide a helpful response explaining this and suggesting alternatives.")
        
        data = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return "Sorry, I couldn't analyze the image. This might be due to image format, size, or temporary service issues. Please try with a different image or ask a text question."
            
    except Exception as e:
        logging.error(f"Fallback error: {e}")
        return "Sorry, I couldn't analyze the image. Please try with a different image or ask a text question."

@app.route("/")
def home():
    # Initialize session
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    
    logging.info(f"Chat request for session: {session_id[:8]}...")
    
    try:
        # Get message and image
        message = request.form.get("message", "").strip()
        has_image = 'image' in request.files and request.files['image'].filename
        
        logging.info(f"Message: '{message}', Has image: {has_image}")
        
        if not message and not has_image:
            return jsonify({"response": "Please enter a message or upload an image."})
        
        # Handle image upload
        if has_image:
            file = request.files['image']
            
            # Validate image
            if not file.mimetype.startswith('image/'):
                return jsonify({"response": "Please upload a valid image file (JPG, PNG, etc.)."})
            
            # Save image temporarily
            filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Set default prompt if none provided
                prompt = message if message else "What do you see in this image? Please describe it in detail."
                
                # Get AI response with image and context
                ai_response = ask_openrouter_with_context(session_id, prompt, image_path=filepath)
                
                # Add to conversation history
                user_message = f"[Image: {file.filename}] {prompt}"
                add_to_conversation(session_id, user_message, ai_response)
                
                return jsonify({"response": ai_response})
                
            finally:
                # Clean up
                try:
                    os.remove(filepath)
                except:
                    pass
        
        # Handle text-only message
        elif message:
            ai_response = ask_openrouter_with_context(session_id, message)
            
            # Add to conversation history
            add_to_conversation(session_id, message, ai_response)
            
            return jsonify({"response": ai_response})
        
        return jsonify({"response": "Please enter a message."})
        
    except Exception as e:
        logging.error(f"Chat route exception: {e}")
        return jsonify({"response": f"Sorry, something went wrong: {str(e)}"})

@app.route("/clear", methods=["POST"])
def clear_conversation():
    """Clear conversation history"""
    session_id = session.get('session_id')
    if session_id and session_id in conversations:
        del conversations[session_id]
    return jsonify({"status": "cleared"})

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy", 
        "api_configured": bool(OPENROUTER_API_KEY),
        "active_conversations": len(conversations)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"API Key loaded: {bool(OPENROUTER_API_KEY)}")
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    print("Starting advanced chatbot with memory and improved image processing...")
    print(f"Server will run on port: {port}")
    app.run(debug=False, host="0.0.0.0", port=port)