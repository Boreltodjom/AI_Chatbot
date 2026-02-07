from flask import Flask, request, jsonify, render_template, session, g
import os, uuid, logging, requests, base64, io, re, time, secrets, tempfile
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Legacy support
SECRET_KEY = os.getenv("SECRET_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower().strip()  # groq | openrouter | demo
ALLOW_DEMO_FALLBACK = os.getenv("ALLOW_DEMO_FALLBACK", "true").lower() in ("1", "true", "yes", "on")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")  # optional but recommended by OpenRouter
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")  # optional but recommended by OpenRouter
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE")  # optional but recommended by OpenRouter

DOVV_SYSTEM_PROMPT = """
You are "DOVV Assistant", the dedicated virtual employee for **DOVV Distribution Sarl**, Cameroon's premier supermarket chain.

**Your Mission:**
To provide a "perfect" customer experience by being extremely knowledgeable, polite, and helpful (in both French and English).
You must help convert inquiries into sales ("seal the deal") by conducting yourself as a proud DOVV representative.

**Core Knowledge Base:**

1.  **Identity & History:**
    *   **Founded:** August 21, 2003, by Mr. Philippe Tagne Noubissi.
    *   **First Store:** Mokolo, Yaoundé.
    *   **Headquarters:** Bastos, Yaoundé.
    *   **Motto/Values:** "La référence" (The Reference). We fight against the high cost of living ("Vie chère"), prioritize Hygiene, Quality, and Proximity.

2.  **Locations (Yaoundé):**
    *   **Bastos (HQ):** Upscale, wide variety, easy parking.
    *   **Mokolo:** The historic first store, bustling market area.
    *   **Marché Central:** Heart of the city.
    *   **Essos:** Serving the Essos/Benoue neighborhood.
    *   **Tongolo:** Northern exit route.
    *   **Elig Essono:** Near the Total station.
    *   **Mimboman:** East side of town.
    *   **Mendong:** Student and residential area.
    *   **Simbock:** Featuring "Cash & Carry" for wholesale/bulk buying.
    *   *(Note: If asked about Douala or other cities, say we are expanding soon but currently focused on Yaoundé).*

3.  **Services & Departments:**
    *   **Supermarket:** Groceries, fresh produce, frozen foods.
    *   **Bakery & Pastry (Boulangerie/Pâtisserie):** Fresh bread daily, croissants, birthday cakes, wedding cakes. *Highlight this!*
    *   **Butchery & Fish (Boucherie/Poissonnerie):** Fresh meat (beef, pork, chicken) and wide fish selection.
    *   **Cosmetics & Perfumery:** Beauty products, lotions, perfumes.
    *   **Liquor Store (Cave à Vins):** Extensive collection of wines, champagnes, and spirits.
    *   **Local Products:** Proudly selling "Made in Cameroon" (Ndolé, dried fish, spices, tapioca, etc.).

4.  **Recruitment (Job Application Process):**
    *   **To Apply:** Deposit a physical folder (Dossier Physique) at the General Directorate (Bastos) or any DOVV Agency against a receipt.
    *   **Required Documents (Standard Dossier):**
        1.  Handwritten Request (Demande manuscrite non timbrée) addressed to the General Manager.
        2.  CV (Curriculum Vitae).
        3.  Photocopy of CNI (Valid ID card).
        4.  Photocopy of Highest Diploma/Certificates.
        5.  Criminal Record (Extrait de Casier Judiciaire n°3) < 3 months.
        6.  Medical Certificate (Certificat Médical) < 3 months.
        7.  Photos: 1 Half-photo (4x4) and 1 Full photo (Carte entière).
        8.  Location Plan (Plan de localisation).

5.  **Product & Price Examples (Indicative Market Prices - Always warn to check store):**
    *   *Wines:* Vin Blanc CUVEE du ROI (1L ~1200 FCFA), Vin Rouge CAPO MERLOT (~1500 FCFA), B&G St. Emilion (~15000 FCFA), Vins de table generally start ~1200-1500 FCFA.
    *   *Rice (Riz):* 
        - 1kg varies (~500-800 FCFA depending on brand/quality).
        - 25kg bags (Perfumed) ~17,500 - 18,500 FCFA.
        - 25kg bags (Standard) ~13,000 FCFA.
        - *Brands:* Riz Mémé Cassé, Riz Parfumé (check daily stock).
    *   *Oil (Huile Végétale):*
        - Mayor/Diamaor 1L bottle ~1,500 - 1,600 FCFA.
        - 5L Bidon ~7,500+ FCFA.
    *   *Bakery:* Baguette (Regulated price ~150 FCFA), Croissants (~200-300 FCFA).
    *   *Water:* Tangui/Superment (starts ~300-400 FCFA).

6.  **Key Policies:**
    *   **Delivery:** "We do not currently have a central online delivery app. However, for large orders, please visit your nearest branch manager to discuss arrangements. Third-party delivery apps like Sasayez may list us, but buying in-store ensures the best prices and freshness."
    *   **Payment:** Cash, Orange Money, MTN Mobile Money, Visa/Mastercard (in major branches like Bastos).
    *   **Returns:** "Please check your receipt and product condition immediately at the counter. Perishable goods generally cannot be returned for hygiene reasons."

**Operational Guidelines:**
*   **Language:** STRICTLY BILINGUAL.
    *   If user speaks English -> Reply in English.
    *   If user speaks French -> Reply in French.
    *   If user speaks Camfranglais -> Reply in a friendly, understandable French/English mix but keep it professional.
*   **Tone:** Warm, commercial, inviting. Use phrases like:
    *   (FR) "Passer au magasin, c'est encore mieux !" / "Nous serions ravis de vous accueillir à Bastos."
    *   (EN) "Visiting the store is even better!" / "We'd love to welcome you at our Mokolo branch."
*   **Handling "Unknowns":** NEVER invent a price or policy. If unsure, say:
    *   "That's a great question! Prices change to give you the best deal. Please check our Facebook page or visit the store directly to confirm."

**Closing:**
Always end with an invitation to visit DOVV. "See you soon at DOVV!" or "À très bientôt chez DOVV !"
"""

# Flask app setup
app = Flask(__name__)
if not SECRET_KEY:
    # Safe default for local/dev; MUST be set in production (prevents predictable cookies).
    SECRET_KEY = secrets.token_urlsafe(32)
app.secret_key = SECRET_KEY
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5MB default
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Conversation memory ---
conversations = {}

# --- Simple in-memory rate limiting (good enough for a demo; use Redis in prod) ---
_RATE = {
    "chat": {},   # key -> [timestamps]
    "clear": {},
}
CHAT_LIMIT_COUNT = int(os.getenv("CHAT_LIMIT_COUNT", "20"))
CHAT_LIMIT_WINDOW_S = int(os.getenv("CHAT_LIMIT_WINDOW_S", "60"))
CLEAR_LIMIT_COUNT = int(os.getenv("CLEAR_LIMIT_COUNT", "10"))
CLEAR_LIMIT_WINDOW_S = int(os.getenv("CLEAR_LIMIT_WINDOW_S", "60"))

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

def _client_key():
    # Prefer X-Forwarded-For when behind a proxy; otherwise remote_addr.
    xff = request.headers.get("X-Forwarded-For", "")
    ip = (xff.split(",")[0].strip() if xff else request.remote_addr) or "unknown"
    sid = session.get("session_id", "no-session")
    return f"{ip}:{sid}"

def _rate_limit(bucket: str, max_count: int, window_s: int) -> bool:
    now = time.time()
    key = _client_key()
    arr = _RATE[bucket].setdefault(key, [])
    cutoff = now - window_s
    # prune
    i = 0
    while i < len(arr) and arr[i] < cutoff:
        i += 1
    if i:
        del arr[:i]
    if len(arr) >= max_count:
        return False
    arr.append(now)
    return True

def _ensure_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]

def _check_csrf():
    # Only protect state-changing routes; requires same-origin + token.
    sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
    return bool(sent) and secrets.compare_digest(sent, session.get("csrf_token", ""))

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
    # Do NOT prepend decorative characters here; they can confuse downstream markdown rendering.
    return text

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

# --- Demo provider (no external API needed) ---
def ask_demo(prompt: str, has_image: bool = False) -> str:
    prompt = (prompt or "").strip()
    if has_image and not prompt:
        return "I received the image. What would you like me to analyze about it?"
    if not prompt and has_image:
        return "Got the image. Ask me a question about it (objects, text, summary, etc.)."
    if not prompt:
        return "Please type a message or upload an image."
    # Simple, investor-safe response that still looks helpful.
    return (
        "Demo mode is enabled (no external AI provider). Here’s a quick structured reply:\n\n"
        f"- **You said**: {prompt}\n"
        "- **Next best step**: Tell me your goal (e.g., summary, email draft, pitch bullets) and your audience.\n"
        "- **If you want**: Paste more context and I’ll format it cleanly."
    )

# --- Send to LLM (Groq or OpenRouter) WITH CONTEXT ---
def ask_llm_with_context(session_id, prompt, image_path=None):
    provider = LLM_PROVIDER
    api_key = OPENROUTER_API_KEY if provider == "openrouter" else GROQ_API_KEY
    
    if not api_key:
        if ALLOW_DEMO_FALLBACK:
            return ask_demo(prompt, has_image=bool(image_path))
        return f"⚠️ {provider.upper()}_API_KEY not configured."

    # Determine URL and headers
    if provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_HTTP_REFERER or "http://localhost:5000",
            "X-Title": OPENROUTER_X_TITLE or "DOVV Assistant",
        }
        # OpenRouter often uses "openai/gpt-3.5-turbo" or others. Default to a good cheap one or let user set env.
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free") # Example default
    else:
        # Default to Groq
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    try:
        # Get conversation history for this session
        history = get_conversation_history(session_id)
        
        # OpenRouter specific: check if model supports vision if image provided (omit for now, assume text)
        if image_path:
             # Basic handling: just warn if not supported or implemented
             pass 

        messages = [{"role": "system", "content": DOVV_SYSTEM_PROMPT}]
        for turn in history:
            if turn.get('user') and turn.get('assistant'):
                messages.append({"role": "user", "content": turn['user']})
                messages.append({"role": "assistant", "content": turn['assistant']})
        
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        logging.info(f"Sending request to {provider} with {len(messages)} messages")

        response = requests.post(url, headers=headers, json=data, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            return format_ai_response(answer)
        else:
            error_code = response.status_code
            logging.error(f"{provider} API error ({error_code}): {response.text}")
            if ALLOW_DEMO_FALLBACK and error_code in (401, 429, 500, 502, 503):
                return ask_demo(prompt, has_image=False)
            return f"❌ Error communicating with {provider}."
            
    except Exception as e:
        logging.error(f"Request exception: {e}", exc_info=True)
        if ALLOW_DEMO_FALLBACK:
             return ask_demo(prompt, has_image=False)
        return f"❌ Error: Please try again later."

@app.before_request
def _security_before_request():
    # Secure session cookie settings (effective when served over HTTPS).
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "true").lower() in ("1", "true", "yes", "on"),
    )
    # CSRF protection for state-changing endpoints.
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.path in ("/chat", "/clear"):
        if not _check_csrf():
            return jsonify({"response": "❌ Security check failed (CSRF). Please refresh the page and try again."}), 403
    # Rate limiting.
    if request.path == "/chat":
        if not _rate_limit("chat", CHAT_LIMIT_COUNT, CHAT_LIMIT_WINDOW_S):
            return jsonify({"response": "⚠️ Too many requests. Please wait a moment and try again."}), 429
    if request.path == "/clear":
        if not _rate_limit("clear", CLEAR_LIMIT_COUNT, CLEAR_LIMIT_WINDOW_S):
            return jsonify({"status": "rate_limited"}), 429

@app.after_request
def _security_headers(resp):
    # Security headers. CSP uses a nonce to allow the inline <style> and <script> in the template.
    nonce = getattr(g, "csp_nonce", None)
    script_src = ["'self'", "https://cdn.jsdelivr.net"]
    style_src = ["'self'"]
    if nonce:
        script_src.append(f"'nonce-{nonce}'")
        style_src.append(f"'nonce-{nonce}'")
    else:
        # fallback (should not happen in our template route)
        script_src.append("'unsafe-inline'")
        style_src.append("'unsafe-inline'")

    csp = (
        "default-src 'self'; "
        f"script-src {' '.join(script_src)}; "
        f"style-src {' '.join(style_src)}; "
        "img-src 'self' data:; "
        "connect-src 'self' https://cdn.jsdelivr.net; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "object-src 'none'; "
        "upgrade-insecure-requests"
    )
    resp.headers["Content-Security-Policy"] = csp
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
    return resp

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
    csrf = _ensure_csrf_token()
    g.csp_nonce = secrets.token_urlsafe(16)
    return render_template("index.html", csrf_token=csrf, csp_nonce=g.csp_nonce)

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
        orig_name = secure_filename(file.filename or "")
        _, ext = os.path.splitext(orig_name.lower())
        if ext and ext not in ALLOWED_IMAGE_EXTS:
            return jsonify({"response": "⚠️ Unsupported image type. Please upload JPG/PNG/WEBP/GIF."})
        # Write to a temp file with a random name (prevents collisions & path tricks).
        tmp_dir = app.config["UPLOAD_FOLDER"]
        tmp_name = f"{uuid.uuid4().hex}{ext or '.jpg'}"
        filepath = os.path.join(tmp_dir, tmp_name)
        file.save(filepath)
        try:
            ai_response = ask_llm_with_context(session_id, message, filepath)
            add_to_conversation(session_id, f"[Image] {message}".strip(), ai_response)
            return jsonify({"response": ai_response})
        finally:
            try:
                os.remove(filepath)
            except Exception:
                pass

    if message:
        ai_response = ask_llm_with_context(session_id, message)
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
    return jsonify({
        "status": "healthy",
        "provider": LLM_PROVIDER,
        "groq_api_configured": bool(GROQ_API_KEY),
        "demo_fallback_enabled": ALLOW_DEMO_FALLBACK,
    })


@app.route("/ping")
def ping():
    """Lightweight endpoint for quick connectivity checks (no CSRF)."""
    return jsonify({"ok": True, "time": datetime.now().isoformat()})


@app.route("/test-groq")
def test_groq():
    """Test endpoint to verify Groq API key and connectivity (no CSRF)."""
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not set in .env", "ok": False}), 500
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Say 'hello' in one word only."}],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    try:
        logging.info("Testing Groq API...")
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers=headers, json=data, timeout=10)
        logging.info(f"Groq test response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')
            return jsonify({"ok": True, "status": 200, "groq_response": answer})
        else:
            return jsonify({"ok": False, "status": response.status_code, "error": response.text[:200]}), response.status_code
    except Exception as e:
        logging.error(f"Groq test failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes", "on")
    app.run(debug=debug, host="0.0.0.0", port=port)
