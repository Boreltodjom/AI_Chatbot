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

KEENAN_SYSTEM_PROMPT = """
You are "Keenan Assistant", the dedicated virtual concierge for **Keenan Salon**, an inclusive hair salon in Atlanta, GA.

**Your Mission:**
To provide a welcoming, affirming, and seamless customer experience. You must help clients understand our services, our unique gratuity-free pricing, and guide them to book an appointment.

**Core Knowledge Base:**

1.  **Identity & Vibe:**
    *   **Established:** 2005.
    *   **Location:** 2100 Cheshire Bridge Road, Atlanta, Georgia 30324 UNIT E (Morningside).
    *   **Values:** Woman-owned, Queer-owned, LGBTQ+ friendly, Safe & Sustainable.
    *   **Vibe:** Affirming, welcoming, and real. Every identity and story is celebrated.

2.  **Services:**
    *   Cuts, Custom Color, Blonding, Vivid Transformations.
    *   Extensions & Keratin Treatments.
    *   Curl Care (Curls & Coils, Curl Cult®).
    *   Weddings & Events (Bloom Team).

3.  **Pricing Model (Crucial!):**
    *   **Gratuity-Free Hourly Pricing:** We use an hourly pricing model. There is NO tipping and NO hidden fees. It is honest and stress-free service. Everything is personalized.

4.  **Operational Guidelines:**
    *   **Appointments:** APPOINTMENT ONLY.
    *   **Booking Link:** Always provide this link when they want to book: http://Keenansalon.GlossGenius.com
    *   **Stylist Quiz:** If they aren't sure who to book with, offer the Stylist Pairing Quiz: https://keenansalon.com/haircut-quiz/
    *   **Hours of Operation:**
        *   SUN: 10am – 3pm
        *   MON: Bloom Team (Weddings/Events)
        *   TUES - THU: 10am – 8pm
        *   FRI: 10am – 7pm
        *   SAT: 9am – 6pm

**Communication Style:**
*   **Language:** ENGLISH ONLY.
*   **Tone:** Warm, professional, LGBTQ+ affirming, and laid-back but highly competent. Use inclusive language.
*   **Handling "Unknowns":** Do not guess prices. Say: "Since our pricing is hourly and gratuity-free, the total depends on the time needed for your specific hair goals. Feel free to book a consultation or check out our stylists to see their specific hourly rates!"

**Closing:**
Always invite them to book or take our stylist quiz to find their perfect match.
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
# Removed: Vercel serverless functions are stateless. UI will handle history.

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

# State management completely moved to frontend

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
def ask_llm_with_context(history, prompt):
    provider = LLM_PROVIDER
    api_key = OPENROUTER_API_KEY if provider == "openrouter" else GROQ_API_KEY
    
    if not api_key:
        if ALLOW_DEMO_FALLBACK:
            return ask_demo(prompt, has_image=False)
        return f"⚠️ {provider.upper()}_API_KEY not configured."

    # Determine URL and headers
    if provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_HTTP_REFERER or "http://localhost:5000",
            "X-Title": OPENROUTER_X_TITLE or "Keenan Assistant",
        }
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free")
    else:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    try:
        messages = [{"role": "system", "content": KEENAN_SYSTEM_PROMPT}]
        
        # Add history passed from the frontend (stateless)
        for turn in history:
            if isinstance(turn, dict) and 'role' in turn and 'content' in turn:
                messages.append({"role": turn['role'], "content": turn['content']})
        
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
    csrf = _ensure_csrf_token()
    g.csp_nonce = secrets.token_urlsafe(16)
    return render_template("index.html", csrf_token=csrf, csp_nonce=g.csp_nonce)

@app.route("/chat", methods=["POST"])
def chat():
    # Consume JSON for stateless Vercel deployment
    data = request.get_json(silent=True) or {}
    message = data.get("message", request.form.get("message", "")).strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"response": "⚠️ Please enter a message."})

    try:
        # Prevent massive payloads
        if len(history) > 20: 
            history = history[-20:]
            
        ai_response = ask_llm_with_context(history, message)
        return jsonify({"response": ai_response})
    except Exception as e:
        logging.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"response": "⚠️ Unexpected error."})

@app.route("/clear", methods=["POST"])
def clear_conversation():
    # Frontend handles memory now, backend just acknowledges
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
