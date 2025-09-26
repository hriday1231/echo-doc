import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import requests

# ---------- Setup ----------
load_dotenv()
app = Flask(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Put it in your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

# Prefer 2.5 (works with Continue); clean fallbacks.
MODEL_CANDIDATES = [
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini-2.5-pro",
    "gemini-1.5-flash-001",
]

def _first_model():
    last = None
    for name in MODEL_CANDIDATES:
        try:
            return genai.GenerativeModel(name)
        except Exception as e:
            last = e
            continue
    raise RuntimeError(f"Could not init any Gemini model: {last}")

GEMINI_MODEL = _first_model()

# Reviewer provider: "coderabbit" or "gemini" (default).
REVIEW_PROVIDER = os.getenv("REVIEW_PROVIDER", "gemini").lower()
CODERABBIT_API_KEY = os.getenv("CODERABBIT_API_KEY")

# ---------- Agents ----------
def generate_code(prompt_text: str) -> str:
    try:
        engineered = (
            "Based on the following request, write a single Python function. "
            "Do not include any explanation, introductory text, or markdown code fences like ```python. "
            "Only return the raw Python code for the function itself.\n\n"
            f"Request: {prompt_text}"
        )
        resp = GEMINI_MODEL.generate_content(engineered)
        text = getattr(resp, "text", "") or ""
        return text if text.strip() else "# Error: Gemini returned no text."
    except Exception as e:
        app.logger.error(f"Coder error: {e}")
        return f"# Error generating code with Gemini: {e}"

def _review_with_coderabbit(code_text: str) -> str:
    if not CODERABBIT_API_KEY:
        return "Review skipped: API key not configured."
    try:
        api_url = "https://api.coderabbit.ai/v1/review"  # ensure plain URL
        headers = {
            "Authorization": f"Bearer {CODERABBIT_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"language": "python", "code": code_text}
        r = requests.post(api_url, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("summary") or data.get("review") or "Review summary not available."
    except requests.exceptions.Timeout:
        return "Could not review code: The request to CodeRabbit timed out."
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if getattr(e, "response", None) else "N/A"
        body = e.response.text if getattr(e, "response", None) else ""
        app.logger.error(f"CodeRabbit API error status={status} body={body[:400]}")
        return f"Could not review code: An API error occurred. Status: {status}"
    except Exception as e:
        app.logger.error(f"CodeRabbit unexpected error: {e}")
        return "Could not review code: An unexpected error occurred."

def _review_with_gemini(code_text: str) -> str:
    try:
        prompt = (
            "You are a senior Python reviewer. Review the following function. "
            "Point out correctness bugs, edge cases, performance issues, security concerns, "
            "style/readability problems, and suggest concrete improvements. "
            "Keep it concise with bullet points.\n\n"
            f"```python\n{code_text}\n```"
        )
        resp = GEMINI_MODEL.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return text.strip() or "No feedback."
    except Exception as e:
        app.logger.error(f"Gemini review error: {e}")
        return "Review skipped due to an internal error."

def review_code(code_text: str) -> str:
    """Try configured provider; fall back to Gemini if needed."""
    if REVIEW_PROVIDER == "coderabbit":
        result = _review_with_coderabbit(code_text)
        # If provider failed (common: Status N/A), fall back automatically.
        if result.startswith("Could not review code"):
            backup = _review_with_gemini(code_text)
            return f"{result}\n\n---\nFallback (Gemini):\n{backup}"
        return result
    # default path
    return _review_with_gemini(code_text)

def generate_docs_with_gemini(code_text: str) -> str:
    if code_text.strip().startswith("# Error"):
        return "Documentation skipped due to an error in code generation."
    try:
        engineered = (
            "You are an expert technical writer. Based on the following Python code, "
            "write clear and concise documentation. Explain what the function does, its parameters (if any), "
            "and what it returns. Use Markdown formatting for your response.\n\n"
            f"Code:\n```python\n{code_text}\n```"
        )
        resp = GEMINI_MODEL.generate_content(engineered)
        text = getattr(resp, "text", "") or ""
        return text if text.strip() else "Error: Gemini returned no documentation."
    except Exception as e:
        return f"Could not generate documentation with Gemini: {e}"

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    # Expect audio via multipart/form-data { audio: Blob(webm/opus) }
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    mime = audio_file.mimetype or "application/octet-stream"
    if not mime.startswith("audio/"):
        return jsonify({"error": f"Unsupported file type: {mime}"}), 400

    try:
        app.logger.info("🎤 Transcribing audio with Gemini...")
        audio_part = {"mime_type": mime, "data": audio_file.read()}
        transcription_prompt = "Transcribe the following audio clearly and concisely."
        resp = GEMINI_MODEL.generate_content([transcription_prompt, audio_part])
        user_prompt = (getattr(resp, "text", "") or "").strip()
        if not user_prompt:
            raise ValueError("Transcription failed: Empty response from model.")
        app.logger.info(f"📝 Transcription: {user_prompt}")
    except Exception as e:
        app.logger.error(f"Audio transcription error: {e}")
        return jsonify({
            "error": (
                f"Failed to transcribe audio: {e}. "
                "Try setting GEMINI_MODEL=gemini-2.5-flash (or gemini-2.5-pro) in your .env. "
                "As a fallback, use gemini-1.5-flash-001."
            )
        }), 500

    generated_code = generate_code(user_prompt)
    code_review = review_code(generated_code)
    documentation = generate_docs_with_gemini(generated_code)
    return jsonify({"code": generated_code, "review": code_review, "docs": documentation})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0") 