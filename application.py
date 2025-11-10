import os, traceback
from flask import Flask, request, jsonify, render_template_string
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "basic_classifier.pkl")
VECT_PATH  = os.path.join(HERE, "count_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

application = Flask(__name__)

# --- map model outputs to 0/1 per PRA spec: FAKE -> 1, REAL -> 0 ---
def to01(pred):
    """
    Accepts string/np.str_/int/bool and returns 0 or 1.
    Known labels mapped:
      'fake'->1, 'real'->0
      'positive'/'pos'/'yes'/'true'->1
      'negative'/'neg'/'no'/'false'->0
    Falls back to int() if already numeric-like.
    """
    s = str(pred).strip().lower()
    if s in ("fake", "1", "true", "yes", "positive", "pos"):
        return 1
    if s in ("real", "0", "false", "no", "negative", "neg"):
        return 0
    # last resort: try numeric cast
    try:
        return int(s)
    except Exception:
        raise ValueError(f"Unrecognized prediction label from model: {pred!r}")

@application.get("/")
def root():
    return jsonify({"status":"ok","message":"Sentiment API is running!"})

@application.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error":"No text provided"}), 400
        X = vectorizer.transform([text])
        raw = model.predict(X)[0]          # may be 'FAKE'/'REAL'
        y = to01(raw)                      # normalize to 0/1
        return jsonify({"prediction": y})
    except Exception:
        tb = traceback.format_exc()
        print("[/predict] ERROR\n" + tb)
        return jsonify({"error":"internal error","traceback":tb}), 500

DEMO_HTML = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>PRA5 Demo</title></head>
  <body style="font-family: system-ui, sans-serif; max-width: 720px; margin: 32px auto;">
    <h2>PRA5 Demo</h2>
    <form method="POST">
      <textarea name="text" rows="6" style="width:100%;" placeholder="Enter text...">{{ text|default('') }}</textarea><br><br>
      <button type="submit">Predict</button>
    </form>
    {% if pred is not none %}
      <p><strong>Prediction (0/1):</strong> {{ pred }}</p>
      {% if raw is not none %}
        <p style="color:#666">Raw model label: <code>{{ raw }}</code></p>
      {% endif %}
    {% endif %}
    {% if error %}
      <h3>Traceback</h3>
      <pre style="white-space:pre-wrap">{{ error }}</pre>
    {% endif %}
  </body>
</html>
"""

@application.route("/demo", methods=["GET","POST"])
def demo():
    try:
        pred, text, raw = None, "", None
        if request.method == "POST":
            text = (request.form.get("text") or "").strip()
            if text:
                X = vectorizer.transform([text])
                raw = model.predict(X)[0]
                pred = to01(raw)
        return render_template_string(DEMO_HTML, pred=pred, text=text, raw=raw, error=None)
    except Exception:
        tb = traceback.format_exc()
        print("[/demo] ERROR\n" + tb)
        return render_template_string(DEMO_HTML, pred=None, text="", raw=None, error=tb), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))  # default to 5001 to avoid AirPlay
    print(f"[BOOT] Starting on http://127.0.0.1:{port}")
    application.run(host="0.0.0.0", port=port, debug=True)
