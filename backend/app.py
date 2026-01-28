import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image
import joblib
import io

app = Flask(__name__)

IMG_SIZE = 224
CONF_THRESHOLD = 0.60

model = load_model("model/mobilenetv3.keras")
classes = joblib.load("model/label_encoder.pkl")  # list string

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image).astype("float32")
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(io.BytesIO(request.files["image"].read()))
    img = preprocess_image(image)

    preds = model.predict(img)[0]
    conf = float(np.max(preds))
    idx = int(np.argmax(preds))

    if conf < CONF_THRESHOLD:
        return jsonify({
            "label": "Bukan Daun Jeruk",
            "confidence": conf,
            "status": "not_leaf"
        })

    label = classes[idx]

    status = "healthy" if "Healthy" in label else "disease"

    return jsonify({
        "label": label,
        "confidence": conf,
        "status": status
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
