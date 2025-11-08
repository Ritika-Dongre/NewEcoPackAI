from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)

# =====================================================
# Load Model & Labels
# =====================================================
MODEL_PATH = "saved_model/my_model.keras"
LABEL_FILE = "saved_model/class_labels.txt"

# Global variables for model and labels
model = None
class_labels = []

def load_model_and_labels():
    global model, class_labels
    try:
        print("Loading model from:", MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
        
        print("Loading class labels from:", LABEL_FILE)
        with open(LABEL_FILE, "r", encoding="utf-8") as f:
            class_labels = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(class_labels)} class labels")
        return True
    except Exception as e:
        print(f"Error loading model or labels: {str(e)}")
        return False

# Try to load on startup
is_model_loaded = load_model_and_labels()

# =====================================================
# Utility
# =====================================================
def normalize(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


# =====================================================
# Auto-Generated Packaging Suggestions
# =====================================================
packaging_suggestions = {
    # --- Footwear ---
    "BalletFlat": {
        "internal": {"material": "Recycled tissue or cotton wrap", "reason": "Prevents scratches on footwear surface"},
        "external": {"material": "Thin recycled cardboard shoe box", "reason": "Eco-friendly and stackable"}
    },
    "FlipFlops": {
        "internal": {"material": "Recycled paper separator", "reason": "Avoids scuffing of soles"},
        "external": {"material": "Lightweight recycled mailer", "reason": "Minimizes material usage"}
    },
    "Sandals": {
        "internal": {"material": "Recycled tissue or paper wrap", "reason": "Protects straps"},
        "external": {"material": "Recycled cardboard box", "reason": "Compact and recyclable"}
    },
    "Shoes": {
        "internal": {"material": "Cloth wrap + paper stuffing", "reason": "Maintains shape during shipping"},
        "external": {"material": "Thick recycled cardboard box", "reason": "Provides strength for heavy footwear"}
    },
    "Sneakers": {
        "internal": {"material": "Soft cotton padding", "reason": "Prevents deformation"},
        "external": {"material": "Recycled shoe box", "reason": "Durable and sustainable"}
    },

    # --- Clothing ---
    "Blazer": {
        "internal": {"material": "Biodegradable garment cover", "reason": "Dust-free and eco-safe"},
        "external": {"material": "Recycled garment box", "reason": "Protects structure during transit"}
    },
    "Coat": {
        "internal": {"material": "Cotton or jute wrap", "reason": "Breathable protection"},
        "external": {"material": "Rigid recycled box", "reason": "Prevents compression"}
    },
    "Dress": {
        "internal": {"material": "Compostable tissue paper", "reason": "Prevents wrinkles"},
        "external": {"material": "Paper mailer bag", "reason": "Plastic-free shipping"}
    },
    "Gaun": {
        "internal": {"material": "Cotton garment bag", "reason": "Preserves fabric finish"},
        "external": {"material": "Recycled cardboard box", "reason": "Provides safe handling"}
    },
    "Hoddies": {
        "internal": {"material": "Recycled paper wrap", "reason": "Fold protection"},
        "external": {"material": "Compostable mailer", "reason": "Eco alternative to plastic"}
    },
    "Jacket": {
        "internal": {"material": "Biodegradable cover", "reason": "Dust-resistant"},
        "external": {"material": "Reinforced paper box", "reason": "Protection from impact"}
    },
    "JacketDenim": {
        "internal": {"material": "Tissue paper wrap", "reason": "Protects denim surface"},
        "external": {"material": "Recycled box", "reason": "Strong and reusable"}
    },
    "Long Sleeve": {
        "internal": {"material": "Recycled tissue wrap", "reason": "Prevents creases"},
        "external": {"material": "Paper envelope", "reason": "Low-cost and compostable"}
    },
    "Lower": {
        "internal": {"material": "Kraft paper sheet", "reason": "Separates folded layers"},
        "external": {"material": "Recycled poly-bag alternative", "reason": "Moisture resistant"}
    },
    "Pants": {
        "internal": {"material": "Recycled tissue paper", "reason": "Prevents fold marks"},
        "external": {"material": "Paper courier bag", "reason": "Sustainable packaging"}
    },
    "Saree": {
        "internal": {"material": "Cotton cloth wrap", "reason": "Prevents fabric snag"},
        "external": {"material": "Cardboard mailer", "reason": "Lightweight and protective"}
    },
    "Shirt": {
        "internal": {"material": "Recycled paper fold support", "reason": "Maintains shape"},
        "external": {"material": "Compostable courier bag", "reason": "Plastic-free shipment"}
    },
    "Shorts": {
        "internal": {"material": "Paper wrap", "reason": "Prevents folding damage"},
        "external": {"material": "Recycled envelope", "reason": "Low-impact packaging"}
    },
    "Skirt": {
        "internal": {"material": "Recycled tissue paper", "reason": "Keeps smooth texture"},
        "external": {"material": "Paper bag", "reason": "Eco-friendly dispatch"}
    },
    "Sweater": {
        "internal": {"material": "Cotton or jute wrap", "reason": "Prevents lint accumulation"},
        "external": {"material": "Compostable mailer", "reason": "Sustainable outer layer"}
    },
    "Tshirt": {
        "internal": {"material": "Tissue paper wrap", "reason": "Prevents wrinkles"},
        "external": {"material": "Recycled mailer", "reason": "Biodegradable and safe"}
    },

    # --- Accessories ---
    "Belt": {
        "internal": {"material": "Recycled kraft wrap", "reason": "Protects from scratches"},
        "external": {"material": "Slim recycled box", "reason": "Compact and protective"}
    },
    "Bracelet": {
        "internal": {"material": "Cotton pouch / jute wrap", "reason": "Scratch-free"},
        "external": {"material": "Recycled kraft box", "reason": "Durable and green"}
    },
    "DigitalWatch": {
        "internal": {"material": "Soft cotton pouch", "reason": "Prevents scratches"},
        "external": {"material": "Small recycled cardboard box", "reason": "Eco-friendly"}
    },
    "Earings": {
        "internal": {"material": "Recycled paper holder", "reason": "Keeps pair together"},
        "external": {"material": "Mini kraft box", "reason": "Biodegradable and secure"}
    },
    "HandbagLuggage": {
        "internal": {"material": "Cotton dust bag", "reason": "Scratch-free storage"},
        "external": {"material": "Rigid recycled box", "reason": "Protects shape during shipping"}
    },
    "Hats": {
        "internal": {"material": "Recycled paper stuffing", "reason": "Maintains shape"},
        "external": {"material": "Large recycled cardboard box", "reason": "Avoids crushing"}
    },
    "Necklace": {
        "internal": {"material": "Cotton pad / paper card", "reason": "Avoids tangling"},
        "external": {"material": "Small kraft box", "reason": "Eco-friendly & elegant"}
    },
    "Ring": {
        "internal": {"material": "Organic cotton pouch", "reason": "Soft & plastic-free"},
        "external": {"material": "Small recycled box", "reason": "Reusable & eco-safe"}
    },
    "WristWatch": {
        "internal": {"material": "Cotton cushion support", "reason": "Protects dial & straps"},
        "external": {"material": "Recycled cardboard / bamboo box", "reason": "Reusable & stylish"}
    }
}

# =====================================================
# Preprocess Function
# =====================================================
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    # MobileNetV2 preprocessing
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


# =====================================================
# Classification API
# =====================================================
@app.route("/classify", methods=["POST"])
def classify():
    global model, class_labels
    
    # Check if model is loaded
    if model is None or not class_labels:
        if not load_model_and_labels():
            return jsonify({
                "error": "Model not loaded. Please train the model first by running train_model.py"
            }), 503

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty file"}), 400

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    try:
        img = preprocess(path)
        preds = model.predict(img)

        idx = np.argmax(preds[0])
        confidence = float(np.max(preds))
        label = class_labels[idx]

        print(f"\n[ PREDICTION ] Label: {label} | Confidence: {confidence:.3f}")

        # Confidence threshold
        if confidence < 0.50:
            os.remove(path)
            return jsonify({
                "product_type": "Uncertain",
                "prediction_accuracy": round(confidence, 3),
                "packaging_suggestion": "No reliable prediction"
            })
    except Exception as e:
        os.remove(path)
        return jsonify({
            "error": f"Error during prediction: {str(e)}"
        }), 500

    # ==============================
    # Lookup with normalization
    # ==============================
    suggestion = packaging_suggestions.get(label)

    if suggestion is None:
        norm_label = normalize(label)
        for k, v in packaging_suggestions.items():
            if normalize(k) == norm_label:
                suggestion = v
                break

    if suggestion is None:
        suggestion = {
            "internal": {"material": "N/A", "reason": "No data available"},
            "external": {"material": "N/A", "reason": "No data available"}
        }

    os.remove(path)

    return jsonify({
        "product_type": label,
        "prediction_accuracy": round(confidence, 3),
        "packaging_suggestion": suggestion
    })


# =====================================================
# Run Server
# =====================================================
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
