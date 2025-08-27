from flask import Flask, request, jsonify
from flask_cors import CORS   # ✅ FIX: import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ---------------------------
# Flask App Setup
# ---------------------------
app = Flask(__name__)
CORS(app)  # ✅ Allow cross-origin (for React frontend)

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = "saved_model/my_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------
# Categories
# ---------------------------
class_labels = [
    # Jewelry
    "Ring", "Necklace", "Bracelet", "Earrings",
    # Clothing
    "Shirt", "TShirt", "Pants", "Lower", "Sweater", "Hoodies",
    "Shorts", "Skirt", "Dress", "Gown", "Jacket", "JacketDenim",
    "Blazer", "Coat", "Saree", "Kurti", "Scarf", "Legging",
    # Footwear
    "Boots", "Sneakers", "Formal_Shoes", "Sandals", "FlipFlops", "Ballet_Flat",
    # Watches
    "Wristwatch", "Digital_Watch",
    # Separate
    "Hat", "Belt"
]

# ---------------------------
# Packaging Suggestions
# ---------------------------
packaging_suggestions = {
    # Jewelry
    "Ring": {
        "internal": {"material": "Hemp or organic cotton pouch", "reason": "Reusable, soft and biodegradable"},
        "external": {"material": "Small wooden or recycled cardboard box", "reason": "Eco-friendly and repurposable"}
    },
    "Necklace": {
        "internal": {"material": "Recycled paper card or cotton pad", "reason": "Prevents tangling & biodegradable"},
        "external": {"material": "Small cardboard box with linen ribbon", "reason": "Elegant & recyclable"}
    },
    "Bracelet": {
        "internal": {"material": "Organic cotton pouch or jute wrap", "reason": "Protects bracelet without plastic"},
        "external": {"material": "Recycled kraft paper box", "reason": "Strong, protective & eco-friendly"}
    },
    "Earrings": {
        "internal": {"material": "Recycled paper holder", "reason": "Keeps pair together & avoids loss"},
        "external": {"material": "Small recycled cardboard box", "reason": "Minimal, sustainable & secure"}
    },

    # Footwear
    "casual_lightweight": {
        "internal": {"material": "Recycled tissue or cloth bag", "reason": "Light protection for casual footwear"},
        "external": {"material": "Thin recycled cardboard box", "reason": "Compact, eco-friendly & stackable"}
    },
    "premium_heavy": {
        "internal": {"material": "Recycled paper stuffing or cloth wrap", "reason": "Strong support for premium shoes"},
        "external": {"material": "Thick recycled cardboard shoe box", "reason": "Durable, protective & sustainable"}
    },

    # Clothing
    "FlatClothing": {
        "internal": {"material": "Recycled tissue paper", "reason": "Prevents wrinkles during fold packaging"},
        "external": {"material": "Compostable mailer or recycled polybag", "reason": "Durable and biodegradable"}
    },
    "HangerClothing": {
        "internal": {"material": "Biodegradable garment cover", "reason": "Keeps garment dust-free"},
        "external": {"material": "Recycled kraft garment box", "reason": "Provides safe shipping & eco-friendly"}
    },

    # Watches
    "Wristwatch": {
        "internal": {"material": "Organic cotton cushion", "reason": "Protects dial and strap"},
        "external": {"material": "Recycled cardboard or wooden box", "reason": "Durable, reusable & stylish"}
    },
    "Digital_Watch": {
        "internal": {"material": "Cotton pouch", "reason": "Simple, scratch-free eco option"},
        "external": {"material": "Small recycled cardboard box", "reason": "Affordable & recyclable"}
    },

    # Separate
    "Hat": {
        "internal": {"material": "Recycled paper stuffing", "reason": "Maintains hat shape"},
        "external": {"material": "Large recycled cardboard box", "reason": "Prevents crushing during shipping"}
    },
    "Belt": {
        "internal": {"material": "Recycled kraft paper wrap", "reason": "Prevents scratches on leather"},
        "external": {"material": "Slim recycled cardboard box", "reason": "Compact & recyclable"}
    }
}

# ---------------------------
# Groups Mapping
# ---------------------------
flat_clothing = ["Shirt", "TShirt", "Pants", "Lower", "Sweater", "Hoodies", "Shorts", "Skirt", "Dress", "Gown", "Kurti", "Legging"]
hanger_clothing = ["Jacket", "JacketDenim", "Blazer", "Coat", "Saree", "Scarf"]
casual_lightweight = ["Sandals", "FlipFlops", "Ballet_Flat", "Sneakers"]
premium_heavy = ["Boots", "Formal_Shoes"]

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # ✅ Match MobileNetV2 training preprocessing
    return img_array / 127.5 - 1.0

# ---------------------------
# API Endpoint
# ---------------------------
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:   # ✅ use "file" (consistent with React)
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    img_array = preprocess(filepath)
    preds = model.predict(img_array)
    predicted_class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    # Safe class label
    predicted_label = class_labels[predicted_class_idx] if predicted_class_idx < len(class_labels) else "Unknown"

    # Assign packaging type
    if predicted_label in flat_clothing:
        suggestion = packaging_suggestions["FlatClothing"]
    elif predicted_label in hanger_clothing:
        suggestion = packaging_suggestions["HangerClothing"]
    elif predicted_label in casual_lightweight:
        suggestion = packaging_suggestions["casual_lightweight"]
    elif predicted_label in premium_heavy:
        suggestion = packaging_suggestions["premium_heavy"]
    elif predicted_label in packaging_suggestions:
        suggestion = packaging_suggestions[predicted_label]
    else:
        suggestion = {
            "internal": {"material": "N/A", "reason": "No suggestion available"},
            "external": {"material": "N/A", "reason": "No suggestion available"}
        }

    return jsonify({
        "product_type": predicted_label,
        "prediction_accuracy": round(confidence, 2),
        "packaging_suggestion": suggestion
    })

# ---------------------------
# Run Server
# ---------------------------
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
