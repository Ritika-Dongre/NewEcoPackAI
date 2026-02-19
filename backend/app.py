from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json
from copy import deepcopy
from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

app = Flask(__name__)
CORS(app)

# =====================================================
# Load Model & Labels
# =====================================================
MODEL_PATH = "saved_model/my_model.keras"
LABEL_FILE = "saved_model/class_labels.txt"
CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.35"))
TOP_K = int(os.getenv("TOP_K_PREDICTIONS", "3"))

FEEDBACK_FILE = "feedback/feedback_log.jsonl"
PREDICTION_HISTORY_FILE = "history/prediction_history.jsonl"
MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ecopack_ai").strip()
VENDORS_COLLECTION = "vendors"
PREDICTIONS_COLLECTION = "prediction_history"
FEEDBACK_COLLECTION = "feedback"

VENDOR_CATALOG = [
    {
        "vendor_id": "eco_pack_hub",
        "name": "EcoPack Hub",
        "materials": ["recycled cardboard", "paper", "kraft", "mailer"],
        "service_regions": ["local", "regional", "long"],
        "min_order_qty": 100,
        "lead_time_days": 4,
        "rating": 4.6,
        "website": "https://example.com/ecopack-hub",
        "email": "sales@ecopackhub.example",
    },
    {
        "vendor_id": "green_wrap_supplies",
        "name": "GreenWrap Supplies",
        "materials": ["compostable", "biodegradable", "molded pulp", "paper tape"],
        "service_regions": ["regional", "long"],
        "min_order_qty": 200,
        "lead_time_days": 6,
        "rating": 4.7,
        "website": "https://example.com/greenwrap",
        "email": "hello@greenwrap.example",
    },
    {
        "vendor_id": "sustain_box_india",
        "name": "SustainBox India",
        "materials": ["corrugated", "double-wall", "recycled box", "rigid box"],
        "service_regions": ["local", "regional", "long"],
        "min_order_qty": 150,
        "lead_time_days": 5,
        "rating": 4.5,
        "website": "https://example.com/sustainbox",
        "email": "contact@sustainbox.example",
    },
]

# Global variables for model and labels
model = None
class_labels = []
mongo_client = None
mongo_db = None
use_mongo = False

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

def _serialize_mongo_doc(doc):
    if not doc:
        return doc
    out = dict(doc)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    return out

def init_mongo():
    global mongo_client, mongo_db, use_mongo
    if not MONGO_URI:
        print("MongoDB disabled: MONGO_URI not set.")
        use_mongo = False
        return
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        mongo_client.admin.command("ping")
        mongo_db = mongo_client[MONGO_DB_NAME]
        mongo_db[PREDICTIONS_COLLECTION].create_index([("timestamp_utc", ASCENDING)])
        mongo_db[FEEDBACK_COLLECTION].create_index([("timestamp_utc", ASCENDING)])
        mongo_db[VENDORS_COLLECTION].create_index([("vendor_id", ASCENDING)], unique=True)
        if mongo_db[VENDORS_COLLECTION].count_documents({}) == 0:
            mongo_db[VENDORS_COLLECTION].insert_many(VENDOR_CATALOG)
            print("Seeded default vendors into MongoDB.")
        use_mongo = True
        print(f"MongoDB connected: db={MONGO_DB_NAME}")
    except PyMongoError as e:
        print(f"MongoDB connection failed, fallback mode: {e}")
        mongo_client = None
        mongo_db = None
        use_mongo = False

def append_jsonl(path, payload):
    if use_mongo and mongo_db is not None:
        try:
            if path == PREDICTION_HISTORY_FILE:
                mongo_db[PREDICTIONS_COLLECTION].insert_one(payload)
                return
            if path == FEEDBACK_FILE:
                mongo_db[FEEDBACK_COLLECTION].insert_one(payload)
                return
        except PyMongoError as e:
            print(f"Mongo write failed, fallback to file: {e}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def read_jsonl(path):
    if use_mongo and mongo_db is not None:
        try:
            if path == PREDICTION_HISTORY_FILE:
                return [_serialize_mongo_doc(d) for d in mongo_db[PREDICTIONS_COLLECTION].find({})]
            if path == FEEDBACK_FILE:
                return [_serialize_mongo_doc(d) for d in mongo_db[FEEDBACK_COLLECTION].find({})]
        except PyMongoError as e:
            print(f"Mongo read failed, fallback to file: {e}")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def record_prediction(payload):
    append_jsonl(PREDICTION_HISTORY_FILE, payload)

def vendor_source():
    if use_mongo and mongo_db is not None:
        try:
            return [_serialize_mongo_doc(d) for d in mongo_db[VENDORS_COLLECTION].find({})]
        except PyMongoError:
            pass
    return VENDOR_CATALOG

# Try to load on startup
is_model_loaded = load_model_and_labels()
init_mongo()

# =====================================================
# Utility
# =====================================================
def normalize(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())

def clamp_score(value, min_value=1, max_value=10):
    return max(min_value, min(max_value, int(round(value))))

def keyword_points(text, keyword_map):
    text_lower = text.lower()
    score = 0
    for keyword, points in keyword_map.items():
        if keyword in text_lower:
            score += points
    return score

def parse_bool(value):
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def parse_float(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None

def parse_product_context(form_data):
    return {
        "weight_kg": parse_float(form_data.get("weight_kg")),
        "fragile": parse_bool(form_data.get("fragile")),
        "moisture_sensitive": parse_bool(form_data.get("moisture_sensitive")),
        "shipping_distance": (form_data.get("shipping_distance") or "local").strip().lower(),
        "budget_priority": (form_data.get("budget_priority") or "balanced").strip().lower(),
    }

def build_packaging_explanation(suggestion):
    sustainability_keywords = {
        "compostable": 4, "biodegradable": 4, "recycled": 3, "kraft": 2,
        "paper": 2, "cardboard": 2, "bamboo": 3, "cotton": 2, "jute": 3,
        "cloth": 1, "organic": 2, "plastic": -3, "poly": -2,
    }
    protection_keywords = {
        "rigid": 3, "thick": 3, "reinforced": 3, "protect": 2, "impact": 2,
        "shape": 2, "cushion": 2, "padding": 2, "dust": 1, "scratch": 2,
        "safe": 1, "compression": 2, "crushing": 2,
    }
    cost_keywords = {
        "lightweight": 3, "low-cost": 3, "paper": 2, "cardboard": 2,
        "mailer": 2, "recycled": 1, "bamboo": -1, "cotton": -1,
        "jute": -1, "rigid": -1,
    }
    explanation = {}
    for layer_name in ["internal", "external"]:
        layer = suggestion.get(layer_name, {})
        text = f"{layer.get('material', '')} {layer.get('reason', '')}"
        explanation[layer_name] = {
            "sustainability_score": clamp_score(5 + keyword_points(text, sustainability_keywords)),
            "protection_score": clamp_score(5 + keyword_points(text, protection_keywords)),
            "cost_efficiency_score": clamp_score(5 + keyword_points(text, cost_keywords)),
        }
    explanation["overall"] = {
        "sustainability_score": round((explanation["internal"]["sustainability_score"] + explanation["external"]["sustainability_score"]) / 2, 1),
        "protection_score": round((explanation["internal"]["protection_score"] + explanation["external"]["protection_score"]) / 2, 1),
        "cost_efficiency_score": round((explanation["internal"]["cost_efficiency_score"] + explanation["external"]["cost_efficiency_score"]) / 2, 1),
    }
    return explanation

def apply_contextual_adjustments(suggestion, context):
    adjusted = deepcopy(suggestion)
    notes = []
    if context.get("fragile") is True:
        if "padding" not in adjusted["internal"]["material"].lower() and "cushion" not in adjusted["internal"]["material"].lower():
            adjusted["internal"]["material"] = f"{adjusted['internal']['material']} + molded pulp cushioning"
        adjusted["internal"]["reason"] = f"{adjusted['internal']['reason']}; Added cushioning for fragile handling."
        notes.append("Added extra cushioning because product is marked fragile.")
    if context.get("weight_kg") is not None and context["weight_kg"] >= 2.5:
        adjusted["external"]["material"] = "Thick corrugated recycled cardboard box"
        adjusted["external"]["reason"] = f"{adjusted['external']['reason']}; Upgraded outer box for heavier shipment."
        notes.append("Upgraded external packaging for higher product weight.")
    if context.get("moisture_sensitive") is True:
        adjusted["external"]["material"] = f"{adjusted['external']['material']} + water-resistant recycled kraft layer"
        adjusted["external"]["reason"] = f"{adjusted['external']['reason']}; Added moisture barrier."
        notes.append("Added moisture-resistant outer layer.")
    if context.get("shipping_distance") == "long":
        adjusted["external"]["reason"] = f"{adjusted['external']['reason']}; Reinforced for long-distance transit."
        notes.append("Reinforced recommendation for long-distance shipping.")
    if context.get("budget_priority") == "low_cost" and context.get("fragile") is not True:
        adjusted["internal"]["material"] = "Recycled tissue paper wrap"
        adjusted["internal"]["reason"] = "Optimized for low cost while maintaining basic protection."
        notes.append("Applied low-cost packaging preference.")
    return adjusted, notes

def generate_packaging_options(base_suggestion, context):
    options = []
    options.append(("Balanced", deepcopy(base_suggestion)))
    eco = deepcopy(base_suggestion)
    eco["internal"]["material"] = "Compostable tissue + molded pulp support"
    eco["internal"]["reason"] = "Maximizes sustainability with biodegradable internal protection."
    eco["external"]["material"] = "FSC recycled cardboard box + paper tape"
    eco["external"]["reason"] = "High recyclability and lower plastic usage."
    options.append(("Eco First", eco))
    low = deepcopy(base_suggestion)
    low["internal"]["material"] = "Recycled paper wrap"
    low["internal"]["reason"] = "Cost-optimized with basic safe handling."
    low["external"]["material"] = "Standard recycled cardboard mailer"
    low["external"]["reason"] = "Affordable outer layer for routine shipping."
    options.append(("Low Cost", low))
    protect = deepcopy(base_suggestion)
    protect["internal"]["material"] = "Molded pulp cushioning + cotton wrap"
    protect["internal"]["reason"] = "Extra shock and scratch protection."
    protect["external"]["material"] = "Double-wall corrugated recycled box"
    protect["external"]["reason"] = "Higher structural strength during transit."
    options.append(("High Protection", protect))

    preferred = (context.get("budget_priority") or "balanced").lower()
    preferred_name = {"balanced": "Balanced", "eco_first": "Eco First", "low_cost": "Low Cost"}.get(preferred, "Balanced")
    payload = []
    for name, opt in options:
        scores = build_packaging_explanation(opt)["overall"]
        payload.append({
            "option_name": name,
            "is_recommended": name == preferred_name,
            "packaging_suggestion": opt,
            "scores": scores,
            "estimated_cost_inr": 48.0 if name == "Balanced" else (42.0 if name == "Low Cost" else (58.0 if name == "Eco First" else 64.0)),
            "estimated_co2_g": 135.0 if name == "Balanced" else (145.0 if name == "Low Cost" else (112.0 if name == "Eco First" else 158.0)),
        })
    return payload

def match_vendors_for_suggestion(suggestion, context, limit=4):
    material_text = f"{suggestion.get('internal', {}).get('material', '')} {suggestion.get('external', {}).get('material', '')}".lower()
    region = (context.get("shipping_distance") or "local").lower()
    ranked = []
    for vendor in vendor_source():
        score = 0.0
        matched = []
        for keyword in vendor.get("materials", []):
            if keyword in material_text:
                score += 2.0
                matched.append(keyword)
        if region in vendor.get("service_regions", []):
            score += 1.5
        score += float(vendor.get("rating", 0)) * 0.3
        if score > 0:
            ranked.append({
                "vendor_id": vendor.get("vendor_id"),
                "name": vendor.get("name"),
                "rating": vendor.get("rating"),
                "min_order_qty": vendor.get("min_order_qty"),
                "lead_time_days": vendor.get("lead_time_days"),
                "service_regions": vendor.get("service_regions", []),
                "matched_materials": matched,
                "website": vendor.get("website", ""),
                "email": vendor.get("email", ""),
                "score": round(score, 2),
            })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:limit]


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

    if model is None or not class_labels:
        if not load_model_and_labels():
            return jsonify({"error": "Model not loaded. Please train the model first by running train_model.py"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty file"}), 400

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)
    context = parse_product_context(request.form)

    try:
        img = preprocess(path)
        preds = model.predict(img, verbose=0)[0]

        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = class_labels[idx]
        top_k = min(TOP_K, len(class_labels))
        top_indices = np.argsort(preds)[::-1][:top_k]
        top_predictions = [
            {"label": class_labels[int(i)], "confidence": round(float(preds[int(i)]), 4)}
            for i in top_indices
        ]

        if confidence < CONFIDENCE_THRESHOLD:
            record_prediction({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "uploaded_file": file.filename,
                "product_type": "Uncertain",
                "prediction_accuracy": round(confidence, 3),
                "top_predictions": top_predictions,
                "product_context": context,
            })
            os.remove(path)
            return jsonify({
                "product_type": "Uncertain",
                "prediction_accuracy": round(confidence, 3),
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "top_predictions": top_predictions,
                "packaging_suggestion": "No reliable prediction",
                "vendor_options": [],
            })
    except Exception as e:
        os.remove(path)
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

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

    final_suggestion, adjustment_notes = apply_contextual_adjustments(suggestion, context)
    packaging_explanation = build_packaging_explanation(final_suggestion)
    packaging_options = generate_packaging_options(final_suggestion, context)
    vendor_options = match_vendors_for_suggestion(final_suggestion, context)

    response_payload = {
        "product_type": label,
        "prediction_accuracy": round(confidence, 3),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "top_predictions": top_predictions,
        "product_context": context,
        "packaging_suggestion": final_suggestion,
        "packaging_explanation": packaging_explanation,
        "adjustment_notes": adjustment_notes,
        "packaging_options": packaging_options,
        "vendor_options": vendor_options,
    }
    record_prediction({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "uploaded_file": file.filename,
        "product_type": label,
        "prediction_accuracy": round(confidence, 3),
        "top_predictions": top_predictions,
        "product_context": context,
    })

    os.remove(path)
    return jsonify(response_payload)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    predicted_label = data.get("predicted_label")
    if not predicted_label:
        return jsonify({"error": "predicted_label is required"}), 400
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "predicted_label": predicted_label,
        "prediction_accuracy": data.get("prediction_accuracy"),
        "correct_label": data.get("correct_label"),
        "note": data.get("note"),
        "uploaded_file": data.get("uploaded_file"),
        "top_predictions": data.get("top_predictions", []),
    }
    try:
        append_jsonl(FEEDBACK_FILE, payload)
        return jsonify({"message": "Feedback saved successfully"}), 201
    except Exception as e:
        return jsonify({"error": f"Failed to save feedback: {str(e)}"}), 500

@app.route("/history", methods=["GET"])
def history():
    limit_raw = request.args.get("limit", "10")
    try:
        limit = max(1, min(100, int(limit_raw)))
    except ValueError:
        limit = 10
    entries = read_jsonl(PREDICTION_HISTORY_FILE)
    entries = sorted(entries, key=lambda x: x.get("timestamp_utc", ""), reverse=True)[:limit]
    return jsonify({"count": len(entries), "items": entries})

@app.route("/history/summary", methods=["GET"])
def history_summary():
    entries = read_jsonl(PREDICTION_HISTORY_FILE)
    total = len(entries)
    class_counts = {}
    uncertain_count = 0
    avg_conf = 0.0
    for e in entries:
        label = e.get("product_type", "Unknown")
        class_counts[label] = class_counts.get(label, 0) + 1
        if label == "Uncertain":
            uncertain_count += 1
        try:
            avg_conf += float(e.get("prediction_accuracy", 0.0))
        except (TypeError, ValueError):
            pass
    avg_conf = round(avg_conf / total, 4) if total else 0.0
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return jsonify({
        "total_predictions": total,
        "uncertain_predictions": uncertain_count,
        "average_confidence": avg_conf,
        "top_classes": [{"label": k, "count": v} for k, v in top_classes],
    })

@app.route("/vendors", methods=["GET"])
def vendors():
    material = (request.args.get("material") or "").strip().lower()
    region = (request.args.get("region") or "").strip().lower()
    filtered = []
    for vendor in vendor_source():
        if material and not any(material in m for m in vendor.get("materials", [])):
            continue
        if region and region not in vendor.get("service_regions", []):
            continue
        filtered.append(_serialize_mongo_doc(vendor))
    filtered = sorted(filtered, key=lambda x: x.get("rating", 0), reverse=True)
    return jsonify({"count": len(filtered), "items": filtered})

@app.route("/vendors", methods=["POST"])
def add_vendor():
    data = request.get_json(silent=True) or {}
    required = ["vendor_id", "name", "materials", "service_regions"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400
    try:
        payload = {
            "vendor_id": str(data["vendor_id"]).strip().lower(),
            "name": str(data["name"]).strip(),
            "materials": [str(x).strip().lower() for x in data.get("materials", []) if str(x).strip()],
            "service_regions": [str(x).strip().lower() for x in data.get("service_regions", []) if str(x).strip()],
            "min_order_qty": int(data.get("min_order_qty", 100)),
            "lead_time_days": int(data.get("lead_time_days", 5)),
            "rating": float(data.get("rating", 4.0)),
            "website": str(data.get("website", "")).strip(),
            "email": str(data.get("email", "")).strip(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if use_mongo and mongo_db is not None:
            mongo_db[VENDORS_COLLECTION].update_one(
                {"vendor_id": payload["vendor_id"]},
                {"$set": payload},
                upsert=True,
            )
            return jsonify({"message": "Vendor upserted in MongoDB", "vendor": payload}), 201
        existing = next((v for v in VENDOR_CATALOG if v.get("vendor_id") == payload["vendor_id"]), None)
        if existing:
            existing.update(payload)
        else:
            VENDOR_CATALOG.append(payload)
        return jsonify({"message": "Vendor upserted in memory", "vendor": payload}), 201
    except (PyMongoError, ValueError) as e:
        return jsonify({"error": f"Failed to save vendor: {str(e)}"}), 500

@app.route("/health/db", methods=["GET"])
def db_health():
    if not use_mongo or mongo_db is None:
        return jsonify({
            "mongo_enabled": False,
            "status": "fallback_mode",
            "message": "Using file/static storage. Set MONGO_URI to enable MongoDB.",
        })
    try:
        mongo_client.admin.command("ping")
        return jsonify({"mongo_enabled": True, "status": "ok", "database": MONGO_DB_NAME})
    except PyMongoError as e:
        return jsonify({"mongo_enabled": True, "status": "error", "message": str(e)}), 500


# =====================================================
# Run Server
# =====================================================
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
