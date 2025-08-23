from flask_cors import CORS
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ==============================
# Load model & labels
# ==============================
model = tf.keras.models.load_model("best_model.keras")

with open("class_labels.txt", "r") as f:
    CLASS_LABELS = [line.strip() for line in f.readlines()]

# ==============================
# Packaging Suggestions for all 32 categories
# ==============================
suggestions = {
    "Biscuits": {
        "internal": {
            "material": "certified compostable cellulose film",
            "reason": "Maintains freshness and is a food-safe, biodegradable alternative to plastic. It is an ideal inner liner for a durable external container."
        },
        "external": {
            "material": "reusable tin or glass jar",
            "reason": "Highly durable and infinitely recyclable. Encourages reuse by the consumer for other food items or household storage."
        }
    },
    "Bracelet": {
        "internal": {
            "material": "organic cotton or linen pouch",
            "reason": "A soft, reusable, and biodegradable option that protects the item from scratches and adds a premium feel. Can be repurposed by the consumer."
        },
        "external": {
            "material": "recycled cardboard box",-
            "reason": "Provides a rigid, protective structure for shipping and is easily recyclable. Can be customized with eco-friendly, water-based inks."
        }
    },
    "Bulb": {
        "internal": {
            "material": "molded paper pulp inserts",
            "reason": "Provides precise, secure cushioning for fragile bulbs and is made from recycled paper. Biodegradable and easily compostable."
        },
        "external": {
            "material": "FSC-certified cardboard box",
            "reason": "A simple, strong, and widely recyclable material that protects the product during transit. FSC certification ensures responsible sourcing."
        }
    },
    "Calculators": {
        "internal": {
            "material": "molded pulp tray",
            "reason": "A custom-fit, protective tray that is fully biodegradable and recyclable, eliminating the need for plastic inserts."
        },
        "external": {
            "material": "compact, minimalist recycled cardboard box",
            "reason": "Reduces overall material use and shipping weight. It’s easily recyclable and visually clean."
        }
    },
    "camera": {
        "internal": {
            "material": "mushroom packaging (MycoFoam)",
            "reason": "A highly protective, biodegradable alternative to Styrofoam. It's custom-moldable to fit the camera body and offers excellent shock absorption."
        },
        "external": {
            "material": "recycled corrugated cardboard box",
            "reason": "A strong and lightweight external package that is easily recyclable and widely accepted in recycling programs."
        }
    },
    "chocolates": {
        "internal": {
            "material": "compostable cellulose or foil wrapper",
            "reason": "Keeps chocolates fresh and protected while being biodegradable and reducing plastic waste."
        },
        "external": {
            "material": "paperboard carton with water-based ink",
            "reason": "A lightweight, recyclable external container that can be designed to hold multiple chocolates and is easily compostable."
        }
    },
    "clothing_fashion": {
        "internal": {
            "material": "glassine paper or acid-free tissue paper",
            "reason": "A translucent, biodegradable paper that protects the garment from moisture and dust and provides a premium unboxing experience."
        },
        "external": {
            "material": "compostable poly mailer or recycled kraft paper bag",
            "reason": "Both are lightweight and durable shipping solutions. The compostable mailer is a great alternative to plastic, while the paper bag is easily recyclable."
        }
    },
    "cosmetic_bottle": {
        "internal": {
            "material": "shredded kraft paper",
            "reason": "Acts as excellent void fill and cushioning, preventing the bottle from breaking during transit. It's fully biodegradable and recyclable."
        },
        "external": {
            "material": "recycled glass bottle with aluminum cap",
            "reason": "The primary product container is highly and infinitely recyclable. This minimizes plastic waste at the source and encourages a circular economy."
        }
    },
    "cosmetic_tube": {
        "internal": {
            "material": "cardboard divider or pulp insert",
            "reason": "Keeps tubes separate and secure within the box. Made from recycled paper, it's a great example of smart, minimalist, and biodegradable packaging."
        },
        "external": {
            "material": "aluminum tube with a plastic cap",
            "reason": "Aluminum is a highly recyclable material and a great alternative to plastic for tubes. The plastic cap can be separated and recycled as well."
        }
    },
    "Drawing_colors": {
        "internal": {
            "material": "molded paper pulp tray",
            "reason": "A custom-fit, biodegradable tray that holds colors securely, replacing plastic trays."
        },
        "external": {
            "material": "recycled cardboard box",
            "reason": "A sturdy outer box for protection, easily recyclable."
        }
    },
    "Drones": {
        "internal": {
            "material": "molded bamboo fiber or hemp fiber inserts",
            "reason": "Provides a rigid and shock-absorbent cradle for the drone and its parts. These materials are fast-growing, biodegradable, and strong."
        },
        "external": {
            "material": "corrugated cardboard suitcase-style box",
            "reason": "Provides a durable, reusable, and recyclable solution that gives a high-tech feel while being eco-friendly."
        }
    },
    "DryFruits": {
        "internal": {
            "material": "compostable stand-up pouch with a bio-based zipper",
            "reason": "Ensures freshness and is made from certified compostable materials, making it a sustainable alternative to a traditional plastic pouch."
        },
        "external": {
            "material": "reusable glass jar with a metal lid",
            "reason": "A durable, airtight, and highly recyclable container that can be refilled or reused indefinitely by the consumer."
        }
    },
    "Flow_Wrap_chocolates_Toffees": {
        "internal": {
            "material": "compostable cellulose film",
            "reason": "The primary wrapper is biodegradable, ensuring that the most commonly discarded part of the packaging is eco-friendly."
        },
        "external": {
            "material": "reusable tin or paperboard carton",
            "reason": "A durable and reusable or recyclable outer container that holds the twist-wrapped toffees."
        }
    },
    "Footwear": {
        "internal": {
            "material": "paper shoe horns and crumpled kraft paper",
            "reason": "A minimalist approach that provides necessary support and protection without plastic. The materials are fully recyclable and biodegradable."
        },
        "external": {
            "material": "recycled cardboard shoe box",
            "reason": "The standard in shoe packaging, it's protective, widely recyclable, and can be easily made from 100% recycled content."
        }
    },
    "Fountain_pen": {
        "internal": {
            "material": "paper-based felt or pulp tray",
            "reason": "A luxurious yet biodegradable insert that holds the pen securely. Adds a premium feel while remaining fully recyclable."
        },
        "external": {
            "material": "small, hinged recycled cardboard box",
            "reason": "A protective and elegant outer box that is easily recycled. Can be designed with a minimalist aesthetic and water-based ink."
        }
    },
    "Handbag_&luggage": {
        "internal": {
            "material": "recycled tissue paper and compostable stuffing",
            "reason": "Used to maintain the shape of the bag and is a fully biodegradable and recyclable alternative to traditional plastic stuffing and foam."
        },
        "external": {
            "material": "recycled and unbleached cardboard box",
            "reason": "Provides a strong, protective container for shipping. Its unbleached nature reduces the chemical footprint of production."
        }
    },
    "Headphones": {
        "internal": {
            "material": "molded pulp insert or shredded cardboard",
            "reason": "Custom-designed to fit the headphones and cables, providing excellent cushioning. Both materials are widely recyclable and biodegradable."
        },
        "external": {
            "material": "compact, recycled cardboard box",
            "reason": "Reduces the amount of packaging material and is lightweight, which also lowers shipping emissions. It's fully recyclable."
        }
    },
    "Home_decor": {
        "internal": {
            "material": "starch-based packing peanuts and kraft paper",
            "reason": "Packing peanuts dissolve in water and are fully biodegradable, providing excellent void fill. The kraft paper provides a soft, protective wrap."
        },
        "external": {
            "material": "heavy-duty corrugated cardboard box",
            "reason": "A durable, strong, and recyclable material that can withstand the weight and bulk of home decor items during shipping."
        }
    },
    "Input_devices": {
        "internal": {
            "material": "molded bamboo or hemp pulp",
            "reason": "Creates a perfect, biodegradable cradle for items like keyboards and mice, providing excellent shock protection while being eco-friendly."
        },
        "external": {
            "material": "compact, minimalist recycled cardboard box",
            "reason": "Reduces overall material use and shipping weight. It’s easily recyclable and visually clean."
        }
    },
    "Lamp": {
        "internal": {
            "material": "molded pulp inserts",
            "reason": "Provides a perfect custom fit for the lamp's base and shade, ensuring stability and protection. The material is fully biodegradable."
        },
        "external": {
            "material": "double-walled corrugated cardboard box",
            "reason": "The extra strength of the box provides superior protection for fragile items while remaining a fully recyclable material."
        }
    },
    "Laptops": {
        "internal": {
            "material": "mushroom packaging (MycoFoam)",
            "reason": "Custom-molded to fit the laptop, providing superior shock absorption. It's a premium, biodegradable alternative to Styrofoam."
        },
        "external": {
            "material": "corrugated cardboard box with FSC certification",
            "reason": "Ensures the cardboard is sourced from responsibly managed forests. The box is durable and universally recyclable."
        }
    },
    "Mobile_phones": {
        "internal": {
            "material": "pulp or bamboo fiber tray",
            "reason": "A minimalist, biodegradable insert that holds the phone securely within the box, eliminating the need for plastic trays."
        },
        "external": {
            "material": "compact, recycled cardboard box",
            "reason": "A small form factor reduces material use and shipping emissions, and the box is fully recyclable."
        }
    },
    "Necklace": {
        "internal": {
            "material": "recycled paper card or cotton pad",
            "reason": "Secures the necklace to prevent tangling. It is a simple, biodegradable, and recyclable solution."
        },
        "external": {
            "material": "small recycled cardboard box with linen ribbon",
            "reason": "An elegant and recyclable box that provides protection. The linen ribbon is a natural, biodegradable alternative to plastic ribbons."
        }
    },
    "Notebook": {
        "internal": {
            "material": "kraft paper belly band",
            "reason": "A simple, recyclable, and biodegradable way to group multiple notebooks or add branding without needing a full-coverage wrap."
        },
        "external": {
            "material": "corrugated mailer envelope",
            "reason": "Protective, lightweight, and made from a high percentage of recycled paper, ideal for shipping single or a few notebooks."
        }
    },
    "Remotes": {
        "internal": {
            "material": "molded pulp insert",
            "reason": "Custom-fit to hold the remote and any accessories, providing secure transport. The material is fully biodegradable and made from recycled paper."
        },
        "external": {
            "material": "small, unbleached recycled cardboard box",
            "reason": "A simple and protective box that is easily recyclable and requires less processing, reducing its environmental footprint."
        }
    },
    "Ring": {
        "internal": {
            "material": "hemp or organic cotton pouch",
            "reason": "A soft, reusable, and biodegradable pouch that protects the ring. It is a premium, eco-friendly alternative to foam inserts."
        },
        "external": {
            "material": "small wooden box or recycled cardboard box",
            "reason": "Wooden boxes can be repurposed and are biodegradable, while the cardboard box is a classic, recyclable choice for secure shipping."
        }
    },
    "Router": {
        "internal": {
            "material": "molded hemp or bamboo fiber tray",
            "reason": "Provides rigid, shock-absorbent cushioning for the router and its power adapters. These are rapidly renewable and biodegradable resources."
        },
        "external": {
            "material": "recycled corrugated cardboard box",
            "reason": "A standard, protective, and fully recyclable external container."
        }
    },
    "Snacks": {
        "internal": {
            "material": "certified compostable stand-up pouch",
            "reason": "Ensures freshness and is a great alternative to traditional plastic pouches for single-serve items. The pouch is designed to be composted after use."
        },
        "external": {
            "material": "corrugated cardboard display box",
            "reason": "Provides a secondary layer of protection and is ideal for retail display. It is fully recyclable."
        }
    },
    "Small_stationary": {
        "internal": {
            "material": "recycled tissue paper or paper twine",
            "reason": "Used for bundling and wrapping, this is a minimalist, fully recyclable and biodegradable way to organize small items like clips or erasers."
        },
        "external": {
            "material": "small cardboard box with integrated dividers",
            "reason": "A smart, minimalist, and fully recyclable box that organizes small items without needing extra internal plastic trays."
        }
    },
    "Telephone": {
        "internal": {
            "material": "molded pulp inserts",
            "reason": "Secures the phone base and handset, providing excellent shock absorption. Fully biodegradable and made from recycled content."
        },
        "external": {
            "material": "recycled corrugated cardboard box",
            "reason": "A sturdy and protective box that is easily recyclable."
        }
    },
    "Twist_Wrap_Toffees": {
        "internal": {
            "material": "compostable cellulose twist wrap",
            "reason": "The primary wrapper is biodegradable, ensuring that the most commonly discarded part of the packaging is eco-friendly."
        },
        "external": {
            "material": "reusable tin or paperboard carton",
            "reason": "A durable and reusable or recyclable outer container that holds the twist-wrapped toffees."
        }
    },
    "Watches": {
        "internal": {
            "material": "recycled paper pulp watch holder",
            "reason": "A custom-fit, protective holder that is fully recyclable and biodegradable, replacing plastic inserts."
        },
        "external": {
            "material": "recycled cardboard box or wooden box",
            "reason": "Both options provide a premium, durable outer layer. The cardboard box is easily recyclable, while the wooden box is reusable and biodegradable."
        }
    }
}

# ==============================
# Preprocess image for model input
# ==============================
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB").resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# ==============================
# API endpoint for classification
# ==============================
@app.route("/classify", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img_file = request.files["image"]
        image = Image.open(img_file)
        processed = preprocess_image(image)

        prediction = model.predict(processed)[0]
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        confidence_display = "Unknown" if np.isnan(confidence) else f"{round(confidence * 100, 2)}%"

        packaging_info = suggestions.get(predicted_class, {
            "internal": {"material": "N/A", "reason": "No packaging suggestion available"},
            "external": {"material": "N/A", "reason": "No packaging suggestion available"}
        })

        return jsonify({
            "product_type": predicted_class,
            "prediction_accuracy": confidence_display,
            "packaging_suggestion": packaging_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# Run Flask
# ==============================
if __name__ == "__main__":
    app.run(debug=True)





'''
from flask_cors import CORS
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ==============================
# Load model & labels
# ==============================
model = tf.keras.models.load_model("best_model.keras")

with open("class_labels.txt", "r") as f:
    CLASS_LABELS = [line.strip() for line in f.readlines()]

# ==============================
# Packaging Suggestions
# ==============================
suggestions = {
    # ------------------- Existing categories -------------------
    'books_stationery': {
        'internal': {
            'material': 'recycled kraft paper',
            'reason': 'Eco-friendly cushioning for books, pens, notebooks and other stationery; reduces waste and is recyclable.'
        },
        'external': {
            'material': 'corrugated envelopes',
            'reason': 'Protective and lightweight, suitable for most stationery items, and made from recycled paper.'
        }
    },
    'clothing_fashion': {
        'internal': {
            'material': 'compostable polybags',
            'reason': 'Protects garments from dust/moisture while decomposing naturally in compost.'
        },
        'external': {
            'material': 'jute bags',
            'reason': 'Reusable and biodegradable material, perfect for shipping apparel sustainably.'
        }
    },
    'cosmetic_bottle': {
        'internal': {
            'material': 'paper mesh wrap',
            'reason': 'Provides shock absorption for fragile cosmetic bottles while being biodegradable.'
        },
        'external': {
            'material': 'compostable boxes',
            'reason': 'Sturdy and sustainable boxes ideal for cosmetics packaging.'
        }
    },
    'cosmetic_tube': {
        'internal': {
            'material': 'biodegradable bubble wrap',
            'reason': 'Protects small cosmetic tubes during transport while reducing plastic use.'
        },
        'external': {
            'material': 'eco-mailer',
            'reason': 'Lightweight, recyclable and eco-friendly for shipping compact items.'
        }
    },
    'home_decor': {
        'internal': {
            'material': 'shredded paper',
            'reason': 'Reusable and cushioning for vases, lamps, and other decor items.'
        },
        'external': {
            'material': 'recycled carton',
            'reason': 'Sturdy and recyclable box ideal for decorative and fragile items.'
        }
    },
    'jewelry': {
        'internal': {
            'material': 'paper pulp trays',
            'reason': 'Secure and molded compartments for rings, earrings, and more — fully recyclable.'
        },
        'external': {
            'material': 'reusable fabric boxes',
            'reason': 'Elegant and reusable boxes that reduce single-use waste.'
        }
    },
    'Biscuits': {
        'internal': {
            'material': 'kraft paper or compostable film',
            'reason': 'Keeps biscuits fresh while using recyclable materials.'
        },
        'external': {
            'material': 'recyclable cardboard box',
            'reason': 'Eco-friendly box suitable for shipping biscuit packs.'
        }
    },
    'DryFruits': {
        'internal': {
            'material': 'paper pouch / compostable bag',
            'reason': 'Protects nuts and dry fruits and is biodegradable.'
        },
        'external': {
            'material': 'cardboard box / paper bag / glass jar',
            'reason': 'Eco-friendly options for storing and shipping dry fruits.'
        }
    },
    'chocolates': {
        'internal': {
            'material': 'compostable foil wrap or paper-based wrap',
            'reason': 'Keeps chocolates fresh and protected while being biodegradable and reducing plastic waste.'
        },
        'external': {
            'material': 'recyclable cardboard box',
            'reason': 'Strong outer packaging that protects chocolates during shipping and is eco-friendly.'
        }
    },
    'Snacks': {
        'internal': {
            'material': 'compostable film or kraft paper pouch',
            'reason': 'Provides freshness and protection for chips, namkeen, and other snacks while reducing single-use plastic.'
        },
        'external': {
            'material': 'paperboard carton or recycled cardboard box',
            'reason': 'Sturdy and stackable, suitable for bulk snack shipments while being recyclable.'
        }
    },

    # ------------------- New categories -------------------
    'mobile_wearables': {
        'internal': {
            'material': 'molded paper pulp trays',
            'reason': 'Holds phones, smartwatches securely while avoiding plastic inserts.'
        },
        'external': {
            'material': 'corrugated recyclable boxes',
            'reason': 'Strong outer box for safe delivery of mobiles and wearables.'
        }
    },
    'computing_devices': {
        'internal': {
            'material': 'molded pulp & corrugated dividers',
            'reason': 'Protects laptops, desktops, and other electronics from impact.'
        },
        'external': {
            'material': 'reinforced corrugated cartons',
            'reason': 'Durable eco-friendly outer packaging for heavy devices.'
        }
    },
    'cameras_drones': {
        'internal': {
            'material': 'biodegradable molded inserts',
            'reason': 'Keeps drones/cameras stable and protected during shipping.'
        },
        'external': {
            'material': 'compostable corrugated boxes',
            'reason': 'Eco-friendly and durable for fragile tech.'
        }
    },
    'remotes_calculators': {
        'internal': {
            'material': 'compostable film wrap',
            'reason': 'Lightweight and eco alternative to plastic pouches.'
        },
        'external': {
            'material': 'paper padded mailers',
            'reason': 'Slim protective mailers reduce material use.'
        }
    },
    'input_devices': {
        'internal': {
            'material': 'corrugated inserts or paper wrap',
            'reason': 'Prevents scratches and movement of keyboards/mice.'
        },
        'external': {
            'material': 'recycled cardboard cartons',
            'reason': 'Protective and stackable for bulk logistics.'
        }
    },
    'power_speakers': {
        'internal': {
            'material': 'biodegradable pulp holders',
            'reason': 'Keeps power banks & speakers secure in place.'
        },
        'external': {
            'material': 'kraft paperboard boxes',
            'reason': 'Strong eco-friendly outer pack.'
        }
    }
}

# ==============================
# Preprocess image for model input
# ==============================
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB").resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# ==============================
# API endpoint for classification
# ==============================
@app.route("/classify", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img_file = request.files["image"]
        image = Image.open(img_file)
        processed = preprocess_image(image)

        prediction = model.predict(processed)[0]
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        confidence_display = "Unknown" if np.isnan(confidence) else f"{round(confidence * 100, 2)}%"

        # Safely fetch packaging info
        packaging_info = suggestions.get(predicted_class, {
            "internal": {"material": "N/A", "reason": "No packaging suggestion available"},
            "external": {"material": "N/A", "reason": "No packaging suggestion available"}
        })

        return jsonify({
            "product_type": predicted_class,
            "prediction_accuracy": confidence_display,
            "packaging_suggestion": packaging_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# Run Flask
# ==============================
if __name__ == "__main__":
    app.run(debug=True)

'''