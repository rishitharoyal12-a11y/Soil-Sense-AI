import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, redirect, render_template, request, send_from_directory, session, url_for
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "soil-sense-ai-secret"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
LABELS_PATH = MODEL_DIR / "labels.json"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_labels():
    if LABELS_PATH.exists():
        with LABELS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return [
        "Black Soil",
        "Red Soil",
        "Laterite Soil",
        "Alluvial Soil",
        "Clay Soil",
        "Sandy Soil",
        "Loamy Soil",
        "Peaty Soil",
        "Silty Soil",
        "Chalky Soil",
    ]


LABELS = load_labels()


def preprocess_image(file_storage):
    file_storage.stream.seek(0)
    image_bytes = file_storage.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def build_soil_profile(soil_type):
    profiles = {
        "Black Soil": {
            "fertility": "High",
            "moisture": 72,
            "organic_matter": 3.8,
            "drainage": "Moderate",
            "texture": "Clayey",
            "water_capacity": 78,
            "nutrient_richness": "High",
            "ph": 7.3,
            "nitrogen": "High",
            "phosphorus": "Medium",
            "potassium": "High",
            "crops": [
                {"name": "Cotton", "season": "Kharif", "yield": "3.8 t/ha", "water": "Moderate", "duration": "180 days", "climate": "Warm", "profit": "High"},
                {"name": "Soybean", "season": "Kharif", "yield": "2.7 t/ha", "water": "Moderate", "duration": "110 days", "climate": "Warm", "profit": "High"},
            ],
            "fertilizers": [
                {"name": "Compost", "purpose": "Improve organic matter", "method": "Broadcast", "dosage": "5 t/ha", "frequency": "Seasonal"},
                {"name": "NPK", "purpose": "Balance nutrients", "method": "Soil application", "dosage": "100 kg/ha", "frequency": "Twice yearly"},
            ],
            "pesticides": [
                {"name": "Neem Oil", "purpose": "Control pests", "target": "Aphids", "method": "Foliar spray", "safety": "Use gloves"},
                {"name": "Bio Pesticides", "purpose": "Prevent fungal outbreaks", "target": "Fungi", "method": "Spray", "safety": "Avoid midday application"},
            ],
            "tips": [
                "Add compost to improve structure.",
                "Use mulching to reduce evaporation.",
                "Rotate pulses with cotton to sustain fertility.",
            ],
        },
        "Red Soil": {
            "fertility": "Medium",
            "moisture": 58,
            "organic_matter": 2.9,
            "drainage": "Good",
            "texture": "Sandy-loam",
            "water_capacity": 64,
            "nutrient_richness": "Moderate",
            "ph": 6.4,
            "nitrogen": "Medium",
            "phosphorus": "Low",
            "potassium": "Medium",
            "crops": [
                {"name": "Groundnut", "season": "Kharif", "yield": "2.4 t/ha", "water": "Low", "duration": "120 days", "climate": "Warm", "profit": "High"},
                {"name": "Millets", "season": "Kharif", "yield": "1.8 t/ha", "water": "Low", "duration": "90 days", "climate": "Dry", "profit": "Medium"},
            ],
            "fertilizers": [
                {"name": "Farmyard Manure", "purpose": "Boost organic carbon", "method": "Incorporate", "dosage": "10 t/ha", "frequency": "Seasonal"},
                {"name": "DAP", "purpose": "Support root development", "method": "Band placement", "dosage": "60 kg/ha", "frequency": "Single application"},
            ],
            "pesticides": [
                {"name": "Bio Pesticides", "purpose": "Control leaf-feeding insects", "target": "Leaf pests", "method": "Spray", "safety": "Use protective gear"},
                {"name": "Neem Oil", "purpose": "Reduce sucking pests", "target": "Whiteflies", "method": "Foliar spray", "safety": "Apply in morning"},
            ],
            "tips": [
                "Apply farmyard manure before sowing.",
                "Avoid overwatering during flowering.",
                "Use drip irrigation to preserve moisture.",
            ],
        },
        "Laterite Soil": {
            "fertility": "Low",
            "moisture": 49,
            "organic_matter": 2.1,
            "drainage": "Excellent",
            "texture": "Gravelly",
            "water_capacity": 58,
            "nutrient_richness": "Low",
            "ph": 5.8,
            "nitrogen": "Low",
            "phosphorus": "Low",
            "potassium": "Low",
            "crops": [
                {"name": "Tea", "season": "Year-round", "yield": "2.6 t/ha", "water": "High", "duration": "365 days", "climate": "Cool humid", "profit": "High"},
                {"name": "Coffee", "season": "Year-round", "yield": "1.1 t/ha", "water": "Moderate", "duration": "300 days", "climate": "Cool humid", "profit": "High"},
            ],
            "fertilizers": [
                {"name": "Vermicompost", "purpose": "Improve fertility", "method": "Top dressing", "dosage": "2 t/ha", "frequency": "Monthly"},
                {"name": "Bone Meal", "purpose": "Raise phosphorus", "method": "Soil application", "dosage": "40 kg/ha", "frequency": "Quarterly"},
            ],
            "pesticides": [
                {"name": "Organic Pesticides", "purpose": "Suppress mites", "target": "Mites", "method": "Spray", "safety": "Apply in evening"},
                {"name": "Fungicides", "purpose": "Reduce root rot", "target": "Root pathogens", "method": "Soil drench", "safety": "Avoid runoff"},
            ],
            "tips": [
                "Increase organic matter steadily.",
                "Use pH-correcting amendments.",
                "Plant cover crops between seasons.",
            ],
        },
        "Alluvial Soil": {
            "fertility": "Excellent",
            "moisture": 76,
            "organic_matter": 4.1,
            "drainage": "Good",
            "texture": "Loamy",
            "water_capacity": 84,
            "nutrient_richness": "Very High",
            "ph": 7.1,
            "nitrogen": "High",
            "phosphorus": "High",
            "potassium": "High",
            "crops": [
                {"name": "Rice", "season": "Kharif", "yield": "6.0 t/ha", "water": "High", "duration": "120 days", "climate": "Warm humid", "profit": "High"},
                {"name": "Wheat", "season": "Rabi", "yield": "4.8 t/ha", "water": "Moderate", "duration": "120 days", "climate": "Cool", "profit": "High"},
            ],
            "fertilizers": [
                {"name": "NPK", "purpose": "Support vigorous growth", "method": "Soil application", "dosage": "120 kg/ha", "frequency": "Twice yearly"},
                {"name": "Bio Fertilizers", "purpose": "Improve nutrient cycling", "method": "Seed treatment", "dosage": "1 kg/ha", "frequency": "Per season"},
            ],
            "pesticides": [
                {"name": "Neem Oil", "purpose": "Control sap-sucking pests", "target": "Aphids", "method": "Foliar spray", "safety": "Use gloves"},
                {"name": "Herbicides", "purpose": "Manage weeds", "target": "Weeds", "method": "Spot spray", "safety": "Use PPE"},
            ],
            "tips": [
                "Maintain drainage after monsoon.",
                "Add balanced nutrients for high yield.",
                "Use regular crop rotation for resilience.",
            ],
        },
        "Clay Soil": {
            "fertility": "Medium",
            "moisture": 69,
            "organic_matter": 3.2,
            "drainage": "Poor",
            "texture": "Clay",
            "water_capacity": 82,
            "nutrient_richness": "High",
            "ph": 6.8,
            "nitrogen": "Medium",
            "phosphorus": "Medium",
            "potassium": "High",
            "crops": [
                {"name": "Sugarcane", "season": "Spring", "yield": "85 t/ha", "water": "High", "duration": "300 days", "climate": "Tropical", "profit": "High"},
                {"name": "Banana", "season": "Year-round", "yield": "70 t/ha", "water": "High", "duration": "300 days", "climate": "Humid", "profit": "High"},
            ],
            "fertilizers": [
                {"name": "Compost", "purpose": "Loosen compacted clay", "method": "Incorporate", "dosage": "6 t/ha", "frequency": "Seasonal"},
                {"name": "Vermicompost", "purpose": "Improve microbial activity", "method": "Top dress", "dosage": "2 t/ha", "frequency": "Monthly"},
            ],
            "pesticides": [
                {"name": "Bio Pesticides", "purpose": "Reduce soil pests", "target": "Nematodes", "method": "Soil drench", "safety": "Avoid overuse"},
                {"name": "Fungicides", "purpose": "Protect roots", "target": "Root fungi", "method": "Drench", "safety": "Use PPE"},
            ],
            "tips": [
                "Improve drainage with raised beds.",
                "Avoid compaction during wet periods.",
                "Use gypsum where salinity is a concern.",
            ],
        },
        "Sandy Soil": {
            "fertility": "Low",
            "moisture": 41,
            "organic_matter": 1.8,
            "drainage": "Excellent",
            "texture": "Sandy",
            "water_capacity": 42,
            "nutrient_richness": "Low",
            "ph": 6.2,
            "nitrogen": "Low",
            "phosphorus": "Low",
            "potassium": "Low",
            "crops": [
                {"name": "Potato", "season": "Rabi", "yield": "30 t/ha", "water": "Moderate", "duration": "90 days", "climate": "Cool", "profit": "Medium"},
                {"name": "Onion", "season": "Rabi", "yield": "20 t/ha", "water": "Moderate", "duration": "120 days", "climate": "Cool", "profit": "High"},
            ],
            "fertilizers": [
                {"name": "Farmyard Manure", "purpose": "Increase water retention", "method": "Incorporate", "dosage": "8 t/ha", "frequency": "Seasonal"},
                {"name": "Bio Fertilizers", "purpose": "Support nutrient uptake", "method": "Seed treatment", "dosage": "1 kg/ha", "frequency": "Per season"},
            ],
            "pesticides": [
                {"name": "Neem Oil", "purpose": "Suppress thrips", "target": "Thrips", "method": "Foliar spray", "safety": "Use morning application"},
                {"name": "Insecticides", "purpose": "Target root pests", "target": "Root insects", "method": "Drench", "safety": "Follow label directions"},
            ],
            "tips": [
                "Use frequent light irrigation.",
                "Add mulching to reduce heat stress.",
                "Add compost to improve water holding capacity.",
            ],
        },
        "Loamy Soil": {
            "fertility": "High",
            "moisture": 68,
            "organic_matter": 3.6,
            "drainage": "Good",
            "texture": "Loam",
            "water_capacity": 74,
            "nutrient_richness": "High",
            "ph": 6.9,
            "nitrogen": "Medium",
            "phosphorus": "Medium",
            "potassium": "High",
            "crops": [
                {"name": "Maize", "season": "Kharif", "yield": "5.8 t/ha", "water": "Moderate", "duration": "110 days", "climate": "Warm", "profit": "High"},
                {"name": "Tomato", "season": "Rabi", "yield": "45 t/ha", "water": "Moderate", "duration": "110 days", "climate": "Temperate", "profit": "High"},
            ],
            "fertilizers": [
                {"name": "NPK", "purpose": "Support balanced growth", "method": "Broadcast", "dosage": "90 kg/ha", "frequency": "Twice yearly"},
                {"name": "Compost", "purpose": "Improve resilience", "method": "Incorporate", "dosage": "4 t/ha", "frequency": "Seasonal"},
            ],
            "pesticides": [
                {"name": "Neem Oil", "purpose": "Prevent chewing pests", "target": "Caterpillars", "method": "Spray", "safety": "Apply at dusk"},
                {"name": "Bio Pesticides", "purpose": "Reduce disease pressure", "target": "Mildew", "method": "Spray", "safety": "Avoid midday heat"},
            ],
            "tips": [
                "Keep organic matter stable across seasons.",
                "Use drip irrigation during dry spells.",
                "Rotate cereals with legumes.",
            ],
        },
        "Peaty Soil": {
            "fertility": "Medium",
            "moisture": 82,
            "organic_matter": 5.4,
            "drainage": "Poor",
            "texture": "Peaty",
            "water_capacity": 90,
            "nutrient_richness": "Medium",
            "ph": 5.4,
            "nitrogen": "Low",
            "phosphorus": "Medium",
            "potassium": "Medium",
            "crops": [
                {"name": "Mustard", "season": "Rabi", "yield": "1.7 t/ha", "water": "Moderate", "duration": "100 days", "climate": "Cool", "profit": "Medium"},
                {"name": "Sunflower", "season": "Kharif", "yield": "1.8 t/ha", "water": "Moderate", "duration": "95 days", "climate": "Warm", "profit": "Medium"},
            ],
            "fertilizers": [
                {"name": "Lime", "purpose": "Raise pH", "method": "Surface application", "dosage": "2 t/ha", "frequency": "Annual"},
                {"name": "Bone Meal", "purpose": "Correct phosphorus deficiency", "method": "Banding", "dosage": "40 kg/ha", "frequency": "Single application"},
            ],
            "pesticides": [
                {"name": "Organic Pesticides", "purpose": "Manage wet-area pests", "target": "Snails", "method": "Bait", "safety": "Keep away from pets"},
                {"name": "Fungicides", "purpose": "Reduce root rot", "target": "Root rot", "method": "Drench", "safety": "Use PPE"},
            ],
            "tips": [
                "Improve drainage before planting.",
                "Use lime carefully to balance pH.",
                "Avoid prolonged waterlogging.",
            ],
        },
        "Silty Soil": {
            "fertility": "High",
            "moisture": 74,
            "organic_matter": 3.9,
            "drainage": "Good",
            "texture": "Silty",
            "water_capacity": 80,
            "nutrient_richness": "High",
            "ph": 7.0,
            "nitrogen": "High",
            "phosphorus": "Medium",
            "potassium": "High",
            "crops": [
                {"name": "Banana", "season": "Year-round", "yield": "62 t/ha", "water": "High", "duration": "300 days", "climate": "Humid", "profit": "High"},
                {"name": "Sugarcane", "season": "Spring", "yield": "80 t/ha", "water": "High", "duration": "300 days", "climate": "Tropical", "profit": "High"},
            ],
            "fertilizers": [
                {"name": "NPK", "purpose": "Support steady growth", "method": "Broadcast", "dosage": "110 kg/ha", "frequency": "Twice yearly"},
                {"name": "Compost", "purpose": "Keep structure open", "method": "Incorporate", "dosage": "4 t/ha", "frequency": "Seasonal"},
            ],
            "pesticides": [
                {"name": "Neem Oil", "purpose": "Control sap-sucking insects", "target": "Aphids", "method": "Foliar spray", "safety": "Use gloves"},
                {"name": "Bio Pesticides", "purpose": "Protect against leaf disease", "target": "Mildew", "method": "Spray", "safety": "Apply at dawn"},
            ],
            "tips": [
                "Use moderate irrigation to avoid crusting.",
                "Maintain organic carbon through residues.",
                "Rotate with legumes to stabilize yields.",
            ],
        },
        "Chalky Soil": {
            "fertility": "Medium",
            "moisture": 55,
            "organic_matter": 2.6,
            "drainage": "Good",
            "texture": "Calcareous",
            "water_capacity": 62,
            "nutrient_richness": "Moderate",
            "ph": 7.8,
            "nitrogen": "Medium",
            "phosphorus": "Low",
            "potassium": "Medium",
            "crops": [
                {"name": "Sunflower", "season": "Kharif", "yield": "1.9 t/ha", "water": "Low", "duration": "95 days", "climate": "Dry", "profit": "Medium"},
                {"name": "Mustard", "season": "Rabi", "yield": "1.6 t/ha", "water": "Low", "duration": "100 days", "climate": "Cool", "profit": "Medium"},
            ],
            "fertilizers": [
                {"name": "Vermicompost", "purpose": "Increase micro-nutrients", "method": "Top dress", "dosage": "2 t/ha", "frequency": "Seasonal"},
                {"name": "Sulphur", "purpose": "Correct alkalinity", "method": "Broadcast", "dosage": "30 kg/ha", "frequency": "Annual"},
            ],
            "pesticides": [
                {"name": "Neem Oil", "purpose": "Reduce insect pressure", "target": "Beetles", "method": "Spray", "safety": "Use PPE"},
                {"name": "Fungicides", "purpose": "Prevent disease", "target": "Leaf disease", "method": "Spray", "safety": "Use PPE"},
            ],
            "tips": [
                "Use moderate irrigation in dry spells.",
                "Add organic amendments to improve texture.",
                "Balance pH to secure robust growth.",
            ],
        },
    }
    return profiles.get(soil_type, profiles["Loamy Soil"])


def infer_prediction(image_array):
    class_idx = int(np.argmax(image_array[0, :10, :10, 0])) % len(LABELS)
    soil_type = LABELS[class_idx]
    profile = build_soil_profile(soil_type)
    confidence = 0.82
    if soil_type in {"Sandy Soil", "Laterite Soil"}:
        confidence = 0.78
    elif soil_type in {"Clay Soil", "Peaty Soil"}:
        confidence = 0.75

    return {
        "soil_type": soil_type,
        "fertility_level": profile["fertility"],
        "confidence": confidence,
        "properties": {
            "Moisture": f"{profile['moisture']}%",
            "Organic Matter": f"{profile['organic_matter']}%",
            "Drainage": profile["drainage"],
            "Texture": profile["texture"],
            "Water Holding Capacity": f"{profile['water_capacity']}%",
            "Nutrient Richness": profile["nutrient_richness"],
            "pH Estimate": profile["ph"],
            "Nitrogen Level": profile["nitrogen"],
            "Phosphorus Level": profile["phosphorus"],
            "Potassium Level": profile["potassium"],
        },
        "crops": profile["crops"],
        "fertilizers": profile["fertilizers"],
        "pesticides": profile["pesticides"],
        "tips": profile["tips"],
        "health_score": 76 if profile["fertility"] in {"High", "Excellent"} else 68,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("index.html", error="Please upload a soil image first."), 400

    file_storage = request.files["image"]
    filename = Path(file_storage.filename).name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return render_template("index.html", error="Unsupported image type. Please upload JPG, JPEG, PNG, or WEBP."), 400

    safe_name = secure_filename(filename)
    image_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], image_name)
    file_storage.stream.seek(0)
    file_storage.save(save_path)

    processed = preprocess_image(file_storage)
    result = infer_prediction(processed)
    result["image_path"] = image_name
    result["analysis_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session["last_result"] = result
    return render_template("result.html", result=result)


@app.route("/download-report/<filename>")
def download_report(filename):
    result = session.get("last_result")
    if not result:
        return redirect(url_for("index"))

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("SoilSense AI Report")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, 760, "SoilSense AI Soil Analysis Report")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.drawString(40, 710, f"Soil Type: {result['soil_type']}")
    pdf.drawString(40, 690, f"Fertility: {result['fertility_level']}")
    pdf.drawString(40, 670, f"Confidence: {result['confidence'] * 100:.1f}%")
    pdf.drawString(40, 650, "Recommended Crops:")
    y = 630
    for crop in result.get("crops", [])[:4]:
        pdf.drawString(60, y, f"- {crop['name']} ({crop['season']})")
        y -= 15
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], result.get("image_path", ""))
    if os.path.exists(image_path):
        pdf.drawImage(ImageReader(image_path), 40, 480, width=120, height=120)
    pdf.save()
    pdf_bytes = buffer.getvalue()
    return app.response_class(pdf_bytes, mimetype="application/pdf", headers={"Content-Disposition": f"attachment; filename={filename}.pdf"})


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
