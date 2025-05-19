import os
from constants import COLORS_RGB, CAR_TYPES
from fastapi import FastAPI, UploadFile, File
from google.cloud import vision
from data import marcas_modelos as marcas_modelos_data
import re
from datetime import datetime

CREDENTIALS_PATH = "gothic-isotope-460019-b8-cfcab02a3d9a.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

app = FastAPI()
client = vision.ImageAnnotatorClient()


def get_color_name(rgb):
    return find_nearest_color(rgb)

def find_nearest_color(rgb_triplet):
    min_difference = None
    nearest_name = None
    for name, rgb_std in COLORS_RGB.items():
        difference = sum(abs(c1 - c2) for c1, c2 in zip(rgb_triplet, rgb_std))
        if min_difference is None or difference < min_difference:
            min_difference = difference
            nearest_name = name
    return nearest_name

def extract_car_year(labels_and_detections):
    """
    Extracts a possible car year from a list of labels and detections.
    Validates that the year is between the first car (1886) and the current year.
    """
    current_year = datetime.now().year
    possible_years = []
    
    for text in labels_and_detections:
        numbers = re.findall(r'\b(1[89]\d\d|20\d\d)\b', text)
        possible_years.extend([int(num) for num in numbers])
    
    print("possible_years")
    print(possible_years)
    
    valid_years = [year for year in possible_years if 1886 <= year <= current_year]
    
    
    if valid_years:
        return max(valid_years)
    return None

def identify_car_type(labels_and_detections):
    """
    Identifies car type from labels and detections using predefined car type categories.
    Returns the most likely car type based on matches.
    """
    type_matches = {}
    
    lowercase_detections = [text.lower() for text in labels_and_detections]
    
    for car_type, keywords in CAR_TYPES.items():
        matches = 0
        for keyword in keywords:
            for detection in lowercase_detections:
                if keyword in detection:
                    matches += 1
        
        if matches > 0:
            type_matches[car_type] = matches
    
    if type_matches:
        return max(type_matches.items(), key=lambda x: x[1])[0]
    return None

@app.post("/analyze-car/")
async def analyze_car(file: UploadFile = File(...)):

    content = await file.read()
    image = vision.Image(content=content)

    response = client.annotate_image({
        'image': image,
        'features': [
            {'type_': vision.Feature.Type.LABEL_DETECTION},
            {'type_': vision.Feature.Type.OBJECT_LOCALIZATION},
            {'type_': vision.Feature.Type.TEXT_DETECTION},
            {'type_': vision.Feature.Type.WEB_DETECTION},
            {'type_': vision.Feature.Type.IMAGE_PROPERTIES},
        ],
    })

    labels = [label.description for label in response.label_annotations]
    web_detections = [
        entity.description for entity in response.web_detection.web_entities
        if entity.description
    ]

    array_labels_and_web_detections = labels + web_detections
    
    car_year = extract_car_year(array_labels_and_web_detections)
    
    car_type = identify_car_type(array_labels_and_web_detections)

    colors = sorted(
        response.image_properties_annotation.dominant_colors.colors,
        key=lambda c: c.score,
        reverse=True
    )
    dominant_color = {
        "r": int(colors[0].color.red),
        "g": int(colors[0].color.green),
        "b": int(colors[0].color.blue)
    } if colors else None

    brands_models = marcas_modelos_data

    brand = None
    model = None

    for brand_name, models in brands_models.items():
        if any(brand_name.lower() in val.lower() for val in array_labels_and_web_detections):
            brand = brand_name
            
            sorted_models = sorted(models, key=lambda x: len(x.split()), reverse=True)
            for model_name in sorted_models:
                if any(model_name.lower() in val.lower() for val in array_labels_and_web_detections):
                    model = model_name
                    break
            
            if model is None:
                for label in array_labels_and_web_detections:
                    label_lower = label.lower()
                    if brand_name.lower() in label_lower:
                        possible_model = label_lower.replace(brand_name.lower(), "").strip()
                        best_match = None
                        best_match_score = 0
                        for model_name in models:
                            model_lower = model_name.lower()
                            if model_lower in possible_model:
                                if len(model_lower) > best_match_score:
                                    best_match = model_name
                                    best_match_score = len(model_lower)
                        if best_match:
                            model = best_match
                            break
            break

    color_name = get_color_name((dominant_color["r"], dominant_color["g"], dominant_color["b"]))
    
    return {
        "brand": brand,
        "model": model,
        "year": car_year,
        "type": car_type,
        "color": dominant_color,
        "color_name": color_name
    }


@app.post("/general-car-description/")
async def general_car_description(file: UploadFile = File(...)):
    content = await file.read()
    image = vision.Image(content=content)

    response = client.annotate_image({
        'image': image,
        'features': [
            {'type_': vision.Feature.Type.LABEL_DETECTION},
            {'type_': vision.Feature.Type.OBJECT_LOCALIZATION},
            {'type_': vision.Feature.Type.TEXT_DETECTION},
            {'type_': vision.Feature.Type.WEB_DETECTION},
            {'type_': vision.Feature.Type.IMAGE_PROPERTIES},
        ],
    })

    results = {
        "labels": [label.description for label in response.label_annotations],
        "objects": [
            {"name": obj.name, "score": obj.score}
            for obj in response.localized_object_annotations
        ],
        "text": response.text_annotations[0].description if response.text_annotations else "",
        "colors": [
            {
                "color": {
                    "r": color.color.red,
                    "g": color.color.green,
                    "b": color.color.blue
                },
                "score": color.score
            }
            for color in response.image_properties_annotation.dominant_colors.colors
        ],
        "web_detections": [
            entity.description for entity in response.web_detection.web_entities
            if entity.description
        ]
    }

    return results
