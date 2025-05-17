from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import vision
from google.oauth2 import service_account
import io

CREDENTIALS_PATH = "gothic-isotope-460019-b8-cfcab02a3d9a.json"

credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
client = vision.ImageAnnotatorClient(credentials=credentials)

def extract_car_features(labels, objects):
    car_info = {
        "brand": None,
        "model": None,
        "approximate_year": None,
        "color": None,
        "type": None,
        "style": None,
        "distinctive_features": []
    }
    
    brands = [
        "Toyota", "Honda", "Ford", "BMW", "Mercedes", "Audi", "Volkswagen", "Nissan", 
        "Chevrolet", "Hyundai", "Kia", "Lexus", "Porsche", "Ferrari", "Lamborghini",
        "Maserati", "Jaguar", "Land Rover", "Subaru", "Mazda", "Volvo", "Tesla"
    ]
    
    color_map = {
        "red": "red", 
        "blue": "blue", 
        "black": "black", 
        "white": "white",
        "silver": "silver", 
        "gray": "gray", 
        "yellow": "yellow", 
        "green": "green",
        "burgundy": "burgundy", 
        "purple": "purple", 
        "orange": "orange", 
        "brown": "brown",
        "gold": "gold", 
        "beige": "beige"
    }
    
    for label in labels:
        desc = label.description.lower()
        
        for brand in brands:
            if brand.lower() in desc and label.score > 0.7:
                car_info["brand"] = brand
                break
        
        for eng_color, color in color_map.items():
            if eng_color in desc and label.score > 0.7:
                car_info["color"] = color
                break
        
        if ("suv" in desc or "crossover" in desc) and car_info["type"] is None:
            car_info["type"] = "SUV"
        elif "sedan" in desc and car_info["type"] is None:
            car_info["type"] = "Sedan"
        elif ("sports car" in desc or "sport car" in desc) and car_info["type"] is None:
            car_info["type"] = "Sports Car"
        elif ("coupe" in desc or "coupé" in desc) and car_info["type"] is None:
            car_info["type"] = "Coupe"
        elif ("pickup" in desc or "truck" in desc) and car_info["type"] is None:
            car_info["type"] = "Pickup Truck"
        elif "hatchback" in desc and car_info["type"] is None:
            car_info["type"] = "Hatchback"
        elif "convertible" in desc and car_info["type"] is None:
            car_info["type"] = "Convertible"
        elif "van" in desc and car_info["type"] is None:
            car_info["type"] = "Van"
        
        if "luxury" in desc or "premium" in desc and car_info["style"] is None:
            car_info["style"] = "Luxury"
        elif "sport" in desc and "sports car" not in desc and car_info["style"] is None:
            car_info["style"] = "Sporty"
        elif "classic" in desc and car_info["style"] is None:
            car_info["style"] = "Classic"
        elif "vintage" in desc and car_info["style"] is None:
            car_info["style"] = "Vintage"
        elif "modern" in desc and car_info["style"] is None:
            car_info["style"] = "Modern"
        
        if label.score > 0.7:
            if "convertible" in desc and "convertible" not in car_info["distinctive_features"]:
                car_info["distinctive_features"].append("convertible")
            elif "sunroof" in desc and "sunroof" not in car_info["distinctive_features"]:
                car_info["distinctive_features"].append("sunroof")
            elif "spoiler" in desc and "spoiler" not in car_info["distinctive_features"]:
                car_info["distinctive_features"].append("spoiler")
            elif "alloy wheels" in desc and "alloy wheels" not in car_info["distinctive_features"]:
                car_info["distinctive_features"].append("alloy wheels")
            elif ("two door" in desc or "2 door" in desc or "2-door" in desc) and "2 doors" not in car_info["distinctive_features"]:
                car_info["distinctive_features"].append("2 doors")
            elif ("four door" in desc or "4 door" in desc or "4-door" in desc) and "4 doors" not in car_info["distinctive_features"]:
                car_info["distinctive_features"].append("4 doors")
    
    for label in labels:
        desc = label.description.lower()
        if car_info["brand"] is not None:
            brand = car_info["brand"].lower()
            if brand == "toyota" and any(x in desc for x in ["corolla", "camry", "rav4", "prius"]):
                for model in ["Corolla", "Camry", "RAV4", "Prius"]:
                    if model.lower() in desc:
                        car_info["model"] = model
                        break
            elif brand == "honda" and any(x in desc for x in ["civic", "accord", "cr-v"]):
                for model in ["Civic", "Accord", "CR-V"]:
                    if model.lower() in desc:
                        car_info["model"] = model
                        break
    
    if car_info["type"] is None:
        car_info["type"] = "Car"
    
    search_terms = generate_search_terms(car_info)
    car_info["search_terms"] = search_terms
    
    return car_info

def generate_search_terms(car_info):
    search_terms = []
    
    main_term = ""
    if car_info["brand"] is not None:
        main_term += car_info["brand"] + " "
    if car_info["model"] is not None:
        main_term += car_info["model"] + " "
    elif car_info["type"] is not None:
        main_term += car_info["type"] + " "
    if car_info["color"] is not None:
        main_term += car_info["color"]
    
    if main_term.strip():
        search_terms.append(main_term.strip())
    else:
        search_terms.append("car")
    
    alt_term = ""
    if car_info["type"] is not None:
        alt_term += car_info["type"] + " "
    if car_info["style"] is not None:
        alt_term += car_info["style"] + " "
    if car_info["distinctive_features"]:
        features = car_info["distinctive_features"][:2]
        alt_term += " ".join(features)
    
    if alt_term.strip() and alt_term.strip() != search_terms[0]:
        search_terms.append(alt_term.strip())
    
    specific_term = ""
    if car_info["brand"] is not None:
        specific_term += car_info["brand"] + " "
    if car_info["model"] is not None:
        specific_term += car_info["model"] + " "
    elif car_info["type"] is not None:
        specific_term += car_info["type"] + " "
    
    if specific_term.strip() and specific_term.strip() != search_terms[0]:
        specific_term += "similar cars"
        search_terms.append(specific_term.strip())
    
    if not search_terms:
        search_terms.append("car")
    
    return search_terms

app = FastAPI()

@app.post("/describe-car-image/")
async def describe_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = vision.Image(content=contents)

        label_response = client.label_detection(image=image, max_results=15)
        object_response = client.object_localization(image=image)
        
        logo_response = client.logo_detection(image=image)
        
        all_labels = list(label_response.label_annotations)
        for logo in logo_response.logo_annotations:
            fake_label = vision.EntityAnnotation(
                description=logo.description,
                score=logo.score
            )
            all_labels.append(fake_label)

        car_info = extract_car_features(
            all_labels,
            object_response.localized_object_annotations
        )

        return JSONResponse(content=car_info)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
