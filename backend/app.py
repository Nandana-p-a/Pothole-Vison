from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import uuid
import sys
import json
import numpy as np

# --- Configuration ---
MODEL_PATH = os.path.join("model", "best.pt") 
RUNS_DIR = os.path.join("runs", "detect")
CONF_THRES = 0.5

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Model Loading with Error Handling ---
model = None
try:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully.")

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1) 
except Exception as e:
    print(f"CRITICAL ERROR loading YOLO model: {e}")
    sys.exit(1)

os.makedirs(RUNS_DIR, exist_ok=True)

def parse_detection_results(results, is_tracking=False):
    """
    Parses the list of YOLO Results objects to extract bounding box coordinates, 
    confidence, class, and track ID (if tracking), handling potential data structure variations.
    """
    all_detections = []
    
    # Iterate through all result objects (one per image/frame)
    for r in results:
        # Get the names of the classes from the model
        names = r.names 
        
        boxes = r.boxes
        
        # Iterate over each detected bounding box in the current image/frame
        for i in range(len(boxes)):
            # --- SAFER COORDINATE EXTRACTION ---
            # Get the coordinates tensor (shape might be [1, 4] or [4])
            coord_tensor = boxes.xyxy[i].cpu().numpy().flatten()
            
            # Convert the 4 elements (x_min, y_min, x_max, y_max) to a list of floats
            coords = coord_tensor.tolist() 
            
            # Extract confidence, class ID, and track ID
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = names[class_id]
            
            detection = {
                "class_name": class_name,
                "confidence": round(confidence, 4),
                # Format coordinates: (x_min, y_min, x_max, y_max)
                "box_xyxy": [round(c, 2) for c in coords],
            }
            
            if is_tracking and boxes.id is not None:
                # Ensure track ID is correctly converted to a single integer
                track_id = int(boxes.id[i].cpu().numpy())
                detection["track_id"] = track_id
            
            all_detections.append(detection)
            
    return all_detections

@app.route("/detect", methods=["POST"])
def detect():
    """
    Handles file upload, performs object detection/tracking, and returns 
    the count of unique objects and their bounding box coordinates.
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename.lower()

    IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".webp")
    VIDEO_FORMATS = (".mp4", ".avi", ".mov", ".mkv")

    if not filename.endswith(IMAGE_FORMATS + VIDEO_FORMATS):
        return jsonify({"status": "error", "message": "Unsupported file format"}), 400

    # Using UUID for exp_name here again to ensure immediate response is unique,
    # though you might want to use sequential naming from the previous response for better folder organization.
    temp_file = os.path.join(os.getcwd(), f"temp_{uuid.uuid4().hex}_{filename}")
    exp_name = f"exp_{uuid.uuid4().hex[:6]}"
    
    all_detections = [] # List to hold the parsed coordinates and metadata
    is_tracking = False

    try:
        file.save(temp_file)
        
        # --- IMAGE DETECTION ---
        if filename.endswith(IMAGE_FORMATS):
            results = model.predict(
                source=temp_file,
                conf=CONF_THRES,
                save=True,
                project=RUNS_DIR,
                name=exp_name,
                verbose=False
            )
            # Count total detected bounding boxes
            count = sum(len(r.boxes) for r in results)
            
            # Parse results to get coordinates
            all_detections = parse_detection_results(results, is_tracking=False)

        # --- VIDEO TRACKING ---
        elif filename.endswith(VIDEO_FORMATS):
            is_tracking = True
            results = model.track(
                source=temp_file,
                conf=CONF_THRES,
                save=True,
                tracker="bytetrack.yaml", 
                project=RUNS_DIR,
                name=exp_name,
                verbose=False
            )
            
            # Logic to count unique tracked object IDs (potholes)
            unique_ids = set()
            for r in results:
                if r.boxes.id is not None:
                    for tid in r.boxes.id.tolist():
                        unique_ids.add(int(tid))
            
            count = len(unique_ids) # Count is the number of unique potholes
            
            # Parse results to get coordinates and track ID
            all_detections = parse_detection_results(results, is_tracking=True)

    except Exception as e:
        # Catch errors that might occur during the model's predict/track calls
        return jsonify({"status": "error", "message": f"Detection/Tracking failed: {e}"}), 500
    finally:
        # --- CLEANUP ---
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # --- API Response ---
    if count == 0:
        return jsonify({
            "status": "no_detection",
            "message": "No objects detected (potholes, etc.)",
            "detections": []
        })

    # For videos, the final count is the unique count, but we return ALL detections per frame.
    # For images, the count is the number of boxes in the single image.
    return jsonify({
        "status": "success",
        # Use a more generic key since it counts unique tracked objects in video
        "unique_objects_count": count, 
        "output_folder": os.path.join(RUNS_DIR, exp_name).replace("\\", "/"),
        "detections": all_detections # Bounding box coordinates and metadata
    })


if __name__ == "__main__":
    app.run(debug=True)