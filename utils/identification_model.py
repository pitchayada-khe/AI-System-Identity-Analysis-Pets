import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import time

# ---- DEVICE CONFIGURATION ---- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- MODEL ARCHITECTURE (From final.ipynb) ---- #
class EmbeddingModel(nn.Module):
    def __init__(self, base_model):
        super(EmbeddingModel, self).__init__()
        self.base_model = base_model
        
    def forward(self, x):
        x = self.base_model(x)
        # L2 Normalization
        x = F.normalize(x, p=2, dim=1) 
        return x

def build_model(model_path):
    base_model = models.convnext_tiny(weights=None)
    num_features = base_model.classifier[2].in_features
    base_model.classifier = nn.Sequential(
        models.convnext.LayerNorm2d(num_features, eps=1e-6),
        nn.Flatten(1),
        nn.Linear(num_features, 128)
    )
    model = EmbeddingModel(base_model)
    
    # Load Weights
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval()
    return model

def download_model():
    hf_hub_download(repo_id="Muyumq/Dog-Cat_Identification", filename="best_dog_Face_model.pth", local_dir="models/iden")
    hf_hub_download(repo_id="Muyumq/Dog-Cat_Identification", filename="best_dog_Nose_model.pth", local_dir="models/iden")
    hf_hub_download(repo_id="Muyumq/Dog-Cat_Identification", filename="best_cat_Face_model.pth", local_dir="models/iden")
    hf_hub_download(repo_id="Muyumq/Dog-Cat_Identification", filename="best_cat_Nose_model.pth", local_dir="models/iden")
    return True

# ---- LOAD ALL 4 MODELS ---- #
print("Loading Identification Models...")
download_model()
cat_face_model = build_model("models/iden/best_cat_Face_model.pth")
cat_nose_model = build_model("models/iden/best_cat_Nose_model.pth")
dog_face_model = build_model("models/iden/best_dog_Face_model.pth")
dog_nose_model = build_model("models/iden/best_dog_Nose_model.pth")
print("All Identity Models Loaded Successfully!")

# ---- IMAGE PREPROCESSING PREPARATION ---- #
# Expected input for ConvNeXt: RGB, 224x224, normalized
# Using BICUBIC interpolation to match the training pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---- DATABASE ---- #
# Structure: 
# {
#   "dog": [ {"face_emb": tensor_1x128, "nose_emb": tensor_1x128}, ... ],
#   "cat": [ {"face_emb": tensor_1x128, "nose_emb": tensor_1x128}, ... ]
# }
known_database = {"dog": [], "cat": []}

def clear_known_database(max_age_seconds=30):
    """Clear old entries from the database that are older than max_age_seconds."""
    current_time = time.time()
    for animal_class in ["dog", "cat"]:
        known_database[animal_class] = [
            record for record in known_database[animal_class]
            if current_time - record.get("timestamp", current_time) <= max_age_seconds
        ]

# ---- THRESHOLDS ---- #
# Format: (Face_Threshold, Nose_Threshold)
THRESHOLDS = {
    "dog": (0.65, 1.00),
    "cat": (0.50, 0.80)
}

def get_embedding(model, image_bgr):
    """
    Convert BGR image from OpenCV to PyTorch embedding
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image_rgb)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE) # Add batch dimension -> 1x3x224x224
    
    with torch.no_grad():
        embedding = model(img_tensor)
        
    return embedding

def identification(detection_data):
    """
    Compares the current detection against the known_database using Face and Nose embeddings.
    Returns True if it's a known animal, False if it's new.
    """
    if detection_data is None:
        return False
        
    animal_class = detection_data["class"]
    if animal_class not in ["dog", "cat"]:
        return False
        
    face_image = detection_data["image"]
    
    # We MUST have both face and nose logic requested by user
    if "nose_data" not in detection_data or detection_data["nose_data"] is None:
        # Cannot compare both if nose is missing, treat as new or skip
        return False
        
    nose_image = detection_data["nose_data"]["image"]

    # 1. Select the correct models
    if animal_class == "dog":
        face_model = dog_face_model
        nose_model = dog_nose_model
        face_thresh, nose_thresh = THRESHOLDS["dog"]
    else:
        face_model = cat_face_model
        nose_model = cat_nose_model
        face_thresh, nose_thresh = THRESHOLDS["cat"]

    # 2. Extract Embeddings
    current_face_emb = get_embedding(face_model, face_image)
    current_nose_emb = get_embedding(nose_model, nose_image)

    # 3. Compare with known_database for this specific animal class
    db_records = known_database[animal_class]
    
    match_found = False
    
    # Initialize distances in case db_records is empty
    face_dist = -1.0
    nose_dist = -1.0
    
    for record in db_records:
        stored_face_emb = record["face_emb"]
        stored_nose_emb = record["nose_emb"]
        
        # Calculate Euclidean Distance
        face_dist = F.pairwise_distance(current_face_emb, stored_face_emb, p=2).item()
        nose_dist = F.pairwise_distance(current_nose_emb, stored_nose_emb, p=2).item()
        
        # Check against thresholds
        if face_dist < face_thresh and nose_dist < nose_thresh:
            match_found = True
            break
            
    # 4. Handle Result
    if match_found:
        return True, face_dist, nose_dist
    else:
        # Save as a NEW identity
        new_identity = {
            "face_emb": current_face_emb,
            "nose_emb": current_nose_emb,
            "timestamp": time.time()
        }
        known_database[animal_class].append(new_identity)
        
        # If it's new but we compared it against something, return the closest distance
        # Otherwise return -1 to indicate it's the very first one
        return False, face_dist, nose_dist