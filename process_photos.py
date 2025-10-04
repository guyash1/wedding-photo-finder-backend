"""
סקריפט לעיבוד תמונות החתונה וזיהוי פנים - גרסה מהירה
"""
import os
import pickle
from pathlib import Path
from deepface import DeepFace
from tqdm import tqdm
import cv2
import numpy as np

# נתיבים
PHOTOS_DIR = Path("../photos")
OUTPUT_FILE = Path("../processed_data/face_embeddings.pkl")

# הגדרות לאופטימיזציה
MAX_IMAGE_SIZE = 1920  # מקסימום רוחב/גובה של תמונה
BATCH_SIZE = 10  # כמה תמונות לעבד בבת אחת

def find_all_images(directory):
    """מוצא את כל קבצי התמונות"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix in image_extensions:
                images.append(os.path.join(root, file))
    
    return images

def resize_image_if_needed(img):
    """מקטין תמונה אם היא גדולה מדי"""
    height, width = img.shape[:2]
    
    if height > MAX_IMAGE_SIZE or width > MAX_IMAGE_SIZE:
        # מצא את היחס
        scale = MAX_IMAGE_SIZE / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # שנה גודל
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return img

def process_images():
    """עובר על כל התמונות ומזהה פנים"""
    
    # יוצר תיקייה לפלט אם לא קיימת
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    
    # Find all images
    print("Searching for images...")
    all_images = find_all_images(PHOTOS_DIR)
    print(f"Found {len(all_images)} images\n")
    
    # Data structure for storing info
    face_data = []
    failed_images = []
    
    print("Starting to process images...\n")
    
    for img_path in tqdm(all_images, desc="Processing"):
        try:
            # Read image
            img = cv2.imread(img_path)
            
            if img is None:
                failed_images.append((img_path, "Failed to read file"))
                continue
            
            # Resize if too large (speeds up processing significantly)
            img = resize_image_if_needed(img)
            
            # Save temporarily for DeepFace
            temp_path = str(Path(img_path).with_suffix('.temp.jpg'))
            cv2.imwrite(temp_path, img)
            
            try:
                # Use Facenet (not 512) - faster and still accurate
                # opencv detector is much faster than mtcnn/retinaface
                embeddings = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    enforce_detection=False,
                    detector_backend="opencv"
                )
                
                # Save info for all faces found
                for face in embeddings:
                    face_data.append({
                        'image_path': img_path,
                        'embedding': face['embedding'],
                        'facial_area': face.get('facial_area', None)
                    })
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except Exception as e:
            failed_images.append((img_path, str(e)))
    
    # Save data
    print(f"\nSaving data...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(face_data, f)
    
    # Summary
    print(f"\nComplete!")
    print(f"Statistics:")
    print(f"   - Images processed: {len(all_images)}")
    print(f"   - Faces detected: {len(face_data)}")
    print(f"   - Failed images: {len(failed_images)}")
    print(f"   - Data saved to: {OUTPUT_FILE}")
    
    if failed_images:
        print(f"\nFailed images (first 10):")
        for img, error in failed_images[:10]:
            print(f"   - {Path(img).name}: {error}")

if __name__ == "__main__":
    # Set UTF-8 encoding for console output
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("=" * 60)
    print("Wedding Photo Face Recognition System")
    print("=" * 60)
    print()
    
    process_images()

