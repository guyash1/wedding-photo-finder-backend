"""
Flask API for face matching
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
from pathlib import Path
import numpy as np
from deepface import DeepFace
import cv2
import tempfile
import os
import re
import zipfile
import io

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173', 'http://127.0.0.1:5173'])

# Load face embeddings database
EMBEDDINGS_FILE = Path("../processed_data/face_embeddings.pkl")
PHOTOS_BASE_DIR = Path("../photos").resolve()
face_database = None

def load_database():
    """Load the face embeddings database"""
    global face_database
    
    if not EMBEDDINGS_FILE.exists():
        print(f"ERROR: Embeddings file not found at {EMBEDDINGS_FILE}")
        return False
    
    with open(EMBEDDINGS_FILE, 'rb') as f:
        face_database = pickle.load(f)
    
    print(f"Loaded {len(face_database)} face embeddings from database")
    return True

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def find_matching_photos(uploaded_embedding, threshold=0.6):
    """Find all photos where the person appears"""
    matches = []
    
    for face_entry in face_database:
        similarity = cosine_similarity(uploaded_embedding, face_entry['embedding'])
        
        if similarity >= threshold:
            matches.append({
                'image_path': face_entry['image_path'],
                'similarity': float(similarity),
                'facial_area': face_entry.get('facial_area')
            })
    
    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    return matches

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database_loaded': face_database is not None,
        'total_faces': len(face_database) if face_database else 0
    })

@app.route('/search', methods=['POST'])
def search_faces():
    """
    Main endpoint - receives an image and returns matching photos
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    # Get threshold from request
    threshold = float(request.form.get('threshold', 0.6))
    
    try:
        # Save uploaded image temporarily and preprocess it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image_file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Preprocess image for better face detection
        img = cv2.imread(temp_path)
        if img is not None:
            # Resize if too large (helps with detection)
            height, width = img.shape[:2]
            if height > 1920 or width > 1920:
                scale = 1920 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Enhance image quality
            img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)  # Slight brightness/contrast boost
            
            # Save preprocessed image
            cv2.imwrite(temp_path, img)
        
        try:
            # Extract embedding from uploaded image
            # Try multiple detection methods for better compatibility
            embeddings = None
            
            # Try with different detectors
            detectors = ["opencv", "mtcnn", "retinaface"]
            
            for detector in detectors:
                try:
                    embeddings = DeepFace.represent(
                        img_path=temp_path,
                        model_name="Facenet",
                        enforce_detection=False,  # More lenient
                        detector_backend=detector
                    )
                    if embeddings:
                        print(f"Successfully detected face using {detector}")
                        break
                except Exception as e:
                    print(f"Failed with {detector}: {e}")
                    continue
            
            # If still no face detected, try with different model
            if not embeddings:
                try:
                    embeddings = DeepFace.represent(
                        img_path=temp_path,
                        model_name="VGG-Face",  # Alternative model
                        enforce_detection=False,
                        detector_backend="opencv"
                    )
                    print("Successfully detected face using VGG-Face model")
                except Exception as e:
                    print(f"Failed with VGG-Face: {e}")
            
            if not embeddings:
                return jsonify({'error': 'No face detected in uploaded image'}), 400
            
            # Use the first detected face
            uploaded_embedding = embeddings[0]['embedding']
            
            # Find matching photos
            matches = find_matching_photos(uploaded_embedding, threshold)
            
            # Sort by image number (extract number from filename)
            def get_image_number(match):
                filename = match['image_path'].split('\\')[-1]  # Get filename
                number_match = re.search(r'\((\d+)\)', filename)
                return int(number_match.group(1)) if number_match else 0
            
            matches.sort(key=get_image_number)
            
            # Sanity check - if we found more matches than total images, something is wrong
            if len(matches) > 4504:  # We know there are 4504 total images
                return jsonify({
                    'success': False,
                    'error': 'Too many matches found - please try a clearer photo or different angle'
                }), 400
            
            # If too few matches, suggest lowering threshold
            if len(matches) < 3:
                return jsonify({
                    'success': False,
                    'error': 'Very few matches found - please try a clearer photo or different angle'
                }), 400
            
            return jsonify({
                'success': True,
                'total_matches': len(matches),
                'matches': matches  # Return all matches
            })
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get database statistics"""
    if face_database is None:
        return jsonify({'error': 'Database not loaded'}), 500
    
    # Count unique images
    unique_images = set(entry['image_path'] for entry in face_database)
    
    return jsonify({
        'total_faces': len(face_database),
        'total_images': len(unique_images),
        'avg_faces_per_image': round(len(face_database) / len(unique_images), 2)
    })

@app.route('/images/<path:image_path>')
def serve_image(image_path):
    """Serve image files from the photos directory"""
    try:
        # Convert Windows path separators to forward slashes
        image_path = image_path.replace('\\', '/')
        
        # Build full path
        full_path = PHOTOS_BASE_DIR / image_path
        
        # Security check - ensure path is within photos directory
        if not str(full_path.resolve()).startswith(str(PHOTOS_BASE_DIR.resolve())):
            return jsonify({'error': 'Access denied'}), 403
        
        # Check if file exists
        if not full_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        # Get directory and filename
        directory = full_path.parent
        filename = full_path.name
        
        return send_from_directory(directory, filename, as_attachment=False)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:image_path>')
def download_image(image_path):
    """Download image files from the photos directory"""
    try:
        # Convert Windows path separators to forward slashes
        image_path = image_path.replace('\\', '/')
        
        # Build full path
        full_path = PHOTOS_BASE_DIR / image_path
        
        # Security check - ensure path is within photos directory
        if not str(full_path.resolve()).startswith(str(PHOTOS_BASE_DIR.resolve())):
            return jsonify({'error': 'Access denied'}), 403
        
        # Check if file exists
        if not full_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        # Get directory and filename
        directory = full_path.parent
        filename = full_path.name
        
        return send_from_directory(directory, filename, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-all', methods=['POST'])
def download_all_images():
    """Download all matching images as a ZIP file"""
    try:
        # Get the list of image paths from request
        data = request.get_json()
        if not data or 'image_paths' not in data:
            return jsonify({'error': 'No image paths provided'}), 400
        
        image_paths = data['image_paths']
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for image_path in image_paths:
                try:
                    # Convert Windows path separators to forward slashes
                    clean_path = image_path.replace('\\', '/')
                    
                    # Build full path
                    full_path = PHOTOS_BASE_DIR / clean_path
                    
                    # Security check
                    if not str(full_path.resolve()).startswith(str(PHOTOS_BASE_DIR.resolve())):
                        continue
                    
                    # Check if file exists
                    if not full_path.exists():
                        continue
                    
                    # Add file to ZIP
                    filename = full_path.name
                    zip_file.write(full_path, filename)
                    
                except Exception as e:
                    print(f"Error adding {image_path} to ZIP: {e}")
                    continue
        
        zip_buffer.seek(0)
        
        # Return ZIP file
        from flask import Response
        return Response(
            zip_buffer.getvalue(),
            mimetype='application/zip',
            headers={
                'Content-Disposition': 'attachment; filename=wedding_photos.zip',
                'Content-Type': 'application/zip'
            }
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Wedding Photo Finder API")
    print("=" * 60)
    print()
    
    # Load database
    if not load_database():
        print("ERROR: Could not load face database. Please run process_photos.py first.")
        exit(1)
    
    print()
    print("Starting API server...")
    print("API will be available at: http://localhost:5000")
    print()
    print("Endpoints:")
    print("  GET  /health  - Health check")
    print("  GET  /stats   - Database statistics")
    print("  POST /search  - Search for matching photos")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True)

