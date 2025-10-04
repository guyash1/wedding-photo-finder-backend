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
cv2.setNumThreads(0)  # Prevent multi-threading issues
import tempfile
import os
import re
import zipfile
import io
import boto3
from botocore.exceptions import ClientError
import gc  # Garbage collector for memory cleanup

app = Flask(__name__)
CORS(app, origins=['*'])

# Load face embeddings database
EMBEDDINGS_FILE = Path("../processed_data/face_embeddings.pkl")
PHOTOS_BASE_DIR = Path("../photos").resolve()
face_database = None

# R2 Configuration
R2_ACCESS_KEY = os.environ.get('R2_ACCESS_KEY')
R2_SECRET_KEY = os.environ.get('R2_SECRET_KEY')
R2_ACCOUNT_ID = os.environ.get('R2_ACCOUNT_ID')
R2_ENDPOINT = os.environ.get('R2_ENDPOINT')  # Optional: full endpoint URL like https://xxxxx.r2.cloudflarestorage.com
R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME', 'wedding-photos-amit-guy')

# Initialize R2 client if credentials are available
r2_client = None
if R2_ACCESS_KEY and R2_SECRET_KEY and (R2_ENDPOINT or R2_ACCOUNT_ID):
    try:
        endpoint_url = R2_ENDPOINT or f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com'
        r2_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            region_name='auto'
        )
        print("✅ R2 client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize R2 client: {e}")
        r2_client = None
else:
    print("⚠️ R2 credentials/endpoint not found, using local storage")

def load_database():
    """Load the face embeddings database from local disk or R2"""
    global face_database

    # If local file doesn't exist, try R2
    if not EMBEDDINGS_FILE.exists():
        if r2_client:
            try:
                print("📥 Embeddings file not found locally. Downloading from R2...")
                obj = r2_client.get_object(Bucket=R2_BUCKET_NAME, Key='processed_data/face_embeddings.pkl')
                data = obj['Body'].read()
                EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(EMBEDDINGS_FILE, 'wb') as f:
                    f.write(data)
                print("✅ Downloaded embeddings from R2 successfully!")
            except Exception as e:
                print(f"❌ Failed to fetch embeddings from R2: {e}")
                return False
        else:
            print(f"❌ ERROR: No R2 client and embeddings file not found at {EMBEDDINGS_FILE}")
            return False

    # Load the embeddings file
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            face_database = pickle.load(f)
        print(f"✅ Loaded {len(face_database)} face embeddings from database")
        return True
    except Exception as e:
        print(f"❌ Error loading embeddings file: {e}")
        return False

# Startup message (database will load on first request)
print("=" * 60)
print("🚀 Starting Face Recognition API - Lazy Loading Mode")
print("=" * 60)
print("💡 Database will load on first search request to save memory")
print("=" * 60)

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def ensure_database_loaded():
    """Load database only when needed (lazy loading)"""
    global face_database
    
    if face_database is None:
        print("⏳ Loading database on demand...")
        if not load_database():
            raise Exception("Failed to load face database")
    
    return True

def find_matching_photos(uploaded_embedding, threshold=0.6):
    """Find all photos where the person appears"""
    # Ensure database is loaded
    ensure_database_loaded()
    
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
            
            result = jsonify({
                'success': True,
                'total_matches': len(matches),
                'matches': matches  # Return all matches
            })
            
            # Clean up memory after search
            del matches, uploaded_embedding, embeddings
            gc.collect()  # Force garbage collection
            print(f"🧹 Memory cleaned after search")
            
            return result
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        # Clean up memory on error too
        gc.collect()
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
    """Serve image files from R2 or local storage"""
    try:
        # Convert Windows path separators to forward slashes
        image_path = image_path.replace('\\', '/')
        
        # If R2 client is available, serve from R2
        if r2_client:
            try:
                # Get image from R2
                response = r2_client.get_object(Bucket=R2_BUCKET_NAME, Key=image_path)
                
                # Return image data
                from flask import Response
                return Response(
                    response['Body'].read(),
                    mimetype='image/jpeg',
                    headers={
                        'Content-Disposition': f'inline; filename="{os.path.basename(image_path)}"'
                    }
                )
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    return jsonify({'error': 'Image not found in R2'}), 404
                else:
                    return jsonify({'error': f'R2 error: {str(e)}'}), 500
        
        # Fallback to local storage
        else:
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
    """Download image file from R2 or local storage"""
    try:
        # Convert Windows path separators to forward slashes
        image_path = image_path.replace('\\', '/')

        if r2_client:
            try:
                obj = r2_client.get_object(Bucket=R2_BUCKET_NAME, Key=image_path)
                from flask import Response
                return Response(
                    obj['Body'].read(),
                    mimetype='application/octet-stream',
                    headers={
                        'Content-Disposition': f'attachment; filename="{os.path.basename(image_path)}"'
                    }
                )
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    return jsonify({'error': f'R2 error: {str(e)}'}), 500
                # else fall back to local

        # Fallback to local storage
        full_path = PHOTOS_BASE_DIR / image_path
        if not str(full_path.resolve()).startswith(str(PHOTOS_BASE_DIR.resolve())):
            return jsonify({'error': 'Access denied'}), 403
        if not full_path.exists():
            return jsonify({'error': 'Image not found'}), 404
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
                    clean_path = image_path.replace('\\', '/')

                    if r2_client:
                        try:
                            obj = r2_client.get_object(Bucket=R2_BUCKET_NAME, Key=clean_path)
                            data = obj['Body'].read()
                            # Use arcname as filename only
                            arcname = os.path.basename(clean_path)
                            zip_file.writestr(arcname, data)
                            continue
                        except ClientError as e:
                            if e.response['Error']['Code'] != 'NoSuchKey':
                                print(f"R2 error for {clean_path}: {e}")
                            # fall back to local

                    full_path = PHOTOS_BASE_DIR / clean_path
                    if not str(full_path.resolve()).startswith(str(PHOTOS_BASE_DIR.resolve())):
                        continue
                    if not full_path.exists():
                        continue
                    zip_file.write(full_path, os.path.basename(clean_path))

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
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

