# Wedding Photo Finder - Backend

Backend API for finding photos of guests in wedding photos using face recognition.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Process Photos (One-time)

Place all wedding photos in the `../photos` directory, then run:

```bash
python process_photos.py
```

This will:
- Scan all images in the photos directory
- Detect faces in each image
- Extract face embeddings
- Save to `../processed_data/face_embeddings.pkl`

**Estimated time:** 10-15 minutes for 4500 photos

### Step 2: Start API Server

```bash
python api.py
```

The API will be available at `http://localhost:5000`

### Step 3: Test API

```bash
# Basic test
python test_api.py

# Test with specific image
python test_api.py path/to/selfie.jpg
```

## API Endpoints

### GET /health
Health check

### GET /stats
Get database statistics

### POST /search
Search for matching photos

**Parameters:**
- `image` (file): Image file to search for
- `threshold` (float, optional): Similarity threshold (default: 0.6)

**Response:**
```json
{
  "success": true,
  "total_matches": 45,
  "matches": [
    {
      "image_path": "path/to/photo.jpg",
      "similarity": 0.87,
      "facial_area": {...}
    }
  ]
}
```

## File Structure

```
backend/
├── process_photos.py    # Process wedding photos (run once)
├── api.py              # Flask API server
├── test_api.py         # API test client
├── requirements.txt    # Python dependencies
└── venv/              # Virtual environment
```

