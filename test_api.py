"""
Script to test the API
"""
import requests
from pathlib import Path

API_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_stats():
    """Test stats endpoint"""
    print("Testing /stats endpoint...")
    response = requests.get(f"{API_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_search(image_path):
    """Test search endpoint"""
    print(f"Testing /search endpoint with image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'threshold': 0.6}
        
        response = requests.post(f"{API_URL}/search", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Total matches: {result['total_matches']}")
        
        if result['total_matches'] > 0:
            print("\nTop 5 matches:")
            for i, match in enumerate(result['matches'][:5], 1):
                print(f"  {i}. {Path(match['image_path']).name} - Similarity: {match['similarity']:.2%}")
    else:
        print(f"Error: {response.json()}")
    
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("API Test Client")
    print("=" * 60)
    print()
    
    # Test basic endpoints
    test_health()
    test_stats()
    
    # Test search with an image
    print("To test search, provide a test image path:")
    print("Example: python test_api.py path/to/your/selfie.jpg")
    print()
    
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        if Path(test_image).exists():
            test_search(test_image)
        else:
            print(f"Error: Image not found at {test_image}")

