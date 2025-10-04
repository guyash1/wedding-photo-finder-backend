#!/usr/bin/env python3
"""
Script to upload wedding photos to Cloudflare R2
Supports configuration via environment variables or interactive prompts.

Environment variables (preferred):
  - R2_ACCESS_KEY
  - R2_SECRET_KEY
  - R2_ENDPOINT  (e.g. https://xxxxxxxxxxxxxxxx.r2.cloudflarestorage.com) OR R2_ACCOUNT_ID
  - R2_BUCKET_NAME (e.g. wedding-photos-amit-guy)
"""
import os
import boto3
from pathlib import Path
import sys
from tqdm import tqdm
import mimetypes

def _detect_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or 'image/jpeg'


def upload_photos_to_r2():
    """Upload all photos to Cloudflare R2"""

    # R2 Configuration (prefer env vars; prompt only if missing)
    R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY') or input("Enter your R2 Access Key ID: ").strip()
    R2_SECRET_KEY = os.getenv('R2_SECRET_KEY') or input("Enter your R2 Secret Access Key: ").strip()
    endpoint_or_account = os.getenv('R2_ENDPOINT') or os.getenv('R2_ACCOUNT_ID') or \
        input("Enter your R2 Endpoint URL (recommended) OR Account ID: ").strip()
    BUCKET_NAME = os.getenv('R2_BUCKET_NAME') or \
        (input("Enter your R2 Bucket name (default: wedding-photos-amit-guy): ").strip() or "wedding-photos-amit-guy")

    # Build endpoint URL
    if endpoint_or_account.startswith('http'):
        endpoint_url = endpoint_or_account.rstrip('/')
    else:
        endpoint_url = f'https://{endpoint_or_account}.r2.cloudflarestorage.com'

    # Initialize R2 client
    r2_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name='auto'
    )
    
    # Photos directory
    photos_dir = Path("../photos")
    if not photos_dir.exists():
        print("❌ Photos directory not found!")
        return
    
    print(f"🚀 Starting upload to R2 bucket: {BUCKET_NAME}")
    print(f"🌍 Endpoint: {endpoint_url}")
    print(f"📁 Photos directory: {photos_dir.absolute()}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(photos_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(Path(root) / file)
    
    print(f"📸 Found {len(image_files)} images to upload")
    
    # Upload with progress bar
    uploaded_count = 0
    failed_count = 0
    
    with tqdm(total=len(image_files), desc="Uploading photos") as pbar:
        for image_path in image_files:
            try:
                # Create R2 key (relative path from photos directory)
                relative_path = image_path.relative_to(photos_dir)
                r2_key = str(relative_path).replace('\\', '/')
                
                # Upload file
                with open(image_path, 'rb') as f:
                    r2_client.upload_fileobj(
                        f,
                        BUCKET_NAME,
                        r2_key,
                        ExtraArgs={'ContentType': _detect_content_type(image_path)}
                    )
                
                uploaded_count += 1
                pbar.set_postfix({
                    'Uploaded': uploaded_count,
                    'Failed': failed_count,
                    'Current': relative_path.name
                })
                
            except Exception as e:
                print(f"\n❌ Failed to upload {image_path}: {e}")
                failed_count += 1
            
            pbar.update(1)
    
    print(f"\n✅ Upload completed!")
    print(f"📊 Uploaded: {uploaded_count}")
    print(f"❌ Failed: {failed_count}")
    
    if uploaded_count > 0:
        print(f"\n📝 Add these environment variables to Railway (do not paste secrets here):")
        print("R2_ACCESS_KEY=<your access key>")
        print("R2_SECRET_KEY=<your secret key>")
        print("R2_ACCOUNT_ID=<your account id>")
        print(f"R2_BUCKET_NAME={BUCKET_NAME}")
        print("If bucket is public, public URL pattern: https://pub-<ACCOUNT_ID>.r2.dev/<BUCKET_NAME>/")

if __name__ == "__main__":
    upload_photos_to_r2()
