#!/usr/bin/env python3
"""
Script to upload wedding photos to Cloudflare R2
"""
import os
import boto3
from pathlib import Path
import sys
from tqdm import tqdm

def upload_photos_to_r2():
    """Upload all photos to Cloudflare R2"""
    
    # R2 Configuration
    R2_ACCESS_KEY = input("Enter your R2 Access Key ID: ").strip()
    R2_SECRET_KEY = input("Enter your R2 Secret Access Key: ").strip()
    R2_ACCOUNT_ID = input("Enter your R2 Account ID: ").strip()
    BUCKET_NAME = input("Enter your R2 Bucket name (default: wedding-photos-amit-guy): ").strip() or "wedding-photos-amit-guy"
    
    # Initialize R2 client
    r2_client = boto3.client(
        's3',
        endpoint_url='https://49ntv19XYJNkI_aepNY8WKA_UK8O2dQNj1aHJFcn.r2.cloudflarestorage.com',
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
    print(f"📁 Photos directory: {photos_dir.absolute()}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
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
                        ExtraArgs={'ContentType': 'image/jpeg'}
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
        print(f"\n🌐 Your photos are now available at:")
        print(f"https://pub-49ntv19XYJNkI_aepNY8WKA_UK8O2dQNj1aHJFcn.r2.dev/{BUCKET_NAME}/")
        print(f"\n📝 Add these environment variables to Railway:")
        print(f"R2_ACCESS_KEY={R2_ACCESS_KEY}")
        print(f"R2_SECRET_KEY={R2_SECRET_KEY}")
        print(f"R2_ACCOUNT_ID=49ntv19XYJNkI_aepNY8WKA_UK8O2dQNj1aHJFcn")
        print(f"R2_BUCKET_NAME={BUCKET_NAME}")

if __name__ == "__main__":
    upload_photos_to_r2()
