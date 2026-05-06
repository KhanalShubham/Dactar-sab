import os
import argparse
from tqdm import tqdm
import faiss
import pickle
import numpy as np
from clip_integration import MedicalCLIPAnalyzer

def index_directory(image_dir, output_dir="./embeddings_cache"):
    """
    Scans a directory of images, extracts CLIP embeddings, and saves a FAISS index.
    """
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist.")
        return

    analyzer = MedicalCLIPAnalyzer(cache_dir=output_dir)
    
    # Supported extensions
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir) 
        if f.lower().endswith(valid_exts)
    ]

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Indexing {len(image_paths)} images...")
    
    embeddings = []
    metadata = []

    for path in tqdm(image_paths):
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()
            
            emb = analyzer.get_image_embedding(img_bytes)
            embeddings.append(emb[0])
            metadata.append({"path": path})
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if not embeddings:
        print("No embeddings were successfully extracted.")
        return

    # Convert to numpy array
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]

    # Create FAISS index (Inner Product for cosine similarity since vectors are normalized)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    # Save index and metadata
    index_path = os.path.join(output_dir, "medical_cases.index")
    meta_path = os.path.join(output_dir, "medical_cases.pkl")
    
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nSuccessfully indexed {len(metadata)} cases.")
    print(f"Index saved to: {index_path}")
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index historical MRI images for similarity search.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing MRI images.")
    parser.add_argument("--out", type=str, default="./embeddings_cache", help="Output directory for index.")
    
    args = parser.parse_args()
    index_directory(args.dir, args.out)
