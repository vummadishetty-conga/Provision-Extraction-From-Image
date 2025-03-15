import torch
import numpy as np
import faiss
from model import initiate_model
from pdf2image import convert_from_path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

index_file_path = "Faiss_index.bin"

model,processor,device = initiate_model()

def extract_pages_from_pdf(pdf_path):
    """Convert each page of a PDF to an image."""
    images = convert_from_path(pdf_path)
    print("Converting the pdf into images...")
    print("Number of images: {}".format(len(images)))
    return images

def embed_image(image):
    """Generate CLIP embedding for an image."""
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        embedding = model.get_image_features(**inputs).cpu().numpy()
    return embedding / np.linalg.norm(embedding)# Normalize the embedding

def build_or_load_faiss_index(page_images):
    if os.path.exists(index_file_path):
        print("Loading existing FAISS index...")
        index = faiss.read_index(index_file_path)
    else:
        print("Building FAISS index...")
        index, _ = build_faiss_index_Cosine_similarity(page_images)
        faiss.write_index(index, index_file_path)
    return index

def build_faiss_index_L2_distance(page_images):
    """Build FAISS index for page embeddings."""
    d = 512
    index = faiss.IndexFlatL2(d)
    embeddings = []

    for img in page_images:
        emb = embed_image(img)
        embeddings.append(emb)
        index.add(emb)

    print("Building faiss index... using l2 distance...")

    return index, embeddings



def build_faiss_index_Cosine_similarity(page_images):
    """Build FAISS index for page embeddings."""
    d = 768
    index = faiss.IndexFlatIP(d)  # Changed to IndexFlatIP
    embeddings = []

    for img in page_images:
        emb = embed_image(img)
        emb = emb.reshape(1, d)
        embeddings.append(emb)
        index.add(emb)

    print("Building faiss index... using cosine similarity...")

    return index, embeddings

def find_similar_page_similarity(input_img, index, page_images, top_k=3):
    """Find the top_k most similar pages to the input image."""
    matched_pages = []
    query_embedding = embed_image(input_img)
    distances, indices = index.search(query_embedding, top_k)
    print("Finding similar pages...")

    # Display results
    for i in range(top_k):
        score = distances[0][i]  # Now represents similarity
        matched_page = page_images[indices[0][i]]
        matched_pages.append(matched_page) # Store matched page and score
        print(f"Match {i + 1}: Similarity = {score:.4f}")  # Changed label
    return matched_pages



def find_similar_page_distance(input_img, index, page_images, top_k=3):
    """Find the top_k most similar pages to the input image."""
    matched_pages = []
    query_embedding = embed_image(input_img)
    distances, indices = index.search(query_embedding, top_k)
    print("Finding similar pages...")

    # Display results
    for i in range(top_k):
        score = distances[0][i]  # Get score for current match
        matched_page = page_images[indices[0][i]]
        matched_pages.append(matched_page) # Store matched page and score
        print(f"Match {i + 1}: Score = {score:.4f}") #The smaller the distance, the more similar the images are.

    return matched_pages



def split_image_horizontal(image):
    width, height = image.size
    top_half = image.crop((0, 0, width, height // 2))
    bottom_half = image.crop((0, height // 2, width, height))
    return top_half, bottom_half


def find_best_half_similarity(input_embedding, halves_embeddings):
    """Find the highest similarity between two embeddings."""
    print("Finding highest similarity...")
    best_score = -1  # Initialize with a low value
    best_half_embedding = None
    is_top_half = None
    best_page_index = None

    for i, (top_emb, bottom_emb) in enumerate(halves_embeddings):
        # Calculate cosine similarity for both halves
        top_similarity = np.dot(input_embedding, top_emb.T) / (np.linalg.norm(input_embedding) * np.linalg.norm(top_emb))
        bottom_similarity = np.dot(input_embedding, bottom_emb.T) / (np.linalg.norm(input_embedding) * np.linalg.norm(bottom_emb))

        # Check if either half has a better similarity score
        if top_similarity > best_score:
            best_score = top_similarity
            best_half_embedding = top_emb
            is_top_half = True
            best_page_index = i
        if bottom_similarity > best_score:
            best_score = bottom_similarity
            best_half_embedding = bottom_emb
            is_top_half = False
            best_page_index = i

    print("Passed to the LLM for extraction...")

    return best_half_embedding, best_score, is_top_half, best_page_index






