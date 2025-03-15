from pipeline import *
from PIL import Image
from extract import *
import base64
from io import BytesIO

def create_faiss_db(pdf_path):
    document_pages = extract_pages_from_pdf(pdf_path)
    for i in document_pages:
        print(i)

    print("Embedding are being saved")

    index  = build_or_load_faiss_index(document_pages)

    return document_pages, index

def test_extract(document_pages, index):
    query_image = Image.open(query_image_path).convert("RGB")
    matched_page = find_similar_page_similarity(query_image, index, document_pages)

    all_halves_embeddings = []
    print("Splitting Horizontally....")
    for page in matched_page:  # assuming matched_pages is a list of matched pages
        top_half, bottom_half = split_image_horizontal(page)
        top_emb = embed_image(top_half)
        bottom_emb = embed_image(bottom_half)
        all_halves_embeddings.append((top_emb, bottom_emb))

    # 3. find best half
    input_img_embedding = embed_image(query_image)
    best_half_embedding, score, is_top_half, page_index = find_best_half_similarity(input_img_embedding,
                                                                                    all_halves_embeddings)

    # 4. Display result
    best_half_page = split_image_horizontal(matched_page[page_index])[0 if is_top_half else 1]
    if score[0][0] > 0.7:  # Get the actual image
        return best_half_page
    else:
        return None




if __name__ == '__main__':
    #documents
    pdf_path = "#"  # PDF path
    query_image_path = "#" #user-uploading - image 

    #calling the pipeline
    document_pages,index = create_faiss_db(pdf_path)

    image = test_extract(document_pages, index)
    image.show()
    # Convert PIL Image to Bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Use "JPEG" if the image is in JPEG format

    # Get bytes and encode to Base64
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    extract_provisions(base64_image)






