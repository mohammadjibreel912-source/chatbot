from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from io import BytesIO
import zipfile
import requests
from bs4 import BeautifulSoup

app = FastAPI()

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vector_store"
STATIC_DIR = "static"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ===========================
# ربط static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ===========================
# قراءة PDF
def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

# ===========================
# تقسيم النص
def split_text(text, chunk_size=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# ===========================
# حفظ Vector Index (cosine similarity)
def save_vector_index(chunks, embeddings):
    # تطبيع embeddings
    faiss.normalize_L2(embeddings)
    if os.path.exists(f"{VECTOR_DIR}/index.faiss"):
        index = faiss.read_index(f"{VECTOR_DIR}/index.faiss")
        index.add(embeddings)
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine
        index.add(embeddings)
    faiss.write_index(index, f"{VECTOR_DIR}/index.faiss")

    if os.path.exists(f"{VECTOR_DIR}/chunks.pkl"):
        with open(f"{VECTOR_DIR}/chunks.pkl", "rb") as f:
            old_chunks = pickle.load(f)
        chunks = old_chunks + chunks
    with open(f"{VECTOR_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

# ===========================
# رفع PDF
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = read_pdf(file_path)
    chunks = split_text(text)
    embeddings = model.encode(chunks, normalize_embeddings=True)
    save_vector_index(chunks, embeddings)
    return {"message": "تم تخزين PDF في قاعدة البيانات!"}

# ===========================
# رفع ZIP يحتوي PDF
@app.post("/upload_zip")
async def upload_zip(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        return {"message": "الملف يجب أن يكون بصيغة ZIP"}

    try:
        content = await file.read()
        if not content:
            return {"message": "الملف فارغ، الرجاء رفع ملف صالح."}

        zip_io = BytesIO(content)
        with zipfile.ZipFile(zip_io) as zip_file:
            pdf_count = 0
            for name in zip_file.namelist():
                if name.endswith(".pdf"):
                    try:
                        pdf_data = zip_file.read(name)
                        if not pdf_data:
                            continue
                        reader = PdfReader(BytesIO(pdf_data))
                        text = ""
                        for page in reader.pages:
                            t = page.extract_text()
                            if t:
                                text += t + "\n"
                        if not text.strip():
                            continue
                        chunks = split_text(text)
                        embeddings = model.encode(chunks, normalize_embeddings=True)
                        save_vector_index(chunks, embeddings)
                        pdf_count += 1
                    except Exception:
                        continue

        return {"message": f"تم رفع ومعالجة {pdf_count} PDF بنجاح!"}
    except Exception as e:
        return {"message": f"حدث خطأ أثناء الرفع: {e}"}

# ===========================
# سحب بيانات من رابط ويب
@app.post("/scrape_url")
async def scrape_url(url: str = Form(...)):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        content_div = soup.find("div", {"id": "ctl00_PlaceHolderMain_ctl01_Contents"}) or soup.find("body")
        chunks = []
        if content_div:
            paragraphs = content_div.find_all(["p", "li"])
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text:
                    chunks.append(text)
        if not chunks:
            return {"message": "لم يتم العثور على بيانات صالحة في الموقع."}
        embeddings = model.encode(chunks, normalize_embeddings=True)
        save_vector_index(chunks, embeddings)
        return {"message": f"تم سحب {len(chunks)} فقرة/عنصر نصي وحفظها!"}
    except Exception as e:
        return {"message": f"حدث خطأ: {e}"}

# ===========================
# API – سؤال وجواب دقيق
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    if not os.path.exists(f"{VECTOR_DIR}/index.faiss"):
        return {"answer": "لا توجد بيانات بعد!"}

    index = faiss.read_index(f"{VECTOR_DIR}/index.faiss")
    with open(f"{VECTOR_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    q_emb = model.encode([question], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb), k=3)
    best_score = D[0][0]  # أعلى تشابه
    threshold = 0.7  # أقل من هذا = بعيد

    if best_score < threshold:
        return {"answer": "مش موجود"}

    answer = " ".join([chunks[i] for i in I[0]])
    return {"answer": answer}

# ===========================
# صفحات HTML
@app.get("/upload")
def get_upload():
    return FileResponse(os.path.join(STATIC_DIR, "upload.html"))

@app.get("/scrape")
def get_scrape():
    return FileResponse(os.path.join(STATIC_DIR, "scrape.html"))

@app.get("/chatbot")
def get_chatbot():
    return FileResponse(os.path.join(STATIC_DIR, "chat.html"))

# ===========================
# عرض البيانات المخزنة
@app.get("/view_data_api")
async def view_data_api():
    try:
        with open(f"{VECTOR_DIR}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        return JSONResponse(content={"chunks": chunks})
    except FileNotFoundError:
        return JSONResponse(content={"chunks": []})
