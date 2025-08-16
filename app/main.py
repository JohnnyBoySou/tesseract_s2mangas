from fastapi import FastAPI, UploadFile, File
from typing import List
import os, io, httpx
from PIL import Image
import pytesseract
from statistics import mean

# Configurações
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama.railway.internal:8080").rstrip("/")
MODEL = os.getenv("MODEL_NAME", "qwen2.5:3b")

app = FastAPI()

# --- Healthcheck ---
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as cli:
            r = await cli.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return {
                "status": "ok",
                "ollama_url": OLLAMA_URL,
                "configured_model": MODEL,
                "available_models": models
            }
    except Exception as e:
        return {
            "status": "error",
            "ollama_url": OLLAMA_URL,
            "configured_model": MODEL,
            "error": str(e)
        }

# --- Funções auxiliares de OCR ---
def ocr_with_lang(im, lang):
    data = pytesseract.image_to_data(im, lang=lang, output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in data.get("conf", []) if c not in ("-1", "", None)]
    text = " ".join([w for w in data.get("text", []) if w and w.strip()])
    return text, (mean(confs) if confs else 0)

def best_ocr(im):
    t1, c1 = ocr_with_lang(im, "jpn")
    t2, c2 = ocr_with_lang(im, "jpn_vert")
    return (t1, c1) if c1 >= c2 else (t2, c2)

async def chat_ollama(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120) as cli:
        r = await cli.post(f"{OLLAMA_URL}/api/chat", json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Você resume capítulos de mangá em PT-BR."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        })
        r.raise_for_status()
        return r.json()["message"]["content"]

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# --- Endpoint principal ---
@app.post("/summarize")
async def summarize(pages: List[UploadFile] = File(...)):
    page_texts = []
    for f in pages:
        raw = await f.read()
        im = Image.open(io.BytesIO(raw)).convert("L")
        w, h = im.size
        if max(w, h) < 1600:
            r = 1600 / max(w, h)
            im = im.resize((int(w*r), int(h*r)))
        text, _ = best_ocr(im)
        page_texts.append(text)

    # Resumo hierárquico
    partials = []
    for group in chunk_list(page_texts, 6):
        joined = "\n\n".join(group)[:12000]
        partials.append(await chat_ollama(f"Resuma estas páginas de mangá:\n{joined}"))

    final = await chat_ollama("Com base nestes mini-resumos, escreva um único resumo do capítulo:\n" + "\n\n".join(partials))
    return {"summary": final}
