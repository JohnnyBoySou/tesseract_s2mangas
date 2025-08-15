from fastapi import FastAPI, UploadFile, File
from typing import List
import os, io, httpx
from PIL import Image
import pytesseract
from statistics import mean

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL = os.getenv("MODEL_NAME", "qwen2.5:3b")

app = FastAPI()

def ocr_lang(im, lang):
    data = pytesseract.image_to_data(im, lang=lang, output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in data.get("conf", []) if c not in ("-1","",None)]
    txt = " ".join([w for w in data.get("text", []) if w and w.strip()])
    return txt, (mean(confs) if confs else 0)

def best_ocr(im):
    t1, c1 = ocr_lang(im, "jpn")
    t2, c2 = ocr_lang(im, "jpn_vert")
    return (t1, c1) if c1 >= c2 else (t2, c2)

async def chat_ollama(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120) as cli:
        r = await cli.post(f"{OLLAMA_URL}/api/chat", json={
            "model": MODEL,
            "messages": [
                {"role":"system","content":"Você resume capítulos de mangá em PT-BR, de forma objetiva."},
                {"role":"user","content": prompt}
            ],
            "options":{"temperature":0.4}
        })
        r.raise_for_status()
        return r.json()["message"]["content"]

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@app.post("/summarize")
async def summarize(pages: List[UploadFile] = File(...)):
    page_texts = []
    for f in pages:
        raw = await f.read()
        im = Image.open(io.BytesIO(raw)).convert("L")
        w, h = im.size
        if max(w,h) < 1600:
            r = 1600 / max(w,h)
            im = im.resize((int(w*r), int(h*r)))
        text, _ = best_ocr(im)
        page_texts.append(text)

    partials = []
    for grp in chunk(page_texts, 6):
        joined = "\n\n".join(grp)[:12000]
        partials.append(await chat_ollama(f"Resuma estas páginas (ignore ruídos de OCR):\n{joined}"))
    final = await chat_ollama("Com base nestes mini-resumos, escreva um único resumo do capítulo:\n" + "\n\n".join(partials))
    return {"summary": final}
