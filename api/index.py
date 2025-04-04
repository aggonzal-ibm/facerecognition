from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from face_engine import FaceEngine

app = FastAPI()

# Permitir todo (ideal para testing, cambia en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O pon ["http://localhost:5500"] o el dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = FaceEngine()

class VectorRequest(BaseModel):
    embedding: list[float]

@app.post("/register")
async def register(name: str = Form(...), file: UploadFile = Form(...)):
    file_location = f"images/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    success = engine.register_face(name, file_location)
    
    if not success:
        return {"registered": False, "error": "No se detectó rostro en la imagen"}
    
    return {"registered": True}

@app.post("/search")
async def search(file: UploadFile):
    file_location = f"images/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = engine.search_face(file_location)
    return {"match": result}

@app.post("/search-vector")
async def search_vector(data: VectorRequest):
    result = engine.search_embedding(data.embedding)
    return result

# ➕ Esto permite ejecutar el backend con: python index.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="127.0.0.1", port=8000, reload=True)
