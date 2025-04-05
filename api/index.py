from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
from face_engine import FaceEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Asegura existencia de carpeta para imágenes
os.makedirs("images", exist_ok=True)
engine = FaceEngine()

class VectorRequest(BaseModel):
    embedding: list[float]

@app.post("/register")
async def register(name: str = Form(...), file: UploadFile = Form(...)):
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join("images", filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    success = engine.register_face(name, file_path)
    return {
        "registered": success,
        "message": "✅ Registrado correctamente" if success else "❌ No se detectó rostro"
    }

@app.post("/search")
async def search(file: UploadFile):
    import uuid, os, shutil

    # Ruta temporal para guardar la imagen
    file_path = os.path.join("images", f"{uuid.uuid4().hex}_{file.filename}")
    
    # Guardar temporalmente el archivo
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ejecutar reconocimiento facial
    result = engine.search_face(file_path)

    # Eliminar el archivo temporal
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"[WARN] No se pudo eliminar la imagen temporal: {e}")

    return result


@app.post("/search-vector")
async def search_vector(data: VectorRequest):
    return engine.search_embedding(data.embedding)

@app.get("/")
def root():
    return {"status": "🚀 Face Recognition API is running"}


@app.post("/reset")
async def reset_index():
    engine.reset_index()
    return {"message": "🧹 Índice de OpenSearch reiniciado correctamente"}image-generator-serviceimage-generator-service

# 🟢 Punto de entrada
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="127.0.0.1", port=8010, reload=True)
