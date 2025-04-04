import cv2
from insightface.app import FaceAnalysis

# Ruta absoluta o relativa a tu imagen
image_path = "C:/Users/eberm/OneDrive/Desktop/Dataset/Andy/IMG_1808.JPG"

# Inicializa InsightFace
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)

# Carga la imagen
img = cv2.imread(image_path)
if img is None:
    print(f"❌ No se pudo leer la imagen: {image_path}")
    exit()

# Detecta rostros y calcula embeddings
faces = app.get(img)

if not faces:
    print("😕 No se detectó ningún rostro en la imagen.")
else:
    print(f"✅ Se detectaron {len(faces)} rostro(s).")
    embedding = faces[0].embedding
    print(f"🧠 Embedding generado (dim={len(embedding)}):")
    print(embedding)
