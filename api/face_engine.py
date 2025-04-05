import os
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from opensearchpy import OpenSearch, RequestsHttpConnection

load_dotenv()

CSV_PATH = os.path.join("images", "registry.csv")
os.makedirs("images", exist_ok=True)
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["name", "image_path"]).to_csv(CSV_PATH, index=False)

class FaceEngine:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))  # RetinaFace ideal para entornos difíciles

        self.vector_dim = int(os.getenv("VECTOR_DIMENSION", 512))
        self.index_name = os.getenv("OPENSEARCH_INDEX", "face_embeddings")

        self.client = OpenSearch(
            hosts=[{"host": os.getenv("OPENSEARCH_HOST", "localhost"), "port": int(os.getenv("OPENSEARCH_PORT", 9200))}],
            http_auth=(os.getenv("OPENSEARCH_USER", "admin"), os.getenv("OPENSEARCH_PASSWORD", "admin")),
            use_ssl=False,
            verify_certs=False,
            connection_class=RequestsHttpConnection
        )

        self._create_index()

    def _create_index(self):
        if self.client.indices.exists(index=self.index_name):
            print(f"[INFO] 🔄 Índice '{self.index_name}' ya existe. Eliminando...")
            self.client.indices.delete(index=self.index_name)

        print(f"[INFO] ✅ Creando índice '{self.index_name}' con dimensión {self.vector_dim}...")
        self.client.indices.create(index=self.index_name, body={
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.vector_dim,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib",
                            "space_type": "cosinesimil"
                        }
                    }
                }
            }
        })

    def register_face(self, name: str, image_path: str) -> bool:
        print(f"[DEBUG] 📝 Intentando registrar: {name}")
        img = cv2.imread(image_path)
        if img is None:
            print("[ERROR] ❌ No se pudo leer la imagen.")
            return False
        print(f"[DEBUG] 📸 Imagen cargada correctamente con forma: {img.shape}")

        faces = self.app.get(img)
        if not faces:
            print("[WARNING] ⚠️ No se detectó rostro. Intentando en RGB...")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img_rgb)

        if not faces:
            print("[ERROR] ❌ No se detectó rostro tras intento en RGB.")
            return False

        embedding = faces[0].embedding
        if embedding is None:
            print("[ERROR] ❌ No se generó embedding.")
            return False

        embedding = embedding[:self.vector_dim]

        self.client.index(index=self.index_name, body={
            "name": name,
            "embedding": embedding.tolist()
        })

        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([{"name": name, "image_path": image_path}])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

        return True

    def search_face(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            print("[ERROR] ❌ No se pudo leer imagen.")
            return {"name": "Error", "score": 0.0}

        faces = self.app.get(img)
        if not faces:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img_rgb)

        if not faces:
            return {"name": "No face detected", "score": 0.0}

        embedding = faces[0].embedding[:self.vector_dim]
        return self.search_embedding(embedding.tolist())

    def search_embedding(self, embedding: list) -> dict:
        if len(embedding) > self.vector_dim:
            embedding = embedding[:self.vector_dim]

        query = {
            "size": 1,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": 1
                    }
                }
            }
        }

        res = self.client.search(index=self.index_name, body=query)
        if res["hits"]["hits"]:
            hit = res["hits"]["hits"][0]
            return {
                "name": hit["_source"]["name"],
                "score": round(hit["_score"] * 100, 2)
            }

        return {"name": "Unknown", "score": 0.0}
    
    def reset_index(self):
        if self.client.indices.exists(index=self.index_name):
            print(f"[INFO] 🧹 Borrando índice '{self.index_name}'...")
            self.client.indices.delete(index=self.index_name)
        print(f"[INFO] ✅ Recreando índice '{self.index_name}'...")
        self._create_index()
