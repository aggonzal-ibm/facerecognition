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
        self.app.prepare(ctx_id=0)

        self.index_name = os.getenv("OPENSEARCH_INDEX", "face_embeddings")
        self.client = OpenSearch(
            hosts=[{
                "host": os.getenv("OPENSEARCH_HOST", "localhost"),
                "port": int(os.getenv("OPENSEARCH_PORT", 9200))
            }],
            http_auth=(os.getenv("OPENSEARCH_USER", "admin"), os.getenv("OPENSEARCH_PASSWORD", "admin")),
            use_ssl=False,
            verify_certs=False,
            connection_class=RequestsHttpConnection
        )
        self._create_index()

    def _create_index(self):
        if self.client.indices.exists(index=self.index_name):
            print(f"[INFO] 🔄 Índice '{self.index_name}' ya existe. Eliminando para recrear con nueva configuración...")
            self.client.indices.delete(index=self.index_name)

        print(f"[INFO] ✅ Creando índice '{self.index_name}' con dimensión 128...")
        self.client.indices.create(index=self.index_name, body={
            "settings": {
                "index": {
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 128,
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
        print(f"[DEBUG] Registrando: {name} - Imagen: {image_path}")
        img = cv2.imread(image_path)
        faces = self.app.get(img)

        if not faces:
            print("[DEBUG] ❌ No se detectó ningún rostro.")
            return False

        embedding = faces[0].embedding
        if embedding is None:
            print("[DEBUG] ❌ No se generó embedding.")
            return False

        # ⚠️ Reducción a 128 dimensiones
        embedding = embedding[:128]

        # Inserta en OpenSearch
        self.client.index(index=self.index_name, body={
            "name": name,
            "embedding": embedding.tolist()
        })

        # Guarda imagen en CSV
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([{"name": name, "image_path": image_path}])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        return True


    def search_face(self, image_path: str) -> str:
        img = cv2.imread(image_path)
        faces = self.app.get(img)
        if not faces:
            return "No face detected"
        embedding = faces[0].embedding
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
            return res["hits"]["hits"][0]["_source"]["name"]
        return "Unknown"

    def search_embedding(self, embedding: list) -> str:
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
            name = hit["_source"]["name"]
            score = hit["_score"]  # Normalmente en [0.0 - 1.0] para cosine similarity
            return {"name": name, "score": round(score * 100, 2)}  # convierte a porcentaje
        return {"name": "Unknown", "score": 0.0}