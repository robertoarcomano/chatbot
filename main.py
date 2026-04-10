from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import subprocess

def ensure_model(model="llama3"):
    try:
        # verifica se esiste
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if model not in result.stdout:
            print(f"Scarico modello {model}...")
            subprocess.run(["ollama", "pull", model], check=True)
    except Exception as e:
        print("Errore Ollama:", e)

ensure_model()

# 1. Embeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Testo (meglio chunkarlo nella realtà)
texts = []
texts.append("Dopo domani si potrà probabilmente andare al mare")
texts.append("Domani potrebbe piovere")
texts.append("Oggi è una bella giornata")
texts.append("Ieri era nuvolo")
texts.append("L'altroieri pioveva")
texts.append("3 giorni fa nevicava")

# 3. Vector DB
db = Chroma.from_texts(texts=texts, embedding=emb)

# 4. Query
query = "Com'è il tempo oggi?"
docs = db.similarity_search(query, k=1)

# 5. LLM locale
llm = Ollama(model="llama3")

# 6. Risposta
response = llm.invoke(docs[0].page_content)

print(response)