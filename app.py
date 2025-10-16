import os
from flask import Flask, request, render_template
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
from chromadb.config import Settings
from chromadb import PersistentClient

# === CONFIGURAÇÃO DO LM STUDIO ===
LM_API_URL = "http://localhost:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "llhama 3"

# === FLASK APP ===
app = Flask(__name__)

# === EMBEDDINGS E BANCO VETORIAL ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = PersistentClient(path="./chroma_db")

if "docs" in chroma_client.list_collections():
    chroma_client.delete_collection("docs")
collection = chroma_client.get_or_create_collection("docs")

# === INDEXAÇÃO DE PDFs ===
def indexar_pdfs(pasta="documentos"):
    for nome_arquivo in os.listdir(pasta):
        if nome_arquivo.endswith(".pdf"):
            caminho = os.path.join(pasta, nome_arquivo)
            reader = PdfReader(caminho)
            texto = ""
            for page in reader.pages:
                texto += page.extract_text() + "\n"

            partes = [texto[i:i+500] for i in range(0, len(texto), 500)]
            embeddings = embedding_model.encode(partes).tolist()

            for i, chunk in enumerate(partes):
                collection.add(
                    documents=[chunk],
                    embeddings=[embeddings[i]],
                    ids=[f"{nome_arquivo}_{i}"]
                )

# === BUSCA VETORIAL ===
def buscar_contexto(pergunta, k=3):
    emb = embedding_model.encode([pergunta])[0].tolist()
    resultados = collection.query(query_embeddings=[emb], n_results=k)
    trechos = resultados['documents'][0]
    return "\n\n".join(trechos)

# === CHAMADA PARA LM STUDIO ===
def perguntar_para_ia(pergunta):
    contexto = buscar_contexto(pergunta)
    prompt = f"""
Responda com base nos documentos abaixo:

Você é um assistente de IA treinado para fornecer respostas com base em informações extraídas de documentos. Sua tarefa é responder perguntas de maneira clara e útil, utilizando apenas os dados disponíveis nos documentos abaixo. No entanto, **você deve garantir que nenhum dado sensível, confidencial ou pessoal seja revelado**.

Aqui estão as diretrizes que você deve seguir:
1. Não forneça informações pessoais como nomes, endereços, números de CPF, telefones ou qualquer dado pessoal identificado.
2. Apenas use informações gerais ou factuais que não envolvam dados privados.
3. Se a pergunta envolver informações sensíveis que você não pode acessar, forneça uma resposta genérica ou avise que não é possível fornecer uma resposta.
4. Seja sempre claro e objetivo.

{contexto}

Pergunta: {pergunta}
Resposta:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(LM_API_URL, headers=HEADERS, json=payload)
        resposta = response.json()["choices"][0]["message"]["content"]
        return resposta.strip()
    except Exception as e:
        return f"Erro ao consultar LM Studio: {e}"

# === ROTAS DO FLASK ===
@app.route("/", methods=["GET", "POST"])
def index():
    resposta = ""
    if request.method == "POST":
        pergunta = request.form["pergunta"]
        resposta = perguntar_para_ia(pergunta)
    return render_template("index.html", resposta=resposta)

# === INICIALIZAÇÃO ===
if __name__ == "__main__":
    print("Indexando PDFs...")
    indexar_pdfs()
    print("Pronto! Acesse: http://localhost:5000")
    app.run(debug=True)