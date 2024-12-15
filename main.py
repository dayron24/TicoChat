import os
from flask import Flask, request, jsonify
from openai import AzureOpenAI
import dotenv
import chromadb

# Cargar variables de entorno
dotenv.load_dotenv()

# Inicializar el cliente de Azure OpenAI
client = AzureOpenAI(
    azure_endpoint="https://hatchworksai.openai.azure.com/",
    api_key=os.environ['AZURE_OPENAI_KEY'],
    api_version="2023-10-01-preview", 
)
deployment = "gpt-35"

# Función para crear embeddings
def create_embeddings(text, model="ada-02"):
    embeddings = client.embeddings.create(input=text, model=model).data[0].embedding
    return embeddings

# Inicializar el cliente de ChromaDB
chroma_client = chromadb.PersistentClient(path="./DB/")

# Nombre de la colección
collection_name = "embeddings_collection"

# Obtener la colección
collection = chroma_client.get_collection(name=collection_name)

# Función para obtener recomendaciones
def get_recommendation(activity_or_place):
    query_vector = create_embeddings(activity_or_place)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    result_query = []
    for metadata in results["metadatas"][0]:
        result_query.append(metadata["chunk"])
    return result_query

# Crear la aplicación Flask
app = Flask(__name__)

# Endpoint para obtener recomendaciones con parámetro en la URL
@app.route('/recommendation', methods=['GET'])
def recommendation():
    try:
        # Obtener el parámetro 'query' de la URL
        activity_or_place = request.args.get('query')

        if not activity_or_place:
            return jsonify({"error": "Falta el parámetro 'query' en la URL"}), 400

        # Obtener la recomendación
        result = get_recommendation(activity_or_place)

        return jsonify({"recommendation": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Correr el servidor Flask
if __name__ == '__main__':
    app.run(debug=True)
