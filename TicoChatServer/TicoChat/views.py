from django.shortcuts import render

import os
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from openai import AzureOpenAI
import dotenv
import chromadb

dotenv.load_dotenv()

client = AzureOpenAI(
    azure_endpoint="https://hatchworksai.openai.azure.com/",
    api_key=os.environ.get('AZURE_OPENAI_KEY'),
    api_version="2023-10-01-preview",
)

deployment = "gpt-35"

if not os.environ.get('AZURE_OPENAI_KEY'):
    raise Exception("La variable de entorno 'AZURE_OPENAI_KEY' no está configurada.")

chroma_client = chromadb.PersistentClient(path="./DB/")
collection = chroma_client.get_collection(name="embeddings_collection")

def create_embeddings(text, model="ada-02"):
    embeddings = client.embeddings.create(input=text, model=model).data[0].embedding
    return embeddings

def get_recommendation(activity_or_place):
    query_vector = create_embeddings(activity_or_place)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    return [metadata["chunk"] for metadata in results["metadatas"][0]]

@require_GET
def recommendation(request):
    try:
        activity_or_place = request.GET.get('query')
        if not activity_or_place:
            return JsonResponse({"error": "Falta el parámetro 'query' en la URL"}, status=400)

        result = get_recommendation(activity_or_place)
        return JsonResponse({"recommendation": result})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
