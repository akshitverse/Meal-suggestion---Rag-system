import os
import re
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
import requests

# ---------------------------
# 1. Helper functions
def iso_to_minutes(iso_str):
    if pd.isna(iso_str):
        return 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", str(iso_str))
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    return hours * 60 + minutes

def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = re.sub(r'c\(|\)|["]', '', text)
    return text.strip()

def estimate_calories(protein, carbs, fat):
    return int(protein*4 + carbs*4 + fat*9)

substitute_map = {
    "peanut": "almond",
    "soy dairy firm tofu": "tempeh",
    "vegan mayonnaise": "avocado",
    "butter": "coconut oil",
    "milk": "soy milk"
}

# ---------------------------
# 2. Embeddings
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].numpy().tolist()

# ---------------------------
def connect_pinecone(api_key, env, index_name="recipes-hf-index", dimension=384):
    if not api_key:
        raise ValueError("Pinecone API key not provided!")

    # Create Pinecone client instance
    pc = Pinecone(api_key=api_key, environment=env)

    # List existing indexes
    existing_indexes = pc.list_indexes().names()

    # Create index if it doesn't exist
    if index_name not in existing_indexes:
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except Exception as e:
            if "ALREADY_EXISTS" not in str(e):
                raise e

    # Connect to the index
    index = pc.Index(index_name)
    return index

# ---------------------------
# 4. Format recipe metadata
def format_recipe(metadata):
    output = f"Recipe Name: {metadata.get('name','Unknown')}\n"
    protein = int(round(metadata.get('protein',0)))
    carbs = int(round(metadata.get('carbs',0)))
    fat = int(round(metadata.get('fat',0)))
    calories = metadata.get('calories') or estimate_calories(protein, carbs, fat)
    output += f"Macros: Protein {protein}g | Carbs {carbs}g | Fat {fat}g | Calories {calories} kcal\n"
    prep_time = metadata.get('prep_time') or 0
    cook_time = metadata.get('cook_time') or 0
    total_time = metadata.get('total_time') or 0
    output += f"Prep Time: {prep_time} min | Cook Time: {cook_time} min | Total Time: {total_time} min\n\n"

    ingredients = clean_text(metadata.get('ingredients',''))
    quantities = clean_text(metadata.get('quantities',''))
    substitutes = []
    if ingredients:
        ing_list = [i.strip() for i in re.split(r',|\n', ingredients) if i.strip()]
        qty_list = [q.strip() for q in re.split(r',|\n', quantities) if q.strip()]
        output += "Ingredients:\n"
        for i, q in zip(ing_list, qty_list):
            output += f"- {q} {i}\n"
            if i.lower() in substitute_map:
                substitutes.append(f"{i} → {substitute_map[i.lower()]}")
    else:
        output += "Ingredients: Not listed\n"

    if substitutes:
        output += "\nSuggested Substitutes:\n"
        for s in substitutes:
            output += f"- {s}\n"

    instructions = clean_text(metadata.get('instructions',''))
    if instructions:
        instr_list = [s.strip() for s in re.split(r'\.|\n', instructions) if s.strip()]
        output += "\nInstructions:\n"
        for idx, step in enumerate(instr_list,1):
            output += f"{idx}. {step}\n"
    else:
        output += "\nInstructions: Not provided\n"

    output += "\n" + "-"*50 + "\n"
    return output

# ---------------------------
# 5. RAG query
def query_rag(user_query, tokenizer, model, index, top_k=5):
    embedding = get_embedding(user_query, tokenizer, model)
    response = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    results = [format_recipe(match['metadata']) for match in response['matches']]
    return results

# ---------------------------
# 6. Web search
def web_search(query, api_key=None, num_results=5):
    # You can integrate SerpAPI/Brave here
    return []

# ---------------------------
# 7. Hybrid query
def hybrid_query(user_query, tokenizer, model, index, top_k=5, web_api_key=None):
    results = query_rag(user_query, tokenizer, model, index, top_k=top_k)
    if web_api_key:
        web_links = web_search(user_query, api_key=web_api_key, num_results=top_k)
        results += [f"Web Recipe: {link}" for link in web_links]
    return results
