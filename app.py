import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from rag_logic import connect_pinecone, hybrid_query

# ---------------------------
load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# ---------------------------
# Load embedding model
with open("embedding_model.pkl", "rb") as f:
    tokenizer, model = pickle.load(f)

# ---------------------------
# Connect Pinecone
index = connect_pinecone(pinecone_key, pinecone_env)

# ---------------------------
st.set_page_config(page_title="Hybrid RAG Meal Planner", layout="wide")
st.title("Meal Planner")

with st.form("meal_form"):
    goal = st.selectbox("Select your goal:", ["Weight Loss", "Weight Gain", "Maintainance diet"])
    num_meals = st.slider("Number of meals per day:", 1, 6, value=3)
    allergy_choice = st.selectbox("Select allergy:", ["No Applicable", "Peanuts", "Gluten", "Other"])
    allergy_input = ""
    if allergy_choice == "Other":
        allergy_input = st.text_input("Specify your allergy:")
    diet_type = st.radio("Select your diet type:", ["Vegetarian", "Non-Vegetarian", "Vegan"])
    culture_region = st.text_input("Enter culture / regional cuisine:", placeholder="e.g., Indian, Mediterranean, Mexican")

    # Macronutrient suggestion
    if goal == "Weight Loss":
        suggested_protein, suggested_carbs, suggested_fats = 40, 30, 30
    elif goal == "Weight Gain":
        suggested_protein, suggested_carbs, suggested_fats = 25, 50, 25
    else:
        suggested_protein, suggested_carbs, suggested_fats = 30, 40, 30

    st.subheader("Macronutrient Distribution (%)")
    col1, col2, col3 = st.columns(3)
    with col1:
        protein = st.number_input("Protein (%)", 0, 100, suggested_protein)
    with col2:
        carbs = st.number_input("Carbs (%)", 0, 100, suggested_carbs)
    with col3:
        fats = st.number_input("Fats (%)", 0, 100, suggested_fats)

    submitted = st.form_submit_button("Get Meal Plan")

if submitted:
    allergy_text = allergy_input if allergy_choice=="Other" else allergy_choice
    user_query = f"Goal: {goal}, Meals: {num_meals}, Allergies: {allergy_text}, Diet: {diet_type}, Culture: {culture_region}, Macros: Protein {protein}% Carbs {carbs}% Fat {fats}%"
    results = hybrid_query(user_query, tokenizer, model, index, top_k=num_meals)
    if results:
        with st.expander("Meal Plan Results"):
            for r in results:
                st.text(r)
    else:
        st.info("No recipes found. Try adjusting your preferences.")
