# ==============================================================================
# FINAL CAPSTONE PROJECT: PERSONALIZED AI DIETARY ASSISTANT
# Beast Mode Version: Chat History + Personalization Logic [No Hardcoded Profile]
# ==============================================================================

# ==============================================================================
# STEP 0: SETUP & IMPORTS
# ==============================================================================
print("🚀 Starting setup...")
import torch
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re
import ast  # For safe list parsing
from database import DatabaseManager

print("✅ Imports complete!")

db = DatabaseManager()
# ==============================================================================
# STEP 1: LOAD MODELS AND DATASET
# ==============================================================================
print("\n🔄 Loading models and dataset...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model 1: Flan-T5 (for Recipe Adaptation) ---
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)
print("✅ Flan-T5 model (for recipe adaptation) loaded.")

# --- Model 2: Mistral 7B Instruct (for Q&A) ---
model_name = "Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
chat_tokenizer = AutoTokenizer.from_pretrained(f"mistralai/{model_name}")
chat_model = AutoModelForCausalLM.from_pretrained(
    f"mistralai/{model_name}",
    quantization_config=bnb_config,
    device_map="auto"
)
chat_tokenizer.pad_token = chat_tokenizer.eos_token
print(f"✅ {model_name} Chat model (for Q&A) loaded.")

# --- Model 3: MiniLM (for Embeddings) ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("✅ MiniLM embedding model loaded.")

# --- Model 4: Zero-Shot Classifier (for Intent) ---
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device.type == "cuda" else -1)
print("✅ Zero-shot classifier for intent loaded.")

# --- Load local Kaggle CSV ---
print("🔄 Loading Kaggle 'food.csv' file...")
try:
    df = pd.read_csv("food.csv", on_bad_lines='skip', engine='python')
    print("✅ Kaggle CSV loaded.")

    df = df.rename(columns={
        "Name": "title", "RecipeIngredientParts": "ingredients", "RecipeInstructions": "directions",
        "Calories": "calories", "ProteinContent": "protein_g", "FatContent": "fat_g",
        "CarbohydrateContent": "carbs_g", "SugarContent": "sugar_g", "SodiumContent": "salt_mg"
    })

    required_cols = ['title', 'ingredients', 'directions', 'calories', 'protein_g', 'fat_g', 'carbs_g', 'sugar_g', 'salt_mg']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols: raise Exception(f"Missing columns: {missing_cols}")
    print("✅ Columns renamed successfully.")

    df = df.sample(n=2000, random_state=42).reset_index(drop=True)
    print(f"✅ Demo dataset sampled. Total recipes: {len(df)}")

except FileNotFoundError:
    print("="*50 + "\n❌ ERROR: 'food.csv' not found!\nPlease upload it via the Colab sidebar.\n" + "="*50)
    raise

# ==============================================================================
# STEP 2: DATA PREPROCESSING & ROBUST EMBEDDING INDEXING
# ==============================================================================
print("\n⚙️  Processing data, creating embeddings, and building a robust FAISS index...")

for col in ['title', 'ingredients', 'directions']: df[col] = df[col].astype(str).fillna('')
df['salt_mg'] = df['salt_mg'].fillna(0)
df['sugar_g'] = df['sugar_g'].fillna(0)
df.dropna(subset=['title', 'calories', 'protein_g', 'fat_g', 'carbs_g'], inplace=True)
df = df.reset_index(drop=True)
df['combined_text'] = df['title'] + ". Ingredients: " + df['ingredients'] + ". Directions: " + df['directions']

recipe_embeddings = embedding_model.encode(df['combined_text'].tolist(), convert_to_tensor=True, show_progress_bar=True)
recipe_embeddings = F.normalize(recipe_embeddings, p=2, dim=1)
embedding_dim = recipe_embeddings.shape[1]
recipe_embeddings_np = recipe_embeddings.cpu().numpy().astype('float32')
faiss_index = faiss.IndexFlatIP(embedding_dim)
faiss_index.add(recipe_embeddings_np)
print(f"✅ FAISS index (IndexFlatIP) built for {faiss_index.ntotal} recipes.")

MEAL_TYPE_TAGS = ["breakfast", "brunch", "lunch", "dinner", "dessert", "snack", "appetizer"]
meal_type_embeddings = F.normalize(embedding_model.encode(MEAL_TYPE_TAGS, convert_to_tensor=True), p=2, dim=1)
NUTRITION_TAGS = {"high protein": {"sort_by": "protein_g", "ascending": False}, "low carb": {"sort_by": "carbs_g", "ascending": True}, "low fat": {"sort_by": "fat_g", "ascending": True}, "low calorie": {"sort_by": "calories", "ascending": True}, "keto friendly": {"sort_by": "carbs_g", "ascending": True}}
nutrition_tag_keys = list(NUTRITION_TAGS.keys())
nutrition_tag_embeddings = F.normalize(embedding_model.encode(nutrition_tag_keys, convert_to_tensor=True), p=2, dim=1)
print("✅ Semantic filter tags are ready.")


# ==============================================================================
# STEP 3: PERSONALIZATION & ADAPTATION FUNCTIONS
# ==============================================================================

def apply_user_constraints(df, user_profile):
    if not user_profile: return df # Gracefully handle empty profile
    print("   - Applying user constraint filters...")
    filtered_df = df.copy()
    if user_profile.get("allergies"):
        for allergen in user_profile["allergies"]:
            filtered_df = filtered_df[~filtered_df['ingredients'].str.contains(allergen, case=False, regex=False)]
    if user_profile.get("diabetes"):
        filtered_df = filtered_df[(filtered_df['sugar_g'] < 10) & (filtered_df['carbs_g'] < 50)]
    if user_profile.get("low_sodium"):
        max_salt_mg = user_profile.get("max_salt_mg", 500)
        filtered_df = filtered_df[filtered_df['salt_mg'] < max_salt_mg]
    print(f"   - Recipes remaining after constraint filter: {len(filtered_df)}")
    return filtered_df

def adapt_recipe(recipe, user_profile, use_t5_adaptation=False):
    if not user_profile: return recipe # Gracefully handle empty profile
    adapted_recipe = recipe.copy()

    if use_t5_adaptation and device.type == 'cuda':
        print(f"   - Adapting recipe with T5: '{recipe['title']}'")
        constraints = []
        if user_profile.get("diabetes"): constraints.append("diabetic-friendly (low sugar, low carb)")
        if user_profile.get("low_sodium"): constraints.append("low-sodium")

        # Only proceed if there are constraints to apply
        if constraints:
            prompt = f"Rewrite the following recipe to be {', '.join(constraints)}. Make intelligent ingredient substitutions and adjust the cooking instructions.\n\nOriginal Recipe:\n{recipe['combined_text']}\n\nProvide only the rewritten recipe in the format:\nRewritten Title: [title]\nRewritten Ingredients: [ingredients]\nRewritten Directions: [directions]"
            input_ids = t5_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).input_ids.to(device)
            outputs = t5_model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)
            rewritten_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

            try:
                adapted_recipe['title'] = re.search(r"Rewritten Title: (.*)", rewritten_text).group(1)
                adapted_recipe['ingredients'] = re.search(r"Rewritten Ingredients: (.*)", rewritten_text, re.DOTALL).group(1).strip()
                adapted_recipe['directions'] = re.search(r"Rewritten Directions: (.*)", rewritten_text, re.DOTALL).group(1).strip()
                adapted_recipe['T5_adapted'] = True
            except AttributeError:
                use_t5_adaptation = False # Set flag to false so simple substitution runs
                print("   - T5 parsing failed, falling back to simple substitution.")
        else:
             use_t5_adaptation = False # No constraints, skip T5, potentially use simple substitution

    # Run simple substitution if T5 wasn't used or failed, AND if there's a profile
    if not adapted_recipe.get('T5_adapted', False) and user_profile:
        print(f"   - Adapting recipe with simple substitution: '{recipe['title']}'")
        substitutions = { r'\bsugar\b': 'stevia', r'\bsalt\b': 'low-sodium salt substitute', r'\bbutter\b': 'olive oil', r'\bmilk\b': 'almond milk' }
        for old, new in substitutions.items():
            adapted_recipe['ingredients'] = re.sub(old, new, adapted_recipe['ingredients'], flags=re.IGNORECASE)
            adapted_recipe['directions'] = re.sub(old, new, adapted_recipe['directions'], flags=re.IGNORECASE)
        if not adapted_recipe['title'].endswith(" (Personalized)"): # Avoid double-adding
             adapted_recipe['title'] += " (Personalized)"
        adapted_recipe['T5_adapted'] = False # Explicitly mark as not T5 adapted

    return adapted_recipe

# ==============================================================================
# STEP 4: CORE PIPELINE FUNCTIONS (WITH FINAL FIXES)
# ==============================================================================

def classify_intent(query):
    candidate_labels = ["Nutrition Question", "Recipe Search"]
    result = intent_classifier(query, candidate_labels, multi_label=False)
    return result['labels'][0].replace(" ", "_").upper()

def find_top_recipes(query_embedding, k=150):
    q_np = query_embedding.cpu().numpy().astype('float32')
    distances, indices = faiss_index.search(q_np, k)
    return df.iloc[indices[0]].copy()

def get_nutrition_advice(query, history, profile):
    # If profile is empty, use a generic system prompt
    if not profile:
        system_prompt = """You are a helpful and expert nutrition assistant.
Provide a detailed, specific, and factual answer for the following question.
List examples if possible. Do not invent facts."""
    else:
        # If profile exists, create the personalized prompt
        profile_string = json.dumps(profile)
        system_prompt = f"""You are a personalized nutrition expert.
Your client has the following health profile: {profile_string}.
All of your advice MUST be tailored to this profile.
If they ask for a food, tell them if it's good or bad *for their specific profile*.
Be detailed, specific, and factual. List examples if possible. Do not invent facts."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    #
    # 🛑 CRITICAL FIX 1: THIS LINE IS REMOVED
    # 'history' already contains the latest user query from app.py
    # messages.append({"role": "user", "content": query})
    #

    prompt = chat_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = chat_tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    outputs = chat_model.generate(
        **inputs, max_new_tokens=768, do_sample=True, temperature=0.7, top_k=50, top_p=0.95
    )
    full_response = chat_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    answer = full_response.split("[/INST]")[-1].strip()

    # DO NOT APPEND TO HISTORY. app.py handles all history management.
    return answer

def rerank_and_filter_recipes(candidate_df, query_embedding):
    filtered_df = candidate_df.copy()
    MEAL_SIMILARITY_THRESHOLD = 0.45
    meal_sims = F.cosine_similarity(query_embedding, meal_type_embeddings, dim=1)
    best_meal_score, best_meal_idx = torch.max(meal_sims, dim=0)
    if best_meal_score.item() > MEAL_SIMILARITY_THRESHOLD:
        target_meal = MEAL_TYPE_TAGS[best_meal_idx.item()]
        candidate_indices = filtered_df.index.tolist()
        candidate_embeddings = recipe_embeddings[candidate_indices]
        target_meal_embedding = meal_type_embeddings[best_meal_idx.item()].unsqueeze(0)
        title_sims = F.cosine_similarity(candidate_embeddings, target_meal_embedding, dim=1)
        filtered_df = filtered_df[title_sims.cpu().numpy() > (MEAL_SIMILARITY_THRESHOLD - 0.1)]

    NUTRITION_SIMILARITY_THRESHOLD = 0.6
    nutri_sims = F.cosine_similarity(query_embedding, nutrition_tag_embeddings, dim=1)
    detected = (nutri_sims > NUTRITION_SIMILARITY_THRESHOLD).nonzero(as_tuple=True)[0]
    if len(detected) > 0:
        for idx in detected:
            tag = nutrition_tag_keys[idx.item()]
            cfg = NUTRITION_TAGS[tag]
            filtered_df = filtered_df.sort_values(by=cfg['sort_by'], ascending=cfg['ascending'])
    return filtered_df

# ==============================================================================
# STEP 5: MAIN WORKFLOW ORCHESTRATOR
# ==============================================================================

def process_query(user_query, user_id, chat_history, use_t5_adaptation=False):
    """
    Processes a user query by fetching their profile, classifying intent,
    and returning either nutrition advice or recipes.
    """
    
    # --- 1. FETCH AND BUILD PROFILE ---
    # Fetch data from the database using the user_id
    user_profile_from_db = db.get_user_profile(user_id)
    medical_conditions = db.get_latest_medical_conditions(user_id)
    
    # Build the simple profile dictionary that the ML models expect
    user_profile_for_ml = {
        "allergies": [],
        "diabetes": False,
        "low_sodium": False
    }
    
    if user_profile_from_db:
        # Get allergies from the user_profiles table
        user_profile_for_ml["allergies"] = user_profile_from_db.get("allergies", [])
        
    if "diabetes" in medical_conditions or "prediabetes" in medical_conditions:
        user_profile_for_ml["diabetes"] = True
        
    if "high_blood_pressure" in medical_conditions:
        user_profile_for_ml["low_sodium"] = True
    
    print(f"\n▶️  Processing query: '{user_query}' for user_id: {user_id}")
    print(f"   - Built ML Profile: {user_profile_for_ml}")
    
    # --- 2. CLASSIFY INTENT ---
    user_query_original = user_query # Keep the original phrasing
    user_query_lower = user_query.lower().strip()
    
    intent = classify_intent(user_query_lower)
    print(f"   - Detected Intent: {intent}")

    # --- 3. HANDLE INTENT: NUTRITION QUESTION ---
    if intent == "NUTRITION_QUESTION":
        advice = get_nutrition_advice(user_query_original, chat_history, user_profile_for_ml)
        return {"type": "advice", "data": advice, "status": "success"}

    # --- 4. HANDLE INTENT: RECIPE SEARCH ---
    elif intent == "RECIPE_SEARCH":
        print("   - Asking Mistral to refine the search query based on history...")

        messages_for_refinement = [{"role": "system", "content": "Based on the following conversation, generate a concise and specific recipe search query (3-7 words) that captures the user's latest request. Only output the search query itself."}]
        messages_for_refinement.extend(chat_history)
        
        refinement_prompt = chat_tokenizer.apply_chat_template(messages_for_refinement, tokenize=False, add_generation_prompt=False)
        refinement_prompt += "[/INST] Search Query: " 

        refinement_inputs = chat_tokenizer(refinement_prompt, return_tensors="pt", padding=True).to(device)

        refinement_outputs = chat_model.generate(
            **refinement_inputs, max_new_tokens=20, do_sample=False, temperature=0.1
        )

        full_refinement_response = chat_tokenizer.batch_decode(refinement_outputs, skip_special_tokens=True)[0]
        refined_search_query = full_refinement_response.split("Search Query:")[-1].strip().lower()

        if not refined_search_query or len(refined_search_query) < 3:
            print("   - ⚠️ Mistral refinement failed, using original query for search.")
            search_query = user_query_lower
        else:
             print(f"   - Refined Search Query: '{refined_search_query}'")
             search_query = refined_search_query

        # Now use the (potentially refined) search_query for embedding
        query_embedding = F.normalize(embedding_model.encode([search_query], convert_to_tensor=True), p=2, dim=1)
        initial_candidates = find_top_recipes(query_embedding)
        semantically_filtered = rerank_and_filter_recipes(initial_candidates, query_embedding)
        
        # Apply user profile constraints
        user_filtered_results = apply_user_constraints(semantically_filtered, user_profile_for_ml)

        if user_filtered_results.empty:
            print("   - ⚠️ No results after user constraints. Falling back to semantic results.")
            final_candidates = semantically_filtered
            if final_candidates.empty:
                print("   - ⚠️ No results after semantic filter either. Falling back to initial search.")
                final_candidates = initial_candidates
            status = "fallback_success"
        else:
            final_candidates = user_filtered_results
            status = "success"

        top_recipes = final_candidates.head(3).to_dict(orient='records')
        
        # Adapt recipes based on user profile
        adapted_recipes = [adapt_recipe(recipe, user_profile_for_ml, use_t5_adaptation) for recipe in top_recipes]

        return {"type": "recipes", "data": adapted_recipes, "status": status}

# ==============================================================================
# STEP 6: INTERACTIVE Q&A CHAT (FOR COLAB TESTING ONLY)
# ==============================================================================

print("\n" + "="*60)
print("🤖 AI Nutrition Assistant is ready!")
print("Type your question or recipe request below. Type 'exit' to quit.")
print("="*60)

# --- MODIFIED: Use a test user ID ---
# In your real app, this ID comes from the logged-in user
# Use 1, or any other user ID you know exists in your diet_app.db
test_user_id = 1 
print(f"✅ Test loop running for user_id: {test_user_id}")
print("   (This test will fetch the profile from your database)")

chat_history = [] # Initialize empty chat history

def safe_list_parser(s):
    try: return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        if isinstance(s, str): return s.split('\n')
        return [s]

while True:
    print("\n------------------------------------------------")
    user_query = input("You: ")
    if user_query.lower() in ['exit', 'quit', 'stop']: print("🤖 Goodbye!"); break
    if not user_query: continue

    #
    # ⚠️ THIS TEST LOOP IS NOW "BROKEN" ⚠️
    # Because we are not adding to 'chat_history' in this loop,
    # the AI will NOT have memory. This is EXPECTED.
    # The real history is managed by app.py.
    #

    # We add the user query to the history *for this loop only*
    # to simulate what app.py does.
    chat_history.append({"role": "user", "content": user_query})

    use_t5 = device.type == 'cuda'

    # ==================================================================
    # ✨ BUG FIX IS HERE ✨
    # We now pass 'user_id=test_user_id' instead of 'user_profile'
    # ==================================================================
    result = process_query(
        user_query,
        user_id=test_user_id,
        chat_history=chat_history, # Pass the history
        use_t5_adaptation=use_t5
    )

    print("\n🤖 AI Assistant:")

    if result.get("type") == "advice":
        response_text = result.get('data', 'Sorry, I had trouble finding an answer.')
        print(f"\n   {response_text}")
        # Add assistant response to history *for this loop only*
        chat_history.append({"role": "assistant", "content": response_text})

    elif result.get("type") == "recipes":
        recipes = result.get('data', [])
        print(f"\n   I found {len(recipes)} recipe(s) for you:")

        recipe_summary = [f"I found {len(recipes)} recipe(s) for you:"]

        for i, recipe in enumerate(recipes):
            title = recipe.get('title', 'N/A')
            recipe_summary.append(title)

            print(f"\n   --- Recipe {i+1} ---")
            print(f"   Title: {title}")
            print("\n   Ingredients:")
            ingredients_list = safe_list_parser(recipe.get('ingredients', 'N/A'))
            for item in ingredients_list: print(f"     - {item.strip()}")
            print("\n   Directions:")
            directions_list = safe_list_parser(recipe.get('directions', 'N/A'))
            for j, step in enumerate(directions_list):
                if step.strip(): print(f"     {j+1}. {step.strip()}")

        # Add assistant response to history *for this loop only*
        chat_history.append({"role": "assistant", "content": "\n".join(recipe_summary)})

    else:
        print("\n   Sorry, I couldn't process that request.")