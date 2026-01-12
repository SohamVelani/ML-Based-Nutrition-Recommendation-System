ML-Based Nutrition Recommendation System

Smart full-stack app delivering personalized, medically-aware diet plans and recipe recommendations based on user profile, medical reports (OCR + NLP), allergies, and goals.

Key features

Personalized meal plans & exportable diet plans generated from user profile and goals. (UI/orchestration in app.py). 
Medical report extraction (PDF/DOCX/TXT) with OCR fallback and medical NER/ontology for conditions, meds, labs, negation detection. (medical_processor.py). 
Recipe search & allergy/diet filtering using TF-IDF + semantic ranking and (optional) T5 recipe adaptation. (recipe_recommender.py & model pipeline). 
data cleaning, TF-IDF, allergy/diet filtering and recommendation logic.
Question-answering & intent classification using local LLMs / pipelines (Mistral, Flan-T5, embedding models) for both nutrition advice and recipe refinement. (QNA_ML.py). 
Persistent storage of users, profiles, medical reports and recommendation history using SQLite (DB logic in database.py). 
Authentication forms and session handling (streamlit UI helpers referenced via AuthManager in app.py — see auth.py). 

Quick start (dev)

Clone repo
git clone- (https://github.com/SohamVelani/ML-Based-Nutrition-Recommendation-System/edit/main/README.md)


Install dependencies
pip install -r requirements.txt
(see Notes below — large ML models may require GPU/CUDA and extra installs or should run on cloud GPU).

Place dataset Food.csv in repo root.

Run Streamlit app
streamlit run app.py

What’s inside (modules)

requirements.txt - all the required libraries and their installation details.
app.py — Streamlit UI, form, file upload, results dashboard, export & session management. 
medical_processor.py — OCR + spaCy patterns, ontology, lab/med extraction and negation detection. 
recipe_recommender.py — data cleaning, TF-IDF, allergy/diet filtering and recommendation logic. 
QNA_ML.py — model-loading script: embeddings, T5 adaptation, Mistral chat usage & recipe reranking. 
database.py — SQLite schema, user/profile/report saving & retrieval. 
auth.py — sign-up/login forms + validation (used by app.py).

Notes & disclaimers
## Dataset
The dataset used for training and recommendations is publicly available.
Due to size and licensing constraints, it is not included in this repository.
Dataset source: Kaggle https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews

The QNA_ML file if run on any Cloud GPU provider like Google Collab, use FASTAPI to connect to the front end of the project.  
This tool assists with nutrition suggestions. It is not a medical device — always include a visible disclaimer and require professional consultation for critical cases. (You already include disclaimers in exports.) 
Large models in QNA_ML.py require GPU, significant RAM, and extra libs (transformers, bitsandbytes, faiss, sentence-transformers). See Deployment notes.
