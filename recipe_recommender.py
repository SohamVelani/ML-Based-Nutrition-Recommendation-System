import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
import numpy as np

class RecipeRecommender:
    def __init__(self, dataset_path='Food.csv'):
        """
        Initializes the recommender by loading and preparing the data and vectorizer.
        """
        self.df = self._load_and_preprocess_data(dataset_path)
        self.tfidf_matrix, self.tfidf_vectorizer = self._vectorize_text()

    @st.cache_data(show_spinner=False)
    def _load_and_preprocess_data(_self, dataset_path):
        """
        Loads a sample of the necessary columns from Food.csv, cleans it,
        and separates it into Veg/Non-Veg.
        """
        columns_to_load = [
            'Name', 
            'Keywords', 
            'RecipeIngredientParts', 
            'Calories', 
            'FatContent', 
            'CarbohydrateContent', 
            'ProteinContent',
            'RecipeInstructions'
        ]
        
        df = pd.read_csv(dataset_path, usecols=columns_to_load)
        
        # Handle potential missing values
        df['RecipeIngredientParts'] = df['RecipeIngredientParts'].fillna('')
        df['RecipeInstructions'] = df['RecipeInstructions'].fillna('')
        df['Keywords'] = df['Keywords'].fillna('')
        df['Name'] = df['Name'].fillna('')
        
        # Ensure numeric columns are clean and have a default value of 0
        numeric_cols = ['Calories', 'FatContent', 'CarbohydrateContent', 'ProteinContent']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # ==================================================================
        # ✨ NEW DATA CLEANING SECTION ✨
        # ==================================================================
        print(f"Original dataset size: {len(df)}")

        # 1. Filter out recipes with no nutritional value (e.g., < 50 calories)
        df = df[df['Calories'] >= 50].copy()
        print(f"After calorie filter (>50): {len(df)}")

        # 2. Filter out recipes with no real instructions
        # (Remove rows where instructions are empty or less than 10 chars)
        df = df[df['RecipeInstructions'].str.len() > 10].copy()
        print(f"After instruction filter: {len(df)}")

        # 3. Filter out "recipes" with too few ingredients (e.g., < 3)
        # We count ingredients by counting the commas in the string list
        df['ingredient_count'] = df['RecipeIngredientParts'].str.count(',') + 1
        df = df[df['ingredient_count'] >= 3].copy()
        print(f"After ingredient filter (>=3): {len(df)}")
        
        # Drop the helper column
        df = df.drop(columns=['ingredient_count'])
        # ==================================================================
        
        # Create a combined 'tags' column for better text-based matching
        df['tags'] = df['Name'] + ' ' + df['Keywords'] + ' ' + df['RecipeIngredientParts']
        df['tags'] = df['tags'].str.lower()
        
        # Define comprehensive lists of non-vegetarian indicators
        non_veg_indicators = [
            # ... (your existing list is fine, no changes needed) ...
            'chicken', 'beef', 'pork', 'lamb', 'mutton', 'turkey', 'duck', 'goose', 'ham', 
            'bacon', 'sausage', 'pepperoni', 'salami', 'chorizo', 'prosciutto', 'pastrami',
            'ground beef', 'ground turkey', 'ground chicken', 'ground pork', 'meatball',
            'steak', 'chop', 'roast', 'ribs', 'wing', 'thigh', 'breast', 'drumstick',
            'veal', 'venison', 'rabbit', 'bison', 'elk',
            'fish', 'salmon', 'tuna', 'cod', 'tilapia', 'mahi', 'halibut', 'trout', 
            'bass', 'snapper', 'flounder', 'sole', 'sardine', 'anchovy', 'mackerel',
            'shrimp', 'prawn', 'lobster', 'crab', 'scallop', 'oyster', 'mussel', 
            'clam', 'squid', 'octopus', 'calamari', 'crawfish', 'crayfish',
            'seafood', 'shellfish', 'crustacean',
            'gelatin', 'lard', 'tallow', 'suet', 'bone broth', 'chicken broth', 'beef broth',
            'fish sauce', 'oyster sauce', 'worcestershire', 'anchovy paste', 'fish stock',
            'meat extract', 'bouillon', 'consomme'
        ]
        
        # Create a function to check if a recipe is vegetarian
        def is_vegetarian(row):
            # ... (this function is fine, no changes needed) ...
            text_to_check = (
                str(row['Name']) + ' ' + 
                str(row['Keywords']) + ' ' + 
                str(row['RecipeIngredientParts']) + ' ' + 
                str(row['RecipeInstructions'])
            ).lower()
            
            for indicator in non_veg_indicators:
                if indicator in text_to_check:
                    return False
            return True
        
        # Apply vegetarian classification
        df['is_vegetarian'] = df.apply(is_vegetarian, axis=1)
        
        # Separate vegetarian and non-vegetarian recipes
        veg_df = df[df['is_vegetarian'] == True].copy()
        non_veg_df = df[df['is_vegetarian'] == False].copy()
        
        # Sample 25,000 from each category if available
        veg_sample_size = min(25000, len(veg_df))
        non_veg_sample_size = min(25000, len(non_veg_df))
        
        if len(veg_df) > 0:
            veg_sample = veg_df.sample(n=veg_sample_size, random_state=42)
        else:
            veg_sample = pd.DataFrame()
            
        if len(non_veg_df) > 0:
            non_veg_sample = non_veg_df.sample(n=non_veg_sample_size, random_state=43)
        else:
            non_veg__sample = pd.DataFrame() # Fixed a typo here
        
        # Combine the samples
        final_df = pd.concat([veg_sample, non_veg_sample], ignore_index=True)
        
        # Shuffle the final dataset
        final_df = final_df.sample(frac=1, random_state=44).reset_index(drop=True)
        
        print(f"Cleaned dataset loaded: {len(veg_sample)} vegetarian, {len(non_veg_sample)} non-vegetarian")
        
        return final_df
    
    @st.cache_resource(show_spinner=False)
    def _vectorize_text(_self):
        """
        Converts the text 'tags' into numerical TF-IDF vectors.
        This is computationally expensive, so it's cached to run only once.
        """
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(_self.df['tags'])
        return tfidf_matrix, tfidf

    def get_recommendations(self, user_query, target_calories, dietary_focus=None, diet_type='Non-Vegetarian', num_recommendations=5,allergies: list = None):
        if diet_type == 'Vegetarian':
            diet_filtered_df = self.df[self.df['is_vegetarian'] == True].copy()
        else:
            diet_filtered_df = self.df.copy()
            
        if diet_filtered_df.empty:
            print(f"No recipes found for diet type: {diet_type}")
            return pd.DataFrame()
        
        if allergies and 'None' not in allergies:
            # Create a regex pattern to find any of the allergies.
            # E.g., 'peanut|gluten|egg'
            # We use \b for word boundaries to avoid matching 'egg' in 'veggie'
            pattern = '|'.join([f"\\b{re.escape(allergy.lower())}\\b" for allergy in allergies if allergy])
            
            if pattern:
                # Find rows where the 'tags' column (which contains ingredients)
                # does NOT contain any of the allergy keywords.
                mask = ~diet_filtered_df['tags'].str.contains(pattern, case=False, na=False)
                pre_filter_count = len(diet_filtered_df)
                diet_filtered_df = diet_filtered_df[mask]
                
                print(f"Allergy filter removed {pre_filter_count - len(diet_filtered_df)} recipes based on: {pattern}")
        if diet_filtered_df.empty:
            print(f"No recipes found after applying allergy filter: {allergies}")
            return pd.DataFrame()
        # 2. Filter by nutritional needs
        calorie_window = 75  # Set the window to +/- 75 calories as requested
        nutritional_filtered_df = diet_filtered_df[
            (diet_filtered_df['Calories'] >= target_calories - calorie_window) &
            (diet_filtered_df['Calories'] <= target_calories + calorie_window)
        ].copy()

        # If no recipes in calorie range, expand the window to find something
        if nutritional_filtered_df.empty:
            calorie_window = 150 # Expanded fallback window
            nutritional_filtered_df = diet_filtered_df[
                (diet_filtered_df['Calories'] >= target_calories - calorie_window) &
                (diet_filtered_df['Calories'] <= target_calories + calorie_window)
            ].copy()

        if nutritional_filtered_df.empty:
            nutritional_filtered_df = diet_filtered_df.copy()

        # ✨ 3. APPLY DIETARY FOCUS SORTING (NOW INCLUDES HIGH-PROTEIN) ✨
        if dietary_focus:
            if dietary_focus == 'low-carb':
                nutritional_filtered_df = nutritional_filtered_df.sort_values(by='CarbohydrateContent', ascending=True)
            elif dietary_focus == 'high-protein':
                nutritional_filtered_df = nutritional_filtered_df.sort_values(by='ProteinContent', ascending=False)
            elif dietary_focus == 'low-fat':
                nutritional_filtered_df = nutritional_filtered_df.sort_values(by='FatContent', ascending=True)

        # 4. Use the ML model on the filtered recipes
        # Get the original indices of the rows that passed all filters
        filtered_indices = nutritional_filtered_df.index
        
        if len(filtered_indices) == 0:
            return pd.DataFrame() # Return empty if no recipes match filters

        # Select only the corresponding vectors from the full TF-IDF matrix
        filtered_tfidf_matrix = self.tfidf_matrix[filtered_indices]

        # 5. Transform the user's query text into a vector
        query_vector = self.tfidf_vectorizer.transform([user_query.lower()])

        # 6. Calculate similarity between the user's query vector and the filtered recipe vectors
        cosine_similarities = cosine_similarity(query_vector, filtered_tfidf_matrix).flatten()

        # 7. Get the top N most similar recipes
        nutritional_filtered_df['similarity'] = cosine_similarities
        
        # Sort by similarity first, then by calories (closer to target)
        nutritional_filtered_df['calorie_diff'] = abs(nutritional_filtered_df['Calories'] - target_calories)
        
        # Weight similarity higher than calorie difference
        top_recipes = nutritional_filtered_df.sort_values(
            by=['similarity', 'calorie_diff'], 
            ascending=[False, True]
        ).head(num_recommendations)

        return top_recipes

    def get_recipe_stats(self):
        """
        Returns statistics about the loaded recipe dataset.
        """
        total_recipes = len(self.df)
        veg_recipes = len(self.df[self.df['is_vegetarian'] == True])
        non_veg_recipes = len(self.df[self.df['is_vegetarian'] == False])
        
        return {
            'total_recipes': total_recipes,
            'vegetarian_recipes': veg_recipes,
            'non_vegetarian_recipes': non_veg_recipes,
            'avg_calories': self.df['Calories'].mean(),
            'avg_protein': self.df['ProteinContent'].mean(),
            'avg_carbs': self.df['CarbohydrateContent'].mean(),
            'avg_fat': self.df['FatContent'].mean()
        }