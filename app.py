import ast 
import math
import re
import streamlit as st
from typing import List, Dict, Any
from auth import AuthManager
from database import DatabaseManager
from medical_processor import MedicalReportProcessor
from recipe_recommender import RecipeRecommender
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import requests, logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
st.set_page_config(page_title="Automatic Diet Recommendation", layout="wide")

# Initialize components
auth_manager = AuthManager()
db = DatabaseManager()
medical_processor = MedicalReportProcessor()
recipe_recommender = RecipeRecommender('Food.csv')
QNA_ML_API_URL = "https://versions-size-walked-dense.trycloudflare.com/process-query"
# Initialize session state
auth_manager.init_session_state()
logger = logging.getLogger(__name__)

# ---------------- ENHANCED STYLES ----------------
st.markdown(
    """
    <style>
    /* Enhanced page styling */
    .css-18e3th9 { padding-top: 1rem; padding-bottom: 4rem; }
    .block-container { padding-left: 2rem; padding-right: 2rem; }
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3, h4 { color: #ffffff; margin: 0; padding: 0; }
    .stSelectbox, .stNumberInput, .stRadio, .stSlider { margin-bottom: 0.4rem; }
    .generate-btn > button { background-color: #0f766e; color: white; border-radius: 6px; height: 42px; }
    .success-box { background-color:#0b6b3b; color:white; padding:12px; border-radius:6px; }
    .card { background-color: #1a1f25; padding: 16px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.5); }

    /* Enhanced medical upload section */
    .medical-upload-container {
        background: linear-gradient(135deg, #134e4a 0%, #0f766e 100%);
        padding: 24px;
        border-radius: 16px;
        margin: 20px 0;
        border: 2px solid #14b8a6;
        box-shadow: 0 6px 30px rgba(20, 184, 166, 0.3);
    }
    
    .upload-section-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #f0fdfa;
        margin-bottom: 16px;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    
    .medical-info-display {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #10b981;
        margin: 16px 0;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }

    .health-timeline-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 12px 0;
        border: 1px solid #475569;
    }

    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border: 1px solid #475569;
    }
    .metric-title {
        font-weight: bold;
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.3;
    }
    
    /* Macronutrient display */
    .macro-container {
        display: flex;
        justify-content: space-around;
        text-align: center;
        margin-top: 10px;
    }
    .macro-item {
        color: white;
    }
    .macro-label {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .macro-value {
        font-size: 1.1rem;
        font-weight: bold;
    }

    /* Alert styling */
    .health-alert {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        color: white;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
        border-left: 4px solid #ef4444;
    }
    
    .health-warning {
        background: linear-gradient(135deg, #a16207 0%, #d97706 100%);
        color: white;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
        border-left: 4px solid #f59e0b;
    }
    
    .health-good {
        background: linear-gradient(135deg, #166534 0%, #16a34a 100%);
        color: white;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
        border-left: 4px solid #22c55e;
    }

    /* Personal info containers */
    .personal-info-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.3);
        margin: 20px 0;
        border: 1px solid #475569;
    }
    .section-title {
        font-size: 1.6rem;
        font-weight: bold;
        color: #f1f5f9;
        margin-bottom: 24px;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- SESSION STATE MANAGEMENT ----------------
if "page" not in st.session_state:
    st.session_state.page = "form"

if "inputs" not in st.session_state:
    st.session_state.inputs = {}

if "allergies" not in st.session_state:
    st.session_state["allergies"] = ["None"]

# Enhanced medical report session state
if "medical_report_processed" not in st.session_state:
    st.session_state.medical_report_processed = False

if "extracted_medical_info" not in st.session_state:
    st.session_state.extracted_medical_info = {}

if "health_alerts" not in st.session_state:
    st.session_state.health_alerts = []

# NEW: Session state for meal plan management
if "meal_plan_options" not in st.session_state:
    st.session_state.meal_plan_options = {}

if "current_plan_index" not in st.session_state:
    st.session_state.current_plan_index = 0

if "current_meal_plan" not in st.session_state:
    st.session_state.current_meal_plan = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- UTILITY FUNCTIONS ----------------
def safe_list_parse(s: str) -> list:
    """
    Safely parses a string that should be a list. 
    Handles:
    1. CSV-style lists (e.g., "['item1', 'item2']")
    2. API-style newline-separated strings (e.g., "item1\nitem2")
    """
    try:
        # 1. Try to parse as a Python literal (handles CSV-style)
        parsed_list = ast.literal_eval(s)
        if isinstance(parsed_list, list):
            return parsed_list
    except (ValueError, SyntaxError, TypeError):
        # 2. If literal_eval fails, try regex for quoted items (robust CSV)
        try:
            items = re.findall(r"'(.*?)'|\"(.*?)\"", s)
            if items:
                return [item[0] or item[1] for item in items]
        except Exception:
            pass  # Fall through to the next method

    # 3. If both fail, assume it's a simple newline-separated string (API-style)
    if isinstance(s, str):
        return [line.strip() for line in s.split('\n') if line.strip()]
    
    # 4. Final fallback
    return []
    
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    return weight_kg / ((height_cm / 100) ** 2)

def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    elif 30 <= bmi < 35:
        return "Obese (Class I)"
    elif 35 <= bmi < 40:
        return "Obese (Class II)"
    else:
        return "Obese (Class III)"

def calculate_bmr(gender: str, weight: float, height: float, age: int) -> float:
    if gender == "Male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def activity_multiplier(idx: int) -> float:
    return [1.2, 1.375, 1.55, 1.725, 1.9][idx]

def derive_calorie_plan(bmr: float, activity_idx: int) -> Dict[str, float]:
    maint = bmr * activity_multiplier(activity_idx)
    return {
        "Maintain weight": maint,
        "Mild weight loss": max(1200, maint - 250),
        "Weight loss": max(1000, maint - 500),
        "Extreme weight loss": max(800, maint - 1000),
    }

def calculate_days_to_goal(current_weight: float, target_weight: float, plan: str, bmr: float, activity_idx: int) -> tuple:
    plan_map = derive_calorie_plan(bmr, activity_idx)
    maintenance = float(plan_map["Maintain weight"])
    plan_cals = float(plan_map.get(plan, maintenance))
    
    daily_change = int(round(plan_cals - maintenance))
    weight_diff = float(target_weight) - float(current_weight)
    
    if abs(weight_diff) < 1e-6:
        return 0, daily_change
    
    if daily_change == 0:
        return 0, daily_change
    
    if (weight_diff < 0 and daily_change >= 0) or (weight_diff > 0 and daily_change <= 0):
        return 0, daily_change
    
    total_cal_needed = abs(weight_diff) * 7700
    days_needed = math.ceil(total_cal_needed / abs(daily_change))
    return int(days_needed), int(daily_change)

def call_qna_ml_api(query: str, user_id: int, history: list) -> dict:
    """
    Calls the external QNA_ML FastAPI with the user's query,
    user ID, and current chat history.
    """
    api_url = QNA_ML_API_URL 
    
    payload = {
        "user_query": query,
        "user_id": user_id,
        "chat_history": history
    }
    
    # We REMOVED the 'headers = ...' part because it's not needed for Cloudflare
    
    try:
        # We REMOVED 'headers=headers' from the request
        response = requests.post(
            api_url, 
            json=payload, 
            timeout=120, 
            verify=False  # Keep this, it doesn't hurt
        )
        
        response.raise_for_status() # This will check for errors
        return response.json()
        
    except requests.exceptions.Timeout:
        st.error("The AI assistant took too long to respond. Please try again.")
        return {"type": "error", "data": "Request timed out."}
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to AI assistant: {e}")
        return {"type": "error", "data": "Could not connect to the AI service."}
    
# ---------------- ENHANCED MEDICAL REPORT PROCESSING ----------------
def process_medical_report(uploaded_file, current_user):
    """Enhanced medical report processing with better error handling and health alerts"""
    if uploaded_file is not None:
        with st.spinner("Processing medical report with advanced NLP..."):
            try:
                # Extract text from file
                extracted_text = medical_processor.extract_text_from_file(uploaded_file)
                
                if extracted_text:
                    st.info(f"Successfully extracted {len(extracted_text)} characters from {uploaded_file.name}")
                    
                    # Extract medical information using enhanced processor
                    medical_info = medical_processor.extract_medical_info(extracted_text)
                    
                    if 'error' in medical_info:
                        st.error(f"Medical processing error: {medical_info['error']}")
                        return None, None, None, None
                    
                    # Generate comprehensive summary
                    summary = medical_processor.format_medical_summary(medical_info)
                    
                    # Get diet-relevant conditions
                    diet_conditions = medical_processor.get_diet_relevant_conditions(medical_info)
                    
                    # Get detailed health profile
                    health_profile = medical_processor.get_detailed_health_profile(medical_info)
                    
                    # Generate health alerts based on findings
                    health_alerts = generate_health_alerts(medical_info, health_profile)
                    st.session_state.health_alerts = health_alerts
                    
                    # Calculate confidence score based on data quality
                    confidence_score = calculate_processing_confidence(medical_info)
                    
                    # Save to database with enhanced data
                    file_type = uploaded_file.name.split('.')[-1].upper()
                    processing_method = "enhanced_nlp_with_ocr" if len(extracted_text) > 1000 else "basic_nlp"
                    
                    report_id = db.save_medical_report(
                        current_user['id'],
                        uploaded_file.name,
                        file_type,
                        medical_info,
                        summary,
                        diet_conditions,
                        health_profile,
                        processing_method,
                        confidence_score
                    )
                    
                    if report_id > 0:
                        st.session_state.extracted_medical_info = medical_info
                        st.session_state.medical_report_processed = True
                        
                        # Show processing results
                        show_processing_results(medical_info, confidence_score, health_alerts)
                        
                        return medical_info, summary, diet_conditions, health_profile
                    else:
                        st.error("Failed to save medical report to database.")
                        return None, None, None, None
                else:
                    st.error("Failed to extract text from the uploaded file. Please ensure the file is not corrupted.")
                    return None, None, None, None
                    
            except Exception as e:
                st.error(f"Unexpected error processing medical report: {str(e)}")
                return None, None, None, None
    
    return None, None, None, None

def show_processing_results(medical_info: Dict[str, Any], confidence_score: float, health_alerts: List[str]):
    """Display processing results with enhanced UI"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        conditions_count = len(medical_info.get('medical_conditions', []))
        st.metric("Medical Conditions", conditions_count)
        
    with col2:
        lab_count = len(medical_info.get('lab_values', []))
        st.metric("Lab Values", lab_count)
        
    with col3:
        confidence_pct = int(confidence_score * 100)
        st.metric("Processing Confidence", f"{confidence_pct}%")
    
    # Show health alerts if any
    if health_alerts:
        for alert in health_alerts:
            alert_type = alert.get('type', 'info')
            message = alert.get('message', '')
            
            if alert_type == 'critical':
                st.markdown(f'<div class="health-alert">🚨 <strong>Critical:</strong> {message}</div>', unsafe_allow_html=True)
            elif alert_type == 'warning':
                st.markdown(f'<div class="health-warning">⚠️ <strong>Warning:</strong> {message}</div>', unsafe_allow_html=True)
            elif alert_type == 'good':
                st.markdown(f'<div class="health-good">✅ <strong>Good:</strong> {message}</div>', unsafe_allow_html=True)

def generate_health_alerts(medical_info: Dict[str, Any], health_profile: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate health alerts based on extracted medical information"""
    alerts = []
    
    # Check for critical lab values
    for lab in medical_info.get('lab_values', []):
        try:
            if lab['test'] == 'hba1c' and lab.get('numeric_value'):
                value = float(lab['numeric_value'])
                if value >= 9.0:
                    alerts.append({
                        'type': 'critical',
                        'message': f'HbA1c level of {value}% indicates poor diabetes control. Immediate medical attention recommended.'
                    })
                elif value >= 7.0:
                    alerts.append({
                        'type': 'warning', 
                        'message': f'HbA1c level of {value}% suggests diabetes management needs improvement.'
                    })
                    
            elif lab['test'] == 'blood_pressure' and 'systolic' in lab and 'diastolic' in lab:
                systolic = lab['systolic']
                diastolic = lab['diastolic']
                if systolic >= 180 or diastolic >= 120:
                    alerts.append({
                        'type': 'critical',
                        'message': f'Blood pressure {systolic}/{diastolic} indicates hypertensive crisis. Seek immediate medical care.'
                    })
                elif systolic >= 140 or diastolic >= 90:
                    alerts.append({
                        'type': 'warning',
                        'message': f'Blood pressure {systolic}/{diastolic} indicates hypertension requiring management.'
                    })
                    
        except (ValueError, KeyError):
            continue
    
    # Check cardiovascular risk
    cv_risk = health_profile.get('cardiovascular_risk', 'unknown')
    if cv_risk == 'high':
        alerts.append({
            'type': 'warning',
            'message': 'Multiple cardiovascular risk factors detected. Consider comprehensive lifestyle modifications.'
        })
    
    # Check for medication interactions or concerning combinations
    medications = [med.get('medication', '') for med in medical_info.get('medications', [])]
    if len(medications) > 5:
        alerts.append({
            'type': 'warning',
            'message': f'Taking {len(medications)} medications. Review with healthcare provider for potential interactions.'
        })
    
    # Positive alerts for good control
    for lab in medical_info.get('lab_values', []):
        try:
            if lab['test'] == 'hba1c' and lab.get('numeric_value'):
                value = float(lab['numeric_value'])
                if value < 6.5:
                    alerts.append({
                        'type': 'good',
                        'message': f'HbA1c level of {value}% shows good glucose control.'
                    })
        except (ValueError, KeyError):
            continue
    
    return alerts

def calculate_processing_confidence(medical_info: Dict[str, Any]) -> float:
    """Calculate confidence score based on data extraction quality"""
    confidence = 0.0
    factors = 0
    
    # Factor 1: Number of conditions found
    conditions_count = len(medical_info.get('medical_conditions', []))
    if conditions_count > 0:
        confidence += min(conditions_count * 0.15, 0.3)
    factors += 1
    
    # Factor 2: Lab values found
    lab_count = len(medical_info.get('lab_values', []))
    if lab_count > 0:
        confidence += min(lab_count * 0.1, 0.25)
    factors += 1
    
    # Factor 3: Medications found
    med_count = len(medical_info.get('medications', []))
    if med_count > 0:
        confidence += min(med_count * 0.05, 0.15)
    factors += 1
    
    # Factor 4: Context quality
    contexts_with_data = sum(1 for item in medical_info.get('medical_conditions', []) 
                            if isinstance(item, dict) and item.get('context'))
    if contexts_with_data > 0:
        confidence += 0.2
    factors += 1
    
    # Factor 5: Negation detection
    if medical_info.get('negated_conditions'):
        confidence += 0.1
    factors += 1
    
    # Normalize and cap at 1.0
    if factors > 0:
        confidence = min(confidence + 0.3, 1.0)  # Base confidence
    
    return confidence

# ---------------- ENHANCED ML RECOMMENDATIONS ----------------
def generate_recommendations_ml(inputs: dict) -> Dict[str, List[str]]:
    """Enhanced diet recommendations with medical condition integration"""
    medical_conditions = inputs.get('medical_conditions', [])
    allergies = inputs.get('allergies', ['None'])
    health_profile = inputs.get('health_profile', {})
    
    # Base meal pools with nutritional focus
    breakfast_pool = [
        "Steel-cut Oatmeal with Berries and Nuts", "Greek Yogurt Parfait with Seeds", 
        "Vegetable Scrambled Eggs with Spinach", "Avocado Toast on Ezekiel Bread",
        "Protein Smoothie Bowl with Greens", "Quinoa Breakfast Bowl"
    ]
    
    lunch_pool = [
        "Mediterranean Grilled Chicken Salad", "Quinoa Buddha Bowl with Tahini", 
        "Lentil Soup with Vegetables", "Baked Salmon with Sweet Potato",
        "Turkey and Hummus Wrap", "Chickpea Curry with Brown Rice"
    ]
    
    dinner_pool = [
        "Herb-Baked Salmon with Asparagus", "Chicken Stir-fry with Broccoli", 
        "Mediterranean Vegetable Curry", "Grilled Tofu with Quinoa",
        "Turkey Meatballs with Zucchini Noodles", "Bean and Vegetable Stew"
    ]
    
    snack_pool = [
        "Raw Almonds (portion-controlled)", "Apple with Natural Almond Butter", 
        "Cucumber with Hummus", "Greek Yogurt with Cinnamon",
        "Berries with Cottage Cheese", "Herbal Tea with Walnuts"
    ]
    
    # Apply medical condition modifications
    if 'diabetes' in medical_conditions or 'prediabetes' in medical_conditions:
        breakfast_pool = [
            "Steel-cut Oats with Cinnamon (no added sugar)", "Greek Yogurt with Berries (unsweetened)", 
            "Vegetable Omelet with Fiber", "Chia Seed Pudding (sugar-free)",
            "Protein Smoothie with Spinach", "Almond Flour Pancakes (diabetic-friendly)"
        ]
        lunch_pool = [
            "Grilled Chicken Salad with Olive Oil", "Cauliflower Rice Bowl with Protein", 
            "Vegetable Soup with Lean Protein", "Salmon with Non-starchy Vegetables",
            "Turkey Lettuce Wraps", "Zucchini Noodles with Marinara"
        ]
        snack_pool = [
            "Celery with Almond Butter", "Hard-boiled Egg", "Cucumber Slices",
            "Sugar-free Greek Yogurt", "Raw Nuts (controlled portion)", "Herbal Tea"
        ]
    
    if 'high_blood_pressure' in medical_conditions:
        # DASH diet principles
        lunch_pool = [meal + " (low sodium, high potassium)" for meal in lunch_pool]
        dinner_pool = [meal + " (herb-seasoned, no salt)" for meal in dinner_pool]
        snack_pool = [
            "Fresh Fruit", "Unsalted Nuts", "Low-fat Yogurt", 
            "Vegetable Sticks", "Potassium-rich Smoothie"
        ]
    
    if 'high_cholesterol' in medical_conditions:
        # Heart-healthy modifications
        breakfast_pool = [
            "Oatmeal with Walnuts (soluble fiber)", "Plant-based Smoothie with Flax", 
            "Whole Grain Toast with Avocado", "Quinoa Porridge with Berries",
            "Chia Seed Breakfast Bowl", "Green Tea with Almonds"
        ]
        dinner_pool = [
            "Grilled Salmon with Omega-3", "Plant-based Protein Bowl", 
            "Steamed Fish with Vegetables", "Lentil Curry with Turmeric",
            "Quinoa-stuffed Bell Peppers", "Mediterranean Bean Salad"
        ]
    
    if 'weight_management' in medical_conditions or 'obesity' in medical_conditions:
        # Calorie-controlled, high-fiber options
        breakfast_pool = [
            "Vegetable Egg White Scramble", "Green Smoothie with Protein", 
            "High-fiber Oatmeal Bowl", "Greek Yogurt with Chia Seeds",
            "Cottage Cheese with Cucumber", "Herbal Tea with Berries"
        ]
        lunch_pool = [
            "Large Mixed Greens Salad", "Vegetable Soup (broth-based)", 
            "Grilled Chicken Breast with Vegetables", "Cauliflower Rice Stir-fry",
            "Zucchini Noodles with Lean Protein", "Cabbage Rolls (turkey-filled)"
        ]
        dinner_pool = [
            "Steamed Fish with Broccoli", "Turkey Lettuce Cups",
            "Vegetable Stir-fry (minimal oil)", "Grilled Chicken Salad", 
            "Spaghetti Squash with Marinara", "Stuffed Portobello Mushrooms"
        ]
    
    if 'kidney_disease' in medical_conditions:
        # Kidney-friendly modifications
        lunch_pool = [
            "Low-phosphorus Chicken with Rice", "Cucumber Salad with Herbs",
            "Apple and Chicken Salad", "Cauliflower with Lean Protein"
        ]
        snack_pool = [
            "Apple Slices", "White Rice Crackers", "Cucumber Water",
            "Low-phosphorus Fruits", "Herbal Tea (kidney-safe)"
        ]
    
    # Handle allergies
    if 'Gluten' in allergies:
        breakfast_pool = [meal.replace("Toast", "Rice Cakes").replace("Oats", "Quinoa") for meal in breakfast_pool]
        lunch_pool = [meal.replace("Wrap", "Bowl") for meal in lunch_pool]
    
    if 'Lactose' in allergies:
        breakfast_pool = [meal.replace("Greek Yogurt", "Coconut Yogurt").replace("Milk", "Almond Milk") for meal in breakfast_pool]
        snack_pool = [meal.replace("Yogurt", "Coconut Yogurt").replace("Cottage Cheese", "Hummus") for meal in snack_pool]
    
    if 'Eggs' in allergies:
        breakfast_pool = [meal for meal in breakfast_pool if 'egg' not in meal.lower()]
        snack_pool = [meal for meal in snack_pool if 'egg' not in meal.lower()]
    
    return {
        "breakfast": breakfast_pool[:6],
        "lunch": lunch_pool[:6], 
        "dinner": dinner_pool[:6],
        "snack": snack_pool[:6],
        "mid_morning": snack_pool[:6],
        "evening_snack": snack_pool[:6],
        "afternoon": snack_pool[:6],
        "late_snack": snack_pool[:6],
    }

def create_health_timeline_chart(user_id: int):
    """Create interactive health timeline chart"""
    timeline_data = db.get_user_health_timeline(user_id, days_back=180)
    
    if not timeline_data:
        return None
    
    # Group data by metric
    metrics = {}
    for item in timeline_data:
        metric = item['metric_name']
        if metric not in metrics:
            metrics[metric] = {'dates': [], 'values': []}
        metrics[metric]['dates'].append(item['recorded_date'])
        metrics[metric]['values'].append(item['metric_value'])
    
    if not metrics:
        return None
    
    # Create plotly figure
    fig = go.Figure()
    
    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
    
    for i, (metric_name, data) in enumerate(metrics.items()):
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['values'],
            mode='lines+markers',
            name=metric_name.replace('_', ' ').title(),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Health Metrics Timeline",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        height=400,
        showlegend=True
    )
    
    return fig

# ---------------- NEW: MEAL PLAN MANAGEMENT FUNCTIONS ----------------
def generate_multiple_meal_plans(inputs: dict, num_options: int = 5):
    """Generate multiple meal plan options and store them in session state"""
    total_daily_calories = inputs.get('calories_map', {}).get(inputs.get('plan'), 2000)
    meals_per_day = inputs.get('meals', 3)
    diet_type = inputs.get('diet_type', 'Non-Vegetarian')
    allergies_list = inputs.get('allergies', [])
    base_user_query = f"{inputs.get('plan', '')} {' '.join(inputs.get('medical_conditions', []))}"

    # Define meal structures and calorie percentages
    meal_plan_structures = {
        3: [("Breakfast", 0.30), ("Lunch", 0.40), ("Dinner", 0.30)],
        4: [("Breakfast", 0.25), ("Lunch", 0.35), ("Snack", 0.15), ("Dinner", 0.25)],
        5: [("Breakfast", 0.20), ("Morning Snack", 0.10), ("Lunch", 0.30), ("Afternoon Snack", 0.10), ("Dinner", 0.30)],
        6: [("Breakfast", 0.20), ("Morning Snack", 0.10), ("Lunch", 0.25), ("Afternoon Snack", 0.10), ("Dinner", 0.25), ("Evening Snack", 0.10)],
    }

    meal_structure = meal_plan_structures.get(meals_per_day, meal_plan_structures[3])
    dietary_focus = 'high-protein'
    
    # Generate multiple plan options
    plan_options = []
    
    for plan_index in range(num_options):
        # ==================================================================
        # ✨ BUG FIX ✨
        # These MUST be re-initialized to zero *inside* this loop.
        # This is what was causing your 5710 calorie bug.
        # ==================================================================
        plan_recipes = {}
        plan_macros = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
        used_recipe_names_in_this_plan = set()
        
        for meal_name, percentage in meal_structure:
            target_calories_for_meal = total_daily_calories * percentage
            meal_query = f"high protein {meal_name.split()[0].lower()} food {base_user_query}"
            
            # Get 10 options as fallbacks
            recommendation = recipe_recommender.get_recommendations(
                user_query=meal_query,
                target_calories=target_calories_for_meal,
                dietary_focus=dietary_focus,
                diet_type=diet_type,
                num_recommendations=10, 
                allergies=allergies_list
            )
            
            if not recommendation.empty:
                recipe_to_add = None
                
                # 1. Try to get the ideal recipe for this plan_index
                ideal_index = plan_index % num_options
                if ideal_index < len(recommendation):
                    ideal_recipe = recommendation.iloc[ideal_index]
                    if ideal_recipe['Name'] not in used_recipe_names_in_this_plan:
                        recipe_to_add = ideal_recipe
                
                # 2. If ideal was a duplicate, find the next best non-duplicate
                if recipe_to_add is None:
                    for i in range(len(recommendation)):
                        alternate_recipe = recommendation.iloc[i]
                        if alternate_recipe['Name'] not in used_recipe_names_in_this_plan:
                            recipe_to_add = alternate_recipe
                            break 
                
                # 3. If we found a recipe, add it to the plan
                if recipe_to_add is not None:
                    plan_recipes[meal_name] = recipe_to_add
                    used_recipe_names_in_this_plan.add(recipe_to_add['Name'])
                    plan_macros['calories'] += recipe_to_add['Calories']
                    plan_macros['protein'] += recipe_to_add['ProteinContent']
                    plan_macros['carbs'] += recipe_to_add['CarbohydrateContent']
                    plan_macros['fat'] += recipe_to_add['FatContent']
                else:
                    print(f"Warning: Could not find a unique recipe for {meal_name} in plan {plan_index}")

        if plan_recipes:  # Only add if we have recipes
            plan_options.append({
                'recipes': plan_recipes,
                'macros': plan_macros.copy()  # Use .copy() just in case (good practice)
            })
    
    # Store all options in session state
    st.session_state.meal_plan_options = plan_options
    st.session_state.current_plan_index = 0
    
    if plan_options:
        st.session_state.current_meal_plan = plan_options[0]
    
    return plan_options

def get_next_meal_plan():
    """Get the next meal plan option from stored options"""
    if st.session_state.meal_plan_options:
        st.session_state.current_plan_index = (st.session_state.current_plan_index + 1) % len(st.session_state.meal_plan_options)
        st.session_state.current_meal_plan = st.session_state.meal_plan_options[st.session_state.current_plan_index]
        return st.session_state.current_meal_plan
    return None

def create_export_content(inputs: dict, current_user: dict, meal_plan: dict) -> str:
    """Create comprehensive export content for the meal plan"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Get meal plan data
    recipes = meal_plan.get('recipes', {})
    macros = meal_plan.get('macros', {})
    
    export_content = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           PERSONALIZED DIET PLAN                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated on: {current_time}
Generated by: AI-Powered Diet Recommendation System

┌─────────────────────────────────────────────────────────────────────────────┐
│                                USER PROFILE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Personal Information:
• Name: {current_user.get('full_name', current_user.get('username', 'N/A'))}
• Age: {inputs.get('age', 'N/A')} years
• Gender: {inputs.get('gender', 'N/A')}
• Height: {inputs.get('height', 'N/A')} cm
• Current Weight: {inputs.get('weight', 'N/A')} kg
• Target Weight: {inputs.get('target_weight', 'N/A')} kg

Health Metrics:
• BMI: {inputs.get('bmi', 'N/A')} ({inputs.get('bmi_category', 'N/A')})
• BMR: {inputs.get('bmr', 'N/A')} kcal/day
• Activity Level: {inputs.get('activity', 'N/A')}/4
• Goal: {inputs.get('plan', 'N/A')}

Dietary Preferences:
• Diet Type: {inputs.get('diet_type', 'N/A')}
• Allergies/Restrictions: {', '.join(inputs.get('allergies', ['None']))}
• Meals per day: {inputs.get('meals', 'N/A')}
• Meal prep time: {inputs.get('time_available', 'N/A')}
• Timeline preference: {inputs.get('timeline_pref', 'N/A')}

Medical Considerations:
• Health Conditions: {', '.join(inputs.get('medical_conditions', ['None'])) if inputs.get('medical_conditions') else 'None'}

┌─────────────────────────────────────────────────────────────────────────────┐
│                             NUTRITION TARGETS                              │
└─────────────────────────────────────────────────────────────────────────────┘

Daily Calorie Plan:
"""
    
    # Add calorie breakdown
    calories_map = inputs.get('calories_map', {})
    for plan_name, calories in calories_map.items():
        marker = ">>> " if plan_name == inputs.get('plan') else "    "
        export_content += f"{marker}{plan_name}: {calories:.0f} kcal/day\n"
    
    export_content += f"""
Goal Timeline:
• Weight to {'gain' if inputs.get('weight_diff', 0) > 0 else 'lose'}: {abs(inputs.get('weight_diff', 0)):.1f} kg
• Estimated time: {inputs.get('days_to_goal', 'N/A')} days
• Daily calorie adjustment: {inputs.get('daily_calorie_change', 0):+d} kcal

Actual Plan Macronutrients:
• Total Calories: {int(macros.get('calories', 0))} kcal
• Protein: {int(macros.get('protein', 0))}g ({int(macros.get('protein', 0) * 4)} kcal)
• Carbohydrates: {int(macros.get('carbs', 0))}g ({int(macros.get('carbs', 0) * 4)} kcal)
• Fat: {int(macros.get('fat', 0))}g ({int(macros.get('fat', 0) * 9)} kcal)

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DAILY MEAL PLAN                               │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    
    # Add each meal
    for meal_name, recipe in recipes.items():
        export_content += f"""
═══════════════════════════════════════════════════════════════════════════════
{meal_name.upper()}
═══════════════════════════════════════════════════════════════════════════════

Recipe: {recipe['Name']}

Nutrition per serving:
• Calories: {int(recipe['Calories'])} kcal
• Protein: {int(recipe['ProteinContent'])}g
• Carbohydrates: {int(recipe['CarbohydrateContent'])}g  
• Fat: {int(recipe['FatContent'])}g

INGREDIENTS:
"""
        
        # Add ingredients
        try:
            ingredients = safe_list_parse(recipe['RecipeIngredientParts'])
            for i, ingredient in enumerate(ingredients, 1):
                export_content += f"{i:2}. {ingredient}\n"
        except:
            export_content += "    • Ingredients list not available\n"
        
        export_content += "\nINSTRUCTIONS:\n"
        
        # Add instructions
        try:
            instructions = safe_list_parse(recipe['RecipeInstructions'])
            for i, instruction in enumerate(instructions, 1):
                export_content += f"{i:2}. {instruction}\n"
        except:
            export_content += "    • Instructions not available\n"
        
        export_content += "\n"
    
    # Add footer
    export_content += f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                            IMPORTANT DISCLAIMERS                            │
└─────────────────────────────────────────────────────────────────────────────┘

⚠️  MEDICAL DISCLAIMER:
This diet plan is generated based on your inputs and available data. It is NOT 
a substitute for professional medical advice, diagnosis, or treatment. Always 
consult with qualified healthcare professionals before making significant 
dietary changes, especially if you have medical conditions, allergies, or 
special nutritional needs.

📋 USAGE GUIDELINES:
• Monitor your body's response to dietary changes
• Adjust portions based on your individual needs and activity level
• Stay hydrated throughout the day
• Consider consulting with a registered dietitian for personalized guidance
• Stop any meal that causes adverse reactions and seek medical advice

🔄 PLAN FLEXIBILITY:
This plan serves as a starting point. Feel free to:
• Substitute ingredients based on availability and preference
• Adjust cooking methods to suit your skills and equipment
• Modify portion sizes according to your hunger and satiety cues
• Replace meals with similar nutritional profiles if needed

Generated by AI-Powered Diet Recommendation System
For support: Contact your healthcare provider
Plan Version: {st.session_state.current_plan_index + 1} of {len(st.session_state.meal_plan_options)}

═══════════════════════════════════════════════════════════════════════════════
                              END OF MEAL PLAN
═══════════════════════════════════════════════════════════════════════════════
"""
    
    return export_content
# ---------------- MISSING CALLBACKS / NAV HELPERS ----------------
def _sanitize_allergies_on_submit():
    """
    Sanitize allergy selections stored in session_state before the form submits.
    - Deduplicates
    - Ensures 'None' is exclusive (if user picks other allergies, drop 'None')
    - Ensures at least ["None"] exists if nothing selected
    """
    allergies = st.session_state.get("allergies", [])
    # If it's a single string, convert to list
    if isinstance(allergies, str):
        allergies = [allergies]
    # Ensure list
    allergies = list(dict.fromkeys(allergies or []))  # dedupe preserving order
    # If "None" selected with others, remove "None"
    if "None" in allergies and len(allergies) > 1:
        allergies = [a for a in allergies if a != "None"]
    # If empty, set default
    if not allergies:
        allergies = ["None"]
    st.session_state["allergies"] = allergies
    return None


def go_to_recommendations():
    """
    Switch to the recommendations page in session state.
    Caller usually calls st.rerun() or st.rerun() happens after this.
    """
    st.session_state.page = "recommendations"
    return None


def regenerate_meal_plan():
    """
    Regenerate or cycle meal plan options.
    If options already exist, just switch to the next one.
    If no options, generate new options using existing inputs.
    Returns True on success, False otherwise.
    """
    inputs = st.session_state.get("inputs", {})
    if not inputs:
        # No inputs available to generate plans
        st.warning("No inputs found. Please generate recommendations from the form first.")
        return False

    try:
        # If we already have multiple options, go to next
        if st.session_state.get("meal_plan_options"):
            get_next_meal_plan()
            return True

        # Otherwise generate new options and set first plan
        options = generate_multiple_meal_plans(inputs, num_options=5)
        if options:
            st.session_state.current_meal_plan = options[0]
            st.session_state.current_plan_index = 0
            return True
        return False
    except Exception as e:
        st.error(f"Failed to regenerate meal plans: {e}")
        return False


def go_back():
    """
    Return to the form page.
    """
    st.session_state.page = "form"
    return None

def go_to_chatbot():
    """
    Switch to the chatbot page.
    """
    st.session_state.page = "chatbot"
    return None

# ---------------- MAIN APPLICATION ----------------

if not auth_manager.require_authentication():
    if st.session_state.auth_page == "login":
        auth_manager.login_form()
    elif st.session_state.auth_page == "signup":
        auth_manager.signup_form()
else:
    auth_manager.show_user_info_sidebar()
    current_user = auth_manager.get_current_user()

    # ---------------- ENHANCED FORM PAGE ----------------
    if st.session_state.page == "form":
        st.markdown("<h1>Your Health Dashboard & Diet Plan</h1>", unsafe_allow_html=True)
        
        # Enhanced welcome message
        st.markdown(f"""
            <div style="background: linear-gradient(90deg, #667eea, #764ba2);   
                        padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;">
                <h2 style="color: white; margin: 0;">Welcome back, {current_user['full_name'] or current_user['username']}!</h2>
                <p style="color: #e0e7ff; margin: 8px 0 0 0; font-size: 1.1rem;">Get AI-powered diet recommendations based on your health data</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Load existing profile
        existing_profile = db.get_user_profile(current_user['id'])
        

        # Main form with enhanced styling
        with st.container():
            with st.form(key="enhanced_user_form"):
                # Enhanced medical report upload section
                
                
                # Enhanced Personal & Physical Info Section
                st.markdown(
                    """
                    <div class="personal-info-container">
                        <div class="section-title">Personal & Physical Information</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**Basic Info**")
                    age = st.number_input("Age", 10, 100, 
                                        existing_profile['age'] if existing_profile else 25, 
                                        key="age")
                    gender = st.radio("Gender", ("Male", "Female"), 
                                    index=0 if not existing_profile or existing_profile['gender'] == "Male" else 1,
                                    horizontal=True, key="gender")
                    
                with col_b:
                    st.markdown("**Physical Measurements**")
                    height = st.number_input("Height (cm)", 100.0, 230.0, 
                                           float(existing_profile['height']) if existing_profile else 170.0, 
                                           key="height",
                                           format="%.0f",
                                           step=1.0) # <-- Add this line
                    weight = st.number_input("Weight (kg)", 30.0, 200.0, 
                                           float(existing_profile['weight']) if existing_profile else 78.0, 
                                           key="weight",
                                           format="%.1f",
                                           step=0.1) # <-- Add this line
                with col_c:
                    st.markdown("**Lifestyle & Goals**")
                    activity = st.slider("Activity Level", 0, 4, 
                                        existing_profile['activity'] if existing_profile else 1,
                                        format="%d", 
                                        help="0 = Little/no exercise, 1 = Light exercise, 2 = Moderate exercise, 3 = Heavy exercise, 4 = Extra active", 
                                        key="activity")
                    plan_options = ["Maintain weight", "Mild weight loss", "Weight loss", "Extreme weight loss"]
                    plan_index = 0
                    if existing_profile and existing_profile['plan'] in plan_options:
                        plan_index = plan_options.index(existing_profile['plan'])
                    plan = st.selectbox("Weight Goal", plan_options, index=plan_index, key="plan")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Dietary Preferences & Lifestyle Section
                st.markdown(
                    """
                    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 16px; border-radius: 12px; margin: 12px 0;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #f1f5f9; margin-bottom: 16px; text-align: center;">Dietary Preferences & Lifestyle</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                col_d, col_e, col_f = st.columns(3)
                with col_d:
                    st.markdown("**Diet Type & Allergies**")
                    
                    # Add Veg/Non-Veg choice
                    default_diet_type = existing_profile.get('diet_type', 'Non-Vegetarian') if existing_profile else 'Non-Vegetarian'
                    diet_type = st.radio("Diet Type", ("Vegetarian", "Non-Vegetarian"), 
                                        index=0 if default_diet_type == "Vegetarian" else 1,
                                        horizontal=True, key="diet_type")
                    
                    # Get medical conditions from latest report
                    medical_conditions = db.get_latest_medical_conditions(current_user['id'])
                    if medical_conditions:
                        st.markdown("*Medical report conditions:*")
                        for condition in medical_conditions[:3]:  # Show max 3
                            st.markdown(f"• {condition.replace('_', ' ').title()}")
                    
                    default_allergies = existing_profile['allergies'] if existing_profile else ["None"]
                    if not default_allergies:
                        default_allergies = ["None"]
                    allergies = st.multiselect("Allergies / Foods to Avoid", 
                                             ["None", "Gluten", "Lactose", "Peanuts", "Tree Nuts", "Shellfish", "Eggs", "Soy", "Fish"], 
                                             default=default_allergies, 
                                             key="allergies")
                with col_e:
                    st.markdown("**Cooking & Time**")
                    time_options = ["15-30 mins", "30-60 mins", "1-2 hours", "2+ hours"]
                    time_index = 1
                    if existing_profile and existing_profile['time_available'] in time_options:
                        time_index = time_options.index(existing_profile['time_available'])
                    time_available = st.selectbox("Time for meal prep", time_options, index=time_index, key="time_available")
                with col_f:
                    st.markdown("**Preferences**")
                    meals_options = [3, 4, 5, 6]
                    meals_index = 0
                    if existing_profile and existing_profile['meals'] in meals_options:
                        meals_index = meals_options.index(existing_profile['meals'])
                    meals = st.selectbox("Meals per day", meals_options, index=meals_index, key="meals")
                   
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Goals Section
                st.markdown(
                    """
                    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 16px; border-radius: 12px; margin: 12px 0;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #f1f5f9; margin-bottom: 16px; text-align: center;">Goals & Targets</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                col_g, col_h = st.columns(2)
                
                with col_g:
                    st.markdown("**Timeline & Challenges**")
                    target_weight = st.number_input("Target Weight (kg)", 30.0, 200.0, 
                                                   float(existing_profile['target_weight']) if existing_profile else float(weight), 
                                                   key="target_weight")
                with col_h:
                    st.markdown("**Goal time**")
                    timeline_options = ["As fast as possible", "Moderate pace", "Slow and steady"]
                    timeline_index = 0
                    if existing_profile and existing_profile['timeline_pref'] in timeline_options:
                        timeline_index = timeline_options.index(existing_profile['timeline_pref'])
                    timeline_pref = st.selectbox("Preferred Timeline", timeline_options, index=timeline_index, key="timeline_pref")

                st.markdown(
                    """
                    <div class="medical-upload-container">
                        <div class="upload-section-title">
                            Advanced Medical Report Analysis
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("**Upload medical reports, lab results, or health summaries for AI-powered analysis with OCR support for scanned documents.**")
                
                uploaded_file = st.file_uploader(
                    "Choose your medical file",
                    type=['pdf', 'docx', 'txt'],
                    help="Supports PDF (including scanned), DOCX, and TXT files. Our AI can extract conditions, lab values, medications, and generate personalized diet recommendations.",
                    key="enhanced_medical_upload"
                )
                
                # Process medical report with enhanced features
                if uploaded_file is not None:
                    medical_info, summary, diet_conditions, health_profile = process_medical_report(uploaded_file, current_user)
                    
                    if medical_info and summary:
                        st.markdown(
                            f"""
                            <div class="medical-info-display">
                                <h4 style="color: #10b981; margin-bottom: 16px;">Advanced Medical Analysis Complete</h4>
                                <div style="background-color: #0f172a; padding: 16px; border-radius: 10px; margin-top: 12px;">
                                    {summary.replace('📋', '').replace('🏥', '•').replace('🧪', '•').replace('💊', '•').replace('📝', '•')}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        if diet_conditions:
                            st.success(f"Identified {len(diet_conditions)} health condition(s) for personalized nutrition planning")
    
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button(
                    "Generate Recommendations",
                    use_container_width=True,
                )
                
                if submitted:
                    allergies = st.session_state.get("allergies", [])
                    if isinstance(allergies, str):
                        allergies = [allergies]
                    allergies = list(dict.fromkeys(allergies or []))  # Deduplicate
                    
                    # If "None" is selected with other allergies, remove "None"
                    if "None" in allergies and len(allergies) > 1:
                        allergies = [a for a in allergies if a != "None"]
                    
                    # If the list is empty, set it to ["None"]
                    if not allergies:
                        allergies = ["None"]

                    # Include medical conditions in inputs
                    medical_conditions = db.get_latest_medical_conditions(current_user['id'])
                    
                    # Get health profile if available
                    health_profile = {}
                    if st.session_state.extracted_medical_info:
                        health_profile = medical_processor.get_detailed_health_profile(st.session_state.extracted_medical_info)
                    
                    st.session_state.inputs = {
                        "age": age, "gender": gender, "height": height, "weight": weight,
                        "activity": activity, "plan": plan, "allergies": allergies, "meals": meals,
                        "time_available": time_available, "diet_type": diet_type,
                        "timeline_pref": timeline_pref, "target_weight": target_weight,
                        "medical_conditions": medical_conditions,
                        "health_profile": health_profile,
                    }

                    # calculations
                    bmi = calculate_bmi(weight, height)
                    bmi_category = get_bmi_category(bmi)
                    bmr = calculate_bmr(gender, weight, height, age)
                    calories_map = derive_calorie_plan(bmr, activity)

                    # CALCULATE timeline and daily change
                    days_to_goal, daily_calorie_change = calculate_days_to_goal(
                        weight, target_weight, plan, bmr, activity
                    )

                    st.session_state.inputs.update({
                        "bmi": round(bmi, 1),
                        "bmi_category": bmi_category,
                        "bmr": round(bmr, 1),
                        "calories_map": {k: round(v, 0) for k, v in calories_map.items()},
                        "days_to_goal": days_to_goal,
                        "daily_calorie_change": daily_calorie_change,
                        "weight_diff": round(target_weight - weight, 1)
                    })
                    
                    # Save profile to database
                    profile_id = db.save_user_profile(current_user['id'], st.session_state.inputs)
                    st.session_state.inputs['profile_id'] = profile_id
                    
                    go_to_recommendations()
                    st.rerun()

    # ---------------- ENHANCED RECOMMENDATIONS PAGE ----------------
    elif st.session_state.page == "recommendations":
        inputs = st.session_state.inputs
        st.markdown("<h1>🥗 Smart Diet Recommendations</h1>", unsafe_allow_html=True)
        st.write("Your personalized diet plan based on health analysis and nutritional requirements.")

        # Enhanced metric cards with health considerations
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.2, 1.2])
            with c1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">BMI</div>
                        <div class="metric-value">{inputs.get('bmi', '—')}</div>
                        <div class="metric-sub">{inputs.get('bmi_category', '—')}</div>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">BMR</div>
                        <div class="metric-value">{inputs.get('bmr', '—')}</div>
                        <div class="metric-sub">kcal/day</div>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                allergies = inputs.get('allergies', ['None'])
                medical_conditions = inputs.get('medical_conditions', [])
                diet_type = inputs.get('diet_type', 'Non-Vegetarian')
                
                # Create a list of all restrictions
                restrictions = []
                restrictions.append(f"<b>Diet:</b> {diet_type}")
                
                if allergies and allergies != ['None']:
                    # Add each allergy as its own list item
                    for allergy in allergies:
                        restrictions.append(f"<b>Allergy:</b> <span style='color: #f87171;'>{allergy}</span>")
                        
                if medical_conditions:
                    # Add each condition as its own list item
                    for condition in medical_conditions:
                         restrictions.append(f"<b>Medical:</b> <span style='color: #f87171;'>{condition.replace('_', ' ').title()}</span>")

                # If no specific restrictions, add a default message
                if (not allergies or allergies == ['None']) and not medical_conditions:
                     restrictions.append("<i style='color: #94a3b8;'>No specific allergies or conditions.</i>")

                # Build the HTML list
                # This list will give the card the height it needs
                restrictions_html = "<ul style='margin: 8px 0 0 16px; padding: 0; text-align: left; font-size: 0.8rem; line-height: 1.5; min-height: 48px;'>"
                for item in restrictions:
                    restrictions_html += f"<li style='color: #cbd5e1;'>{item}</li>"
                restrictions_html += "</ul>"
                
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Health Considerations</div>
                        {restrictions_html}
                    </div>
                """, unsafe_allow_html=True)
            with c4:
                # Read fresh values from inputs
                days = inputs.get('days_to_goal', 0)
                weight_diff = inputs.get('weight_diff', 0)
                calorie_change = inputs.get('daily_calorie_change', 0)

                # Determine readable timeline text
                if abs(weight_diff) < 1e-6:
                    goal_text = "Already at target!"
                    timeline_text = "No change needed"
                else:
                    if days == 0:
                        goal_text = f"{'Gain' if weight_diff>0 else 'Lose'} {abs(weight_diff)}kg"
                        timeline_text = "Not achievable with selected plan"
                    else:
                        weeks = days // 7
                        remaining_days = days % 7
                        if weeks > 0:
                            timeline_text = f"{weeks}w {remaining_days}d" if remaining_days > 0 else f"{weeks} weeks"
                        else:
                            timeline_text = f"{days} days"
                        goal_text = f"Gain {abs(weight_diff)}kg" if weight_diff > 0 else f"Lose {abs(weight_diff)}kg"

                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Goal Timeline</div>
                        <div class="metric-sub" style="font-weight: bold; color: #10b981;">{goal_text}</div>
                        <div class="metric-sub">Time: {timeline_text}</div>
                        <div class="metric-sub">Daily: {calorie_change:+d} cal</div>
                    </div>
                """, unsafe_allow_html=True)
            with c5:
                calories_map = inputs.get('calories_map', {})
                current_plan = inputs.get('plan', 'Maintain weight')
                plan_calories = calories_map.get(current_plan, 'Unknown')
                
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Daily Nutrition Target</div>
                        <div class="metric-sub">Goal: <b>{current_plan}</b></div>
                        <div class="metric-sub">Calories: <b>{plan_calories:.0f}</b> kcal</div>
                        <div class="metric-sub">Meals: <b>{inputs.get('meals','—')}</b>/day</div>
                    </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Show health alerts if any
        if st.session_state.health_alerts:
            st.markdown("### Health Alerts")
            for alert in st.session_state.health_alerts:
                alert_type = alert.get('type', 'info')
                message = alert.get('message', '')
                
                if alert_type == 'critical':
                    st.markdown(f'<div class="health-alert">🚨 <strong>Critical:</strong> {message}</div>', unsafe_allow_html=True)
                elif alert_type == 'warning':
                    st.markdown(f'<div class="health-warning">⚠️ <strong>Warning:</strong> {message}</div>', unsafe_allow_html=True)
                elif alert_type == 'good':
                    st.markdown(f'<div class="health-good">✅ <strong>Good:</strong> {message}</div>', unsafe_allow_html=True)

        # ------------------- ENHANCED MEAL PLAN GENERATION LOGIC -------------------
        st.markdown("---")
        st.markdown("### 🍳 Your Personalized Daily Meal Plan")
        
        # Generate or get existing meal plans
        if not st.session_state.meal_plan_options:
            with st.spinner("Creating multiple high-protein meal plan options for you..."):
                meal_plan_options = generate_multiple_meal_plans(inputs, num_options=5)
                if meal_plan_options:
                    st.session_state.current_meal_plan = meal_plan_options[0]
        
        # Display current meal plan
        if st.session_state.current_meal_plan:
            current_plan = st.session_state.current_meal_plan
            all_meal_recommendations = current_plan.get('recipes', {})
            total_macros = current_plan.get('macros', {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0})
            
            total_daily_calories = inputs.get('calories_map', {}).get(inputs.get('plan'), 2000)
            
            # UPGRADED DAILY PLAN SUMMARY
            st.markdown(f"#### Daily Plan Summary (Plan {st.session_state.current_plan_index + 1} of {len(st.session_state.meal_plan_options)})")
            
            c1, c2 = st.columns(2)
            c1.metric("Target Daily Calories", f"{int(total_daily_calories)} kcal")
            c2.metric("Plan Total Calories", f"{int(total_macros['calories'])} kcal", 
                      delta=f"{int(total_macros['calories'] - total_daily_calories)} kcal vs Target")

            # Display macronutrients
            st.markdown(f"""
            <div class="metric-card" style="margin-top: 1rem;">
                <div class="metric-title">Total Macronutrients</div>
                <div class="macro-container">
                    <div class="macro-item">
                        <div class="macro-value">{int(total_macros['protein'])}g</div>
                        <div class="macro-label">Protein</div>
                    </div>
                    <div class="macro-item">
                        <div class="macro-value">{int(total_macros['carbs'])}g</div>
                        <div class="macro-label">Carbs</div>
                    </div>
                    <div class="macro-item">
                        <div class="macro-value">{int(total_macros['fat'])}g</div>
                        <div class="macro-label">Fat</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Display each meal's recipe details
            if all_meal_recommendations:
                for meal_name, recipe in all_meal_recommendations.items():
                    st.subheader(f"🍴 {meal_name}")
                    with st.expander(f"**{recipe['Name']}** - ({int(recipe['Calories'])} Calories)"):
                        st.markdown(f"**Protein:** {int(recipe['ProteinContent'])}g | **Carbs:** {int(recipe['CarbohydrateContent'])}g | **Fat:** {int(recipe['FatContent'])}g")
                        
                        st.markdown("**Ingredients:**")
                        ingredients = safe_list_parse(recipe['RecipeIngredientParts'])
                        for item in ingredients:
                            st.write(f"- {item}")
                        
                        st.markdown("**Instructions:**")
                        instructions = safe_list_parse(recipe['RecipeInstructions'])
                        for i, step in enumerate(instructions):
                            st.write(f"{i+1}. {step}")
            else:
                st.warning("Could not generate a meal plan for your specific needs. Please try adjusting your goals.")
        else:
            st.warning("No meal plan available. Please go back and generate recommendations.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="success-box">
            💡 **Disclaimer:** These recommendations are generated based on your inputs and medical information. 
            Always consult with healthcare professionals before making significant dietary changes, especially if you have medical conditions.
            </div><br>
            """,
            unsafe_allow_html=True
        )
        
            
        # ENHANCED ACTION BUTTONS WITH WORKING FUNCTIONALITY
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Regenerate Recommendations", use_container_width=True):
                # Use the smart regeneration function
                if regenerate_meal_plan():
                    # Save new recommendations to database
                    if 'profile_id' in inputs and st.session_state.current_meal_plan:
                        current_plan = st.session_state.current_meal_plan
                        recommendations_data = {
                            'recipes': current_plan.get('recipes', {}),
                            'macros': current_plan.get('macros', {}),
                            'calories_map': inputs.get('calories_map', {}),
                            'goal_timeline': {
                                'days': inputs.get('days_to_goal', 0),
                                'weight_diff': inputs.get('weight_diff', 0),
                                'daily_calorie_change': inputs.get('daily_calorie_change', 0)
                            },
                            'plan_version': st.session_state.current_plan_index + 1
                        }
                        health_considerations = inputs.get('medical_conditions', [])
                        db.save_diet_recommendations(current_user['id'], inputs['profile_id'], recommendations_data, health_considerations)
                    st.rerun()
                else:
                    st.error("Unable to generate new recommendations. Please try going back and creating a new plan.")
        
        with col2:
            if st.button("📄 Export Plan", use_container_width=True) and st.session_state.current_meal_plan:
                # Create comprehensive export content
                export_content = create_export_content(inputs, current_user, st.session_state.current_meal_plan)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"diet_plan_{current_user.get('username', 'user')}_{timestamp}.txt"
                
                st.download_button(
                    label="📥 Download Complete Plan",
                    data=export_content,
                    file_name=filename,
                    mime="text/plain",
                    use_container_width=True
                )
                
                # Show preview of export
                with st.expander("📋 Preview Export Content"):
                    st.text(export_content[:1000] + "..." if len(export_content) > 1000 else export_content)
            if st.button("🤖 Ask AI Nutritionist", on_click=go_to_chatbot, use_container_width=True):
                pass    
        with col3:
            if st.button("⬅️ Back to Form", on_click=go_back, use_container_width=True):
                pass

        
        # Nutritional guidance based on health conditions
        if inputs.get('medical_conditions'):
            st.markdown("### Nutritional Guidelines for Your Health Conditions")
            
            for condition in inputs.get('medical_conditions', []):
                if condition == 'diabetes':
                    st.markdown("""
                    **Diabetes Management:**
                    - Focus on low glycemic index foods
                    - Monitor carbohydrate portions
                    - Include fiber-rich vegetables
                    - Choose lean proteins
                    """)
                elif condition == 'high_blood_pressure':
                    st.markdown("""
                    **Blood Pressure Management:**
                    - Reduce sodium intake (<2300mg/day)
                    - Increase potassium-rich foods
                    - Follow DASH diet principles
                    - Limit processed foods
                    """)
                elif condition == 'high_cholesterol':
                    st.markdown("""
                    **Cholesterol Management:**
                    - Increase soluble fiber intake
                    - Choose heart-healthy fats (omega-3)
                    - Limit saturated and trans fats
                    - Include plant sterols
                    """)
# ==============================================================================
    # NEW: CHATBOT PAGE
    # ==============================================================================
    elif st.session_state.page == "chatbot":
        st.markdown("<h1>🤖 AI Nutrition Assistant</h1>", unsafe_allow_html=True)
        st.markdown("Ask me anything about nutrition, or find recipes tailored to your health profile.")
        
        # Button to go back to the recommendations
        if st.button("⬅️ Back to Recommendations", on_click=go_to_recommendations):
            pass

        st.divider()

        # --- 1. Load Profile ---
        # Get user profile from session state
        if "inputs" not in st.session_state or not st.session_state.inputs:
            st.error("Please fill out the form on the first page to activate the chatbot.")
            st.stop()
        
        user_profile_inputs = st.session_state.inputs
        
        # --- 2. Display Chat History ---
        # We initialized this in Step 3
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # --- 3. Handle New User Input ---
        if prompt := st.chat_input("Ask for advice or a recipe (e.g., 'Is salmon good for me?')"):
            
            # a. Add user message to history and display it
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # b. Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    
                    # Get the current user's ID
                    current_user_id = auth_manager.get_current_user()['id']
                    
                    # Call the updated FastAPI function, sending the user_id
                    response = call_qna_ml_api(
                        query=prompt, 
                        user_id=current_user_id, 
                        history=st.session_state.chat_history 
                    )
                    
                    assistant_message_content = ""
                    
                    # --- c. Process the API response ---
                    if response.get("type") == "advice":
                        assistant_message_content = response.get("data", "I'm not sure how to answer that.")
                        st.markdown(assistant_message_content)

                    elif response.get("type") == "recipes":
                        recipes = response.get("data", [])
                        if not recipes:
                            assistant_message_content = "I couldn't find any recipes that match your request and profile."
                            st.markdown(assistant_message_content)
                        else:
                            # Display the full recipe details in the chat
                            recipe_intro = f"I found {len(recipes)} recipe(s) for you:"
                            st.markdown(recipe_intro)
                            
                            recipe_summary_for_history = [recipe_intro]
                            
                            for i, recipe in enumerate(recipes):
                                title = recipe.get('title', 'N/A')
                                recipe_summary_for_history.append(title)
                                
                                with st.expander(f"**{i+1}. {title}**"):
                                    # Use the safe_list_parse function already in app.py
                                    st.markdown("**Ingredients:**")
                                    try:
                                        ingredients = safe_list_parse(recipe.get('ingredients', 'N/A'))
                                        for item in ingredients: st.write(f"- {item.strip()}")
                                    except Exception:
                                        st.write("- (Could not parse ingredients)")
                                        
                                    st.markdown("**Directions:**")
                                    try:
                                        directions = safe_list_parse(recipe.get('directions', 'N/A'))
                                        for j, step in enumerate(directions):
                                            if step.strip(): st.write(f"{j+1}. {step.strip()}")
                                    except Exception:
                                        st.write("- (Could not parse directions)")
                            
                            # For the chat history, we only store the summary
                            assistant_message_content = "\n".join(recipe_summary_for_history)
                        
                    else:
                        # This handles errors from the call_qna_ml_api function
                        assistant_message_content = response.get("data", "An error occurred.")
                        st.error(assistant_message_content)

            # d. Add AI's (summary) response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_message_content})