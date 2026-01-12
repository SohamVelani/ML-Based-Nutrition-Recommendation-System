import sqlite3
import hashlib
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "diet_app.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # User profiles table (stores diet recommendation inputs)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    age INTEGER,
                    gender VARCHAR(10),
                    height REAL,
                    weight REAL,
                    activity_level INTEGER,
                    weight_goal VARCHAR(50),
                    target_weight REAL,
                    allergies TEXT,
                    meals_per_day INTEGER,
                    time_available VARCHAR(50),
                    timeline_preference VARCHAR(50),
                    bmi REAL,
                    bmr REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Enhanced medical reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS medical_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename VARCHAR(255),
                    file_type VARCHAR(10),
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    extracted_info TEXT,
                    summary_text TEXT,
                    diet_relevant_conditions TEXT,
                    health_profile TEXT,
                    processing_method VARCHAR(50),
                    confidence_score REAL,
                    negated_conditions TEXT,
                    lab_values TEXT,
                    medications TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Medical conditions table (normalized storage)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS medical_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL,
                    condition_name VARCHAR(100),
                    original_text VARCHAR(200),
                    confidence VARCHAR(20),
                    context TEXT,
                    is_negated BOOLEAN DEFAULT FALSE,
                    severity VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (report_id) REFERENCES medical_reports (id)
                )
            ''')
            
            # Lab values table (normalized storage)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lab_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL,
                    test_name VARCHAR(50),
                    value_text VARCHAR(50),
                    numeric_value REAL,
                    unit VARCHAR(20),
                    interpretation VARCHAR(100),
                    is_abnormal BOOLEAN DEFAULT FALSE,
                    reference_range VARCHAR(100),
                    test_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (report_id) REFERENCES medical_reports (id)
                )
            ''')
            
            # Medications table (normalized storage)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS medications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL,
                    medication_name VARCHAR(100),
                    dosage VARCHAR(50),
                    frequency VARCHAR(50),
                    route VARCHAR(50),
                    context TEXT,
                    is_current BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (report_id) REFERENCES medical_reports (id)
                )
            ''')
            
            # Diet recommendations history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diet_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    profile_id INTEGER NOT NULL,
                    recommendations TEXT,
                    health_considerations TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (profile_id) REFERENCES user_profiles (id)
                )
            ''')
            
            # User health timeline (track changes over time)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_timeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    report_id INTEGER,
                    metric_name VARCHAR(50),
                    metric_value REAL,
                    metric_unit VARCHAR(20),
                    recorded_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (report_id) REFERENCES medical_reports (id)
                )
            ''')
            
            # Add indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_medical_reports_user_id ON medical_reports(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_medical_conditions_report_id ON medical_conditions(report_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lab_values_report_id ON lab_values(report_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_medications_report_id ON medications(report_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_timeline_user_id ON health_timeline(user_id)')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, email: str, password: str, full_name: str = "") -> bool:
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, full_name))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError as e:
            logger.warning(f"User creation failed: {e}")
            conn.close()
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating user: {e}")
            conn.close()
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user data if successful"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                SELECT id, username, email, full_name, created_at, is_active
                FROM users 
                WHERE username = ? AND password_hash = ? AND is_active = TRUE
            ''', (username, password_hash))
            
            user = cursor.fetchone()
            
            if user:
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                ''', (user[0],))
                conn.commit()
                
                user_data = {
                    'id': user[0],
                    'username': user[1],
                    'email': user[2],
                    'full_name': user[3],
                    'created_at': user[4],
                    'is_active': user[5]
                }
                return user_data
            
            return None
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
        finally:
            conn.close()
    
    def check_username_exists(self, username: str) -> bool:
        """Check if username already exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            exists = cursor.fetchone() is not None
            return exists
        except Exception as e:
            logger.error(f"Error checking username: {e}")
            return False
        finally:
            conn.close()
    
    def check_email_exists(self, email: str) -> bool:
        """Check if email already exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            exists = cursor.fetchone() is not None
            return exists
        except Exception as e:
            logger.error(f"Error checking email: {e}")
            return False
        finally:
            conn.close()
    
    def save_user_profile(self, user_id: int, profile_data: Dict[str, Any]) -> int:
        """Save user profile data and return profile ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert allergies list to JSON string
            allergies_json = json.dumps(profile_data.get('allergies', []))
            
            cursor.execute('''
                INSERT INTO user_profiles (
                    user_id, age, gender, height, weight, activity_level, weight_goal,
                    target_weight, allergies, meals_per_day, time_available, 
                    timeline_preference, bmi, bmr
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                profile_data.get('age'),
                profile_data.get('gender'),
                profile_data.get('height'),
                profile_data.get('weight'),
                profile_data.get('activity'),
                profile_data.get('plan'),
                profile_data.get('target_weight'),
                allergies_json,
                profile_data.get('meals'),
                profile_data.get('time_available'),
                profile_data.get('timeline_pref'),
                profile_data.get('bmi'),
                profile_data.get('bmr')
            ))
            
            profile_id = cursor.lastrowid
            conn.commit()
            return profile_id
            
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get the latest user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM user_profiles 
                WHERE user_id = ? 
                ORDER BY updated_at DESC 
                LIMIT 1
            ''', (user_id,))
            
            profile = cursor.fetchone()
            
            if profile:
                return {
                    'id': profile[0],
                    'user_id': profile[1],
                    'age': profile[2],
                    'gender': profile[3],
                    'height': profile[4],
                    'weight': profile[5],
                    'activity': profile[6],
                    'plan': profile[7],
                    'target_weight': profile[8],
                    'allergies': json.loads(profile[9]) if profile[9] else [],
                    'meals': profile[10],
                    'time_available': profile[11],
                    'timeline_pref': profile[12],
                    'bmi': profile[13],
                    'bmr': profile[14],
                    'created_at': profile[15],
                    'updated_at': profile[16]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
        finally:
            conn.close()
    
    def save_diet_recommendations(self, user_id: int, profile_id: int, 
                                recommendations: Dict[str, Any], 
                                health_considerations: List[str] = None) -> bool:
        """Save diet recommendations to history with health considerations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            recommendations_json = json.dumps(recommendations)
            health_considerations_json = json.dumps(health_considerations or [])
            
            cursor.execute('''
                INSERT INTO diet_history (user_id, profile_id, recommendations, health_considerations)
                VALUES (?, ?, ?, ?)
            ''', (user_id, profile_id, recommendations_json, health_considerations_json))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")
            return False
        finally:
            conn.close()
    
    def save_medical_report(self, user_id: int, filename: str, file_type: str, 
                           extracted_info: Dict[str, Any], summary_text: str, 
                           diet_conditions: List[str], health_profile: Dict[str, Any] = None,
                           processing_method: str = "nlp", confidence_score: float = 0.0) -> int:
        """Enhanced medical report saving with normalized data storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save main report
            extracted_info_json = json.dumps(extracted_info)
            diet_conditions_json = json.dumps(diet_conditions)
            health_profile_json = json.dumps(health_profile or {})
            negated_conditions_json = json.dumps(extracted_info.get('negated_conditions', []))
            
            cursor.execute('''
                INSERT INTO medical_reports (
                    user_id, filename, file_type, extracted_info, summary_text, 
                    diet_relevant_conditions, health_profile, processing_method,
                    confidence_score, negated_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, filename, file_type, extracted_info_json, 
                  summary_text, diet_conditions_json, health_profile_json,
                  processing_method, confidence_score, negated_conditions_json))
            
            report_id = cursor.lastrowid
            
            # Save normalized medical conditions
            for condition in extracted_info.get('medical_conditions', []):
                if isinstance(condition, dict):
                    cursor.execute('''
                        INSERT INTO medical_conditions 
                        (report_id, condition_name, original_text, confidence, context, is_negated)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        report_id,
                        condition.get('condition', ''),
                        condition.get('original_text', ''),
                        condition.get('confidence', 'medium'),
                        condition.get('context', ''),
                        False
                    ))
                else:
                    cursor.execute('''
                        INSERT INTO medical_conditions 
                        (report_id, condition_name, confidence)
                        VALUES (?, ?, ?)
                    ''', (report_id, condition, 'medium'))
            
            # Save negated conditions
            for neg_condition in extracted_info.get('negated_conditions', []):
                cursor.execute('''
                    INSERT INTO medical_conditions 
                    (report_id, condition_name, original_text, context, is_negated)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    report_id,
                    neg_condition.get('condition', ''),
                    neg_condition.get('original_text', ''),
                    neg_condition.get('context', ''),
                    True
                ))
            
            # Save normalized lab values
            for lab in extracted_info.get('lab_values', []):
                cursor.execute('''
                    INSERT INTO lab_values 
                    (report_id, test_name, value_text, numeric_value, interpretation, is_abnormal)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    report_id,
                    lab.get('test', ''),
                    lab.get('value', ''),
                    lab.get('numeric_value'),
                    lab.get('interpretation', ''),
                    'high' in lab.get('interpretation', '').lower() or 'elevated' in lab.get('interpretation', '').lower()
                ))
                
                # Save to health timeline for tracking
                if lab.get('numeric_value') is not None:
                    cursor.execute('''
                        INSERT INTO health_timeline 
                        (user_id, report_id, metric_name, metric_value, recorded_date)
                        VALUES (?, ?, ?, ?, DATE('now'))
                    ''', (
                        user_id,
                        report_id,
                        lab.get('test', ''),
                        lab.get('numeric_value')
                    ))
            
            # Save medications
            for med in extracted_info.get('medications', []):
                cursor.execute('''
                    INSERT INTO medications 
                    (report_id, medication_name, dosage, context)
                    VALUES (?, ?, ?, ?)
                ''', (
                    report_id,
                    med.get('medication', ''),
                    f"{med.get('dosage', '')} {med.get('unit', '')}".strip(),
                    med.get('context', '')
                ))
            
            conn.commit()
            logger.info(f"Medical report saved successfully with ID: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error saving medical report: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def get_user_medical_reports(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's medical reports history with enhanced data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, filename, file_type, upload_date, summary_text, 
                       diet_relevant_conditions, confidence_score, processing_method
                FROM medical_reports
                WHERE user_id = ?
                ORDER BY upload_date DESC
                LIMIT ?
            ''', (user_id, limit))
            
            reports = cursor.fetchall()
            
            return [
                {
                    'id': r[0],
                    'filename': r[1],
                    'file_type': r[2],
                    'upload_date': r[3],
                    'summary_text': r[4],
                    'diet_conditions': json.loads(r[5]) if r[5] else [],
                    'confidence_score': r[6],
                    'processing_method': r[7]
                }
                for r in reports
            ]
            
        except Exception as e:
            logger.error(f"Error getting medical reports: {e}")
            return []
        finally:
            conn.close()
    
    def get_latest_medical_conditions(self, user_id: int) -> List[str]:
        """Get diet-relevant conditions from latest medical report with confidence filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT diet_relevant_conditions
                FROM medical_reports
                WHERE user_id = ?
                ORDER BY upload_date DESC
                LIMIT 1
            ''', (user_id,))
            
            result = cursor.fetchone()
            
            if result and result[0]:
                return json.loads(result[0])
            return []
            
        except Exception as e:
            logger.error(f"Error getting latest medical conditions: {e}")
            return []
        finally:
            conn.close()
    
    def get_user_diet_history(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's diet recommendation history with health considerations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT dh.id, dh.recommendations, dh.health_considerations, 
                       dh.created_at, up.weight, up.target_weight
                FROM diet_history dh
                JOIN user_profiles up ON dh.profile_id = up.id
                WHERE dh.user_id = ?
                ORDER BY dh.created_at DESC
                LIMIT ?
            ''', (user_id, limit))
            
            history = cursor.fetchall()
            
            return [
                {
                    'id': h[0],
                    'recommendations': json.loads(h[1]) if h[1] else {},
                    'health_considerations': json.loads(h[2]) if h[2] else [],
                    'created_at': h[3],
                    'weight': h[4],
                    'target_weight': h[5]
                }
                for h in history
            ]
            
        except Exception as e:
            logger.error(f"Error getting diet history: {e}")
            return []
        finally:
            conn.close()
    
    def get_user_health_timeline(self, user_id: int, metric_name: str = None, 
                               days_back: int = 90) -> List[Dict[str, Any]]:
        """Get user's health metrics over time for trend analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if metric_name:
                cursor.execute('''
                    SELECT ht.metric_name, ht.metric_value, ht.metric_unit, 
                           ht.recorded_date, mr.filename
                    FROM health_timeline ht
                    LEFT JOIN medical_reports mr ON ht.report_id = mr.id
                    WHERE ht.user_id = ? AND ht.metric_name = ? 
                          AND ht.recorded_date >= DATE('now', '-{} days')
                    ORDER BY ht.recorded_date DESC
                '''.format(days_back), (user_id, metric_name))
            else:
                cursor.execute('''
                    SELECT ht.metric_name, ht.metric_value, ht.metric_unit, 
                           ht.recorded_date, mr.filename
                    FROM health_timeline ht
                    LEFT JOIN medical_reports mr ON ht.report_id = mr.id
                    WHERE ht.user_id = ? 
                          AND ht.recorded_date >= DATE('now', '-{} days')
                    ORDER BY ht.recorded_date DESC, ht.metric_name
                '''.format(days_back), (user_id,))
            
            timeline = cursor.fetchall()
            
            return [
                {
                    'metric_name': t[0],
                    'metric_value': t[1],
                    'metric_unit': t[2],
                    'recorded_date': t[3],
                    'source_file': t[4]
                }
                for t in timeline
            ]
            
        except Exception as e:
            logger.error(f"Error getting health timeline: {e}")
            return []
        finally:
            conn.close()
    
    def get_detailed_medical_conditions(self, user_id: int, include_negated: bool = False) -> List[Dict[str, Any]]:
        """Get detailed medical conditions from all reports"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            where_clause = "WHERE mr.user_id = ?"
            params = [user_id]
            
            if not include_negated:
                where_clause += " AND mc.is_negated = FALSE"
            
            cursor.execute(f'''
                SELECT mc.condition_name, mc.original_text, mc.confidence, 
                       mc.context, mc.is_negated, mc.created_at, mr.filename
                FROM medical_conditions mc
                JOIN medical_reports mr ON mc.report_id = mr.id
                {where_clause}
                ORDER BY mc.created_at DESC
            ''', params)
            
            conditions = cursor.fetchall()
            
            return [
                {
                    'condition_name': c[0],
                    'original_text': c[1],
                    'confidence': c[2],
                    'context': c[3],
                    'is_negated': bool(c[4]),
                    'created_at': c[5],
                    'source_file': c[6]
                }
                for c in conditions
            ]
            
        except Exception as e:
            logger.error(f"Error getting detailed medical conditions: {e}")
            return []
        finally:
            conn.close()
    
    def get_latest_lab_values(self, user_id: int) -> List[Dict[str, Any]]:
        """Get latest lab values for each test type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT lv.test_name, lv.value_text, lv.numeric_value, 
                       lv.interpretation, lv.is_abnormal, lv.created_at, mr.filename
                FROM lab_values lv
                JOIN medical_reports mr ON lv.report_id = mr.id
                WHERE mr.user_id = ?
                  AND lv.created_at = (
                      SELECT MAX(lv2.created_at) 
                      FROM lab_values lv2 
                      JOIN medical_reports mr2 ON lv2.report_id = mr2.id
                      WHERE mr2.user_id = ? AND lv2.test_name = lv.test_name
                  )
                ORDER BY lv.test_name
            ''', (user_id, user_id))
            
            lab_values = cursor.fetchall()
            
            return [
                {
                    'test_name': lv[0],
                    'value_text': lv[1],
                    'numeric_value': lv[2],
                    'interpretation': lv[3],
                    'is_abnormal': bool(lv[4]),
                    'created_at': lv[5],
                    'source_file': lv[6]
                }
                for lv in lab_values
            ]
            
        except Exception as e:
            logger.error(f"Error getting latest lab values: {e}")
            return []
        finally:
            conn.close()
    
    def get_current_medications(self, user_id: int) -> List[Dict[str, Any]]:
        """Get current medications from latest report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT m.medication_name, m.dosage, m.frequency, 
                       m.context, m.created_at, mr.filename
                FROM medications m
                JOIN medical_reports mr ON m.report_id = mr.id
                WHERE mr.user_id = ? AND m.is_current = TRUE
                ORDER BY m.created_at DESC
            ''', (user_id,))
            
            medications = cursor.fetchall()
            
            return [
                {
                    'medication_name': m[0],
                    'dosage': m[1],
                    'frequency': m[2],
                    'context': m[3],
                    'created_at': m[4],
                    'source_file': m[5]
                }
                for m in medications
            ]
            
        except Exception as e:
            logger.error(f"Error getting current medications: {e}")
            return []
        finally:
            conn.close()
    
    def analyze_health_trends(self, user_id: int) -> Dict[str, Any]:
        """Analyze health trends over time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get trending metrics
            cursor.execute('''
                SELECT metric_name, metric_value, recorded_date
                FROM health_timeline
                WHERE user_id = ? AND recorded_date >= DATE('now', '-180 days')
                ORDER BY metric_name, recorded_date
            ''', (user_id,))
            
            timeline_data = cursor.fetchall()
            
            # Group by metric and calculate trends
            trends = {}
            current_metric = None
            metric_values = []
            
            for row in timeline_data:
                metric_name, value, date = row
                
                if metric_name != current_metric:
                    if current_metric and len(metric_values) > 1:
                        # Calculate trend
                        first_value = metric_values[0][0]
                        last_value = metric_values[-1][0]
                        trend_direction = 'stable'
                        
                        if last_value > first_value * 1.1:
                            trend_direction = 'increasing'
                        elif last_value < first_value * 0.9:
                            trend_direction = 'decreasing'
                        
                        trends[current_metric] = {
                            'trend_direction': trend_direction,
                            'first_value': first_value,
                            'last_value': last_value,
                            'data_points': len(metric_values),
                            'date_range': f"{metric_values[0][1]} to {metric_values[-1][1]}"
                        }
                    
                    current_metric = metric_name
                    metric_values = []
                
                metric_values.append((value, date))
            
            # Handle last metric
            if current_metric and len(metric_values) > 1:
                first_value = metric_values[0][0]
                last_value = metric_values[-1][0]
                trend_direction = 'stable'
                
                if last_value > first_value * 1.1:
                    trend_direction = 'increasing'
                elif last_value < first_value * 0.9:
                    trend_direction = 'decreasing'
                
                trends[current_metric] = {
                    'trend_direction': trend_direction,
                    'first_value': first_value,
                    'last_value': last_value,
                    'data_points': len(metric_values),
                    'date_range': f"{metric_values[0][1]} to {metric_values[-1][1]}"
                }
            
            return {
                'trends': trends,
                'analysis_date': datetime.now().isoformat(),
                'total_metrics_tracked': len(trends)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing health trends: {e}")
            return {'trends': {}, 'error': str(e)}
        finally:
            conn.close()
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> Dict[str, int]:
        """Cleanup old data beyond retention period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cleanup_stats = {}
            
            # Clean up old medical reports (keep latest 5 per user regardless of date)
            cursor.execute('''
                DELETE FROM medical_reports 
                WHERE upload_date < DATE('now', '-{} days')
                  AND id NOT IN (
                      SELECT id FROM medical_reports mr
                      WHERE mr.user_id = medical_reports.user_id
                      ORDER BY upload_date DESC
                      LIMIT 5
                  )
            '''.format(days_to_keep))
            cleanup_stats['medical_reports_deleted'] = cursor.rowcount
            
            # Clean up orphaned records
            cursor.execute('DELETE FROM medical_conditions WHERE report_id NOT IN (SELECT id FROM medical_reports)')
            cleanup_stats['orphaned_conditions_deleted'] = cursor.rowcount
            
            cursor.execute('DELETE FROM lab_values WHERE report_id NOT IN (SELECT id FROM medical_reports)')
            cleanup_stats['orphaned_lab_values_deleted'] = cursor.rowcount
            
            cursor.execute('DELETE FROM medications WHERE report_id NOT IN (SELECT id FROM medical_reports)')
            cleanup_stats['orphaned_medications_deleted'] = cursor.rowcount
            
            cursor.execute('DELETE FROM health_timeline WHERE report_id NOT IN (SELECT id FROM medical_reports)')
            cleanup_stats['orphaned_timeline_deleted'] = cursor.rowcount
            
            # Clean up old diet history (keep latest 20 per user)
            cursor.execute('''
                DELETE FROM diet_history 
                WHERE created_at < DATE('now', '-{} days')
                  AND id NOT IN (
                      SELECT id FROM diet_history dh
                      WHERE dh.user_id = diet_history.user_id
                      ORDER BY created_at DESC
                      LIMIT 20
                  )
            '''.format(days_to_keep))
            cleanup_stats['diet_history_deleted'] = cursor.rowcount
            
            conn.commit()
            cleanup_stats['cleanup_date'] = datetime.now().isoformat()
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            conn.rollback()
            return {'error': str(e)}
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics for monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Count records in each table
            tables = ['users', 'user_profiles', 'medical_reports', 'medical_conditions', 
                     'lab_values', 'medications', 'diet_history', 'health_timeline']
            
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Get file size
            stats['database_size_bytes'] = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
        finally:
            conn.close()