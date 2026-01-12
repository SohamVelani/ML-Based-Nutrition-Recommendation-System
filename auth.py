import streamlit as st
import re
from database import DatabaseManager

class AuthManager:
    def __init__(self):
        self.db = DatabaseManager()
    
    @staticmethod
    def init_session_state():
        """Initialize session state variables for authentication"""
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "user_data" not in st.session_state:
            st.session_state.user_data = None
        if "auth_page" not in st.session_state:
            st.session_state.auth_page = "login"
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_password(password: str) -> tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        if not re.search(r'[A-Za-z]', password):
            return False, "Password must contain at least one letter"
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"
        return True, "Password is strong"
    
    @staticmethod
    def validate_username(username: str) -> tuple[bool, str]:
        """Validate username format"""
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"
        return True, "Username is valid"
    
    def login_form(self):
        """Display login form"""
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
                <h2 style="text-align: center; color: white; margin-bottom: 1rem;">
                    🍽️ Diet Recommendation System
                </h2>
                <p style="text-align: center; color: #e0e7ff; font-size: 1.1rem;">
                    Welcome back! Please sign in to continue
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Login Form
        with st.form("login_form", clear_on_submit=False):
            st.markdown("### 🔐 Sign In")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    login_submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                with col_b:
                    signup_button = st.form_submit_button("New User? Sign Up", use_container_width=True)
        
        # Handle signup button
        if signup_button:
            st.session_state.auth_page = "signup"
            st.rerun()
        
        # Handle login submission
        if login_submitted:
            if not username or not password:
                st.error("Please fill in all fields")
                return
            
            user_data = self.db.authenticate_user(username, password)
            if user_data:
                st.session_state.authenticated = True
                st.session_state.user_data = user_data
                st.session_state.page = "form"  # Go to main form
                st.success(f"Welcome back, {user_data['full_name'] or user_data['username']}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    def signup_form(self):
        """Display signup form"""
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
                <h2 style="text-align: center; color: white; margin-bottom: 1rem;">
                    🍽️ Join Diet Recommendation System
                </h2>
                <p style="text-align: center; color: #e0e7ff; font-size: 1.1rem;">
                    Create your account to get personalized diet recommendations
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Signup Form
        with st.form("signup_form", clear_on_submit=False):
            st.markdown("### 📝 Create Account")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                full_name = st.text_input("Full Name", placeholder="Enter your full name")
                username = st.text_input("Username", placeholder="Choose a username")
                email = st.text_input("Email", placeholder="Enter your email address")
                password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    signup_submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                with col_b:
                    login_button = st.form_submit_button("Back to Sign In", use_container_width=True)
        
        # Handle login button
        if login_button:
            st.session_state.auth_page = "login"
            st.rerun()
        
        # Handle signup submission
        if signup_submitted:
            # Validation
            if not all([full_name, username, email, password, confirm_password]):
                st.error("Please fill in all fields")
                return
            
            if password != confirm_password:
                st.error("Passwords do not match")
                return
            
            # Validate email
            if not self.validate_email(email):
                st.error("Please enter a valid email address")
                return
            
            # Validate username
            username_valid, username_msg = self.validate_username(username)
            if not username_valid:
                st.error(username_msg)
                return
            
            # Validate password
            password_valid, password_msg = self.validate_password(password)
            if not password_valid:
                st.error(password_msg)
                return
            
            # Check if username or email already exists
            if self.db.check_username_exists(username):
                st.error("Username already exists. Please choose a different one.")
                return
            
            if self.db.check_email_exists(email):
                st.error("Email already registered. Please use a different email or sign in.")
                return
            
            # Create user
            if self.db.create_user(username, email, password, full_name):
                st.success("Account created successfully! Please sign in.")
                st.session_state.auth_page = "login"
                st.rerun()
            else:
                st.error("Failed to create account. Please try again.")
    
    @staticmethod
    def logout():
        """Logout user"""
        st.session_state.authenticated = False
        st.session_state.user_data = None
        st.session_state.page = "form"
        st.session_state.auth_page = "login"
        st.rerun()
    
    @staticmethod
    def require_authentication():
        """Decorator-like function to require authentication"""
        if not st.session_state.get("authenticated", False):
            return False
        return True
    
    @staticmethod
    def get_current_user():
        """Get current authenticated user data"""
        return st.session_state.get("user_data", None)
    
    def show_user_info_sidebar(self):
        """Show user info in sidebar"""
        if st.session_state.get("authenticated", False):
            user_data = st.session_state.user_data
            
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 👤 User Info")
                # Check if this is the first login by looking at last_login
                if user_data.get('last_login') is None:
                    st.write(f"**Welcome, {user_data['full_name'] or user_data['username']}!**")
                else:
                    st.write(f"**Welcome back, {user_data['full_name'] or user_data['username']}!**")
                st.write(f"**Username:** {user_data['username']}")
                st.write(f"**Email:** {user_data['email']}")
                
                if st.button("Logout", use_container_width=True):
                    self.logout()
                
                st.markdown("---")
                
                # Show user profile summary if exists
                profile = self.db.get_user_profile(user_data['id'])
                if profile:
                    st.markdown("### 📊 Latest Profile")
                    if profile.get('bmi'):
                        st.write(f"**BMI:** {profile['bmi']}")
                    if profile.get('weight') and profile.get('target_weight'):
                        st.write(f"**Current:** {profile['weight']}kg")
                        st.write(f"**Target:** {profile['target_weight']}kg")
                    st.write(f"**Goal:** {profile.get('plan', 'Not set')}")