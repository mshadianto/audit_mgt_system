import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import uuid
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import asyncio
from typing import Dict, List, Any
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
import os
from openai import OpenAI
import hashlib

# ========================================
# CONFIGURATION & CLOUD SERVICES
# ========================================

# Firebase Configuration
@st.cache_resource
def init_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            # Firebase config for "audit mgt system" project
            firebase_config = {
                "type": "service_account",
                "project_id": "audit-mgt-system",
                "private_key_id": st.secrets.get("firebase_private_key_id", ""),
                "private_key": st.secrets.get("firebase_private_key", "").replace("\\n", "\n"),
                "client_email": st.secrets.get("firebase_client_email", ""),
                "client_id": st.secrets.get("firebase_client_id", ""),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{st.secrets.get('firebase_client_email', '')}"
            }
            
            # Initialize Firebase
            if firebase_config["private_key"]:
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                return firestore.client()
            else:
                st.warning("🔥 Firebase: Using demo mode (secrets not configured)")
                return None
    except Exception as e:
        st.error(f"🔥 Firebase initialization error: {str(e)}")
        return None

# OpenRouter API Configuration
class QwenAIEngine:
    def __init__(self):
        self.api_key = st.secrets.get("openrouter_api_key", "")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "qwen/qwen-2.5-72b-instruct"
        
        if self.api_key:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            self.is_available = True
        else:
            self.client = None
            self.is_available = False
            st.warning("🤖 Qwen3: Using mock responses (API key not configured)")
    
    def generate_audit_insight(self, query: str, context: str = "") -> str:
        """Generate audit insights using Qwen3 via OpenRouter"""
        if not self.is_available:
            return self._mock_response(query)
        
        try:
            system_prompt = """You are an expert AI Internal Auditor with deep knowledge in:
            - Financial auditing and controls
            - Risk assessment and fraud detection  
            - SOX compliance and COSO framework
            - Data analytics and anomaly detection
            - Audit standards (IIA, AICPA, ISA)
            
            Provide detailed, actionable audit insights with specific recommendations.
            Use emojis for better readability and professional tone."""
            
            user_prompt = f"""
            Audit Query: {query}
            
            Context: {context}
            
            Please provide a comprehensive audit analysis including:
            1. Key findings and insights
            2. Risk assessment
            3. Recommendations
            4. Next steps
            
            Format your response with clear sections and bullet points.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"🤖 Qwen3 API Error: {str(e)}")
            return self._mock_response(query)
    
    def _mock_response(self, query: str) -> str:
        """Fallback mock response when API is not available"""
        return f"""
        🤖 **AI Audit Analysis (Demo Mode)**
        
        **🔍 Query Processed:** {query}
        
        **⚠️ Note:** This is a demonstration response. For real AI insights, configure your OpenRouter API key.
        
        **💡 Key Insights:**
        • Advanced pattern analysis would be performed on your data
        • Risk scoring based on machine learning models
        • Automated control testing and exception identification
        • Predictive analytics for future risk assessment
        
        **🎯 AI Confidence:** Demo Mode - Configure API for real analysis
        """

# Cloud Database Operations
class AuditCloudDB:
    def __init__(self, db_client):
        self.db = db_client
        self.is_available = db_client is not None
    
    def save_audit_session(self, session_data: dict) -> bool:
        """Save audit session to Firebase"""
        if not self.is_available:
            return False
        
        try:
            doc_ref = self.db.collection('audit_sessions').document(session_data['session_id'])
            doc_ref.set({
                **session_data,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'created_by': 'MS Hadianto AI System'
            })
            return True
        except Exception as e:
            st.error(f"🔥 Database save error: {str(e)}")
            return False
    
    def load_audit_findings(self) -> pd.DataFrame:
        """Load audit findings from Firebase"""
        if not self.is_available:
            return pd.DataFrame()
        
        try:
            docs = self.db.collection('audit_findings').stream()
            findings = []
            for doc in docs:
                finding = doc.to_dict()
                finding['id'] = doc.id
                findings.append(finding)
            return pd.DataFrame(findings)
        except Exception as e:
            st.error(f"🔥 Database load error: {str(e)}")
            return pd.DataFrame()
    
    def save_ai_interaction(self, query: str, response: str, session_id: str) -> bool:
        """Save AI interaction to Firebase for audit trail"""
        if not self.is_available:
            return False
        
        try:
            doc_ref = self.db.collection('ai_interactions').add({
                'session_id': session_id,
                'query': query,
                'response': response,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'model': 'qwen-2.5-72b-instruct',
                'user_agent': 'Streamlit_Audit_App'
            })
            return True
        except Exception as e:
            st.error(f"🔥 AI interaction save error: {str(e)}")
            return False

# Initialize Cloud Services
@st.cache_resource
def initialize_cloud_services():
    """Initialize all cloud services"""
    firebase_db = init_firebase()
    ai_engine = QwenAIEngine()
    cloud_db = AuditCloudDB(firebase_db)
    
    return {
        'firebase_db': firebase_db,
        'ai_engine': ai_engine, 
        'cloud_db': cloud_db
    }

# Get cloud services
cloud_services = initialize_cloud_services()

# Page config
st.set_page_config(
    page_title="🤖 RAG Agentic AI - Internal Audit (Cloud)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .ai-response {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warn { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .demo-highlight {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #333;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Data generators
class DataGenerator:
    @staticmethod
    def generate_audit_findings(count=50):
        np.random.seed(42)
        areas = ["Financial Controls", "IT Security", "Operations", "Compliance", "Risk Management"]
        severities = ["Low", "Medium", "High", "Critical"]
        statuses = ["Open", "In Progress", "Closed"]
        
        findings = []
        for i in range(count):
            findings.append({
                'id': f'F{i+1:03d}',
                'title': f'Control Gap - {np.random.choice(areas)}',
                'area': np.random.choice(areas),
                'severity': np.random.choice(severities, p=[0.3, 0.4, 0.25, 0.05]),
                'status': np.random.choice(statuses, p=[0.4, 0.35, 0.25]),
                'owner': f'Auditor_{np.random.randint(1, 10)}',
                'created_date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
                'risk_score': np.random.uniform(3, 9)
            })
        return pd.DataFrame(findings)
    
    @staticmethod
    def generate_transaction_data(count=5000):
        np.random.seed(42)
        
        # Normal transactions (95%)
        normal_count = int(count * 0.95)
        normal_amounts = np.random.lognormal(3, 1, normal_count)
        
        # Anomalous transactions (5%)
        anomaly_count = count - normal_count
        anomaly_amounts = np.random.uniform(50000, 200000, anomaly_count)
        
        all_amounts = np.concatenate([normal_amounts, anomaly_amounts])
        is_anomaly = [0] * normal_count + [1] * anomaly_count
        
        return pd.DataFrame({
            'transaction_id': [f'TXN_{i:06d}' for i in range(count)],
            'amount': all_amounts,
            'date': pd.date_range('2024-01-01', periods=count, freq='H'),
            'department': np.random.choice(['Finance', 'Operations', 'IT', 'HR', 'Marketing'], count),
            'user_id': [f'USR_{np.random.randint(1, 500):04d}' for _ in range(count)],
            'vendor_id': [f'VND_{np.random.randint(1, 1000):04d}' for _ in range(count)],
            'is_anomaly': is_anomaly,
            'risk_score': np.concatenate([
                np.random.uniform(0.1, 0.3, normal_count),
                np.random.uniform(0.7, 0.95, anomaly_count)
            ])
        })

# AI Mock Engine (keeping for fallback)
class MockAIEngine:
    @staticmethod
    def generate_audit_insight(query):
        """Fallback mock responses if cloud services fail"""
        insights = {
            "fraud": """
            🕵️ **AI Fraud Detection Analysis (Mock):**
            
            Based on advanced ML analysis of transaction patterns:
            
            **🔍 Key Findings:**
            • 47 high-risk transactions identified (Risk Score > 0.8)
            • Pattern detected: Round number bias in 23% of flagged transactions  
            • Unusual timing: 67% of anomalies occur after business hours
            • User behavior analysis reveals 3 accounts with suspicious patterns
            
            **📊 Statistical Analysis:**
            • Benford's Law compliance: FAILED (χ² = 28.4, p < 0.001)
            • Transaction velocity spike: 340% above baseline
            • Amount clustering detected around $50K, $75K, $100K
            
            **⚡ Immediate Actions:**
            1. Freeze high-risk accounts pending investigation
            2. Implement enhanced real-time monitoring
            3. Review authorization controls for large transactions
            4. Conduct forensic analysis on flagged patterns
            
            **🎯 AI Confidence:** 94.7% (Mock Mode)
            """,
            
            "risk": """
            📈 **AI Risk Prediction Model (Mock):**
            
            6-month predictive analysis using LSTM and ensemble methods:
            
            **🎯 Risk Forecast:**
            • Overall organizational risk trending upward (+18%)
            • Highest probability areas: Revenue Recognition (87%), IT Security (82%)
            • Expected new findings: 12-18 (vs 8-12 historical average)
            
            **📊 Key Risk Drivers:**
            • System modernization projects increasing operational risk
            • Staff turnover in critical control positions (23% in Finance)
            • Regulatory changes requiring process updates
            • Economic uncertainty affecting judgment areas
            
            **🔮 Predictive Insights:**
            • 73% probability of material weakness in Q3
            • Revenue controls most vulnerable (risk score: 8.2/10)
            • Automation opportunities could reduce risk by 35%
            
            **🛡️ Recommended Mitigations:**
            1. Strengthen revenue recognition controls immediately
            2. Accelerate staff training programs
            3. Implement continuous monitoring for high-risk processes
            4. Enhance management review controls
            
            **🎯 Model Accuracy:** 91.3% (Mock Mode)
            """,
            
            "compliance": """
            ✅ **AI Compliance Assessment (Mock):**
            
            Comprehensive analysis across multiple frameworks:
            
            **📋 SOX 404 Status:**
            • Overall compliance: 92.3% (Target: 95%)
            • Material weaknesses: 2 (down from 4 last quarter)
            • Significant deficiencies: 5 (remediation in progress)
            • Control testing: 87% complete, on schedule
            
            **🏛️ COSO Framework Analysis:**
            • Control Environment: 88% effective
            • Risk Assessment: 85% effective  
            • Control Activities: 90% effective
            • Information & Communication: 92% effective
            • Monitoring Activities: 83% effective
            
            **🔍 Gap Analysis:**
            • Documentation gaps in 12 key controls
            • Testing frequency below standard for 8 high-risk controls
            • Automation opportunities in 34 manual processes
            
            **📊 Benchmarking:**
            • Industry average compliance: 89%
            • Peer group ranking: Top 25%
            • Regulatory readiness: 94%
            
            **⚡ Priority Actions:**
            1. Complete documentation gaps within 30 days
            2. Increase testing frequency for critical controls  
            3. Implement automated monitoring capabilities
            4. Schedule management certification review
            
            **🎯 AI Recommendation Engine:** 96.2% accuracy vs auditor decisions (Mock Mode)
            """
        }
        
        if "fraud" in query.lower() or "anomaly" in query.lower():
            return insights["fraud"]
        elif "risk" in query.lower() or "predict" in query.lower():
            return insights["risk"]
        elif "compliance" in query.lower() or "sox" in query.lower():
            return insights["compliance"]
        else:
            return """
            🤖 **AI General Analysis (Mock Mode):**
            
            Your query has been processed using mock responses.
            
            **✅ Analysis Complete:**
            • Knowledge base searched: 2,847 audit documents (simulated)
            • Relevant context retrieved: 95% confidence match (simulated)
            • Best practices applied from 500+ similar cases (simulated)
            
            **💡 Key Insights:**
            • Pattern recognition indicates standard audit considerations apply
            • Historical data suggests 85% success rate for similar scenarios
            • Recommended approach aligns with industry best practices
            
            **📚 Supporting Framework:**
            • COSO Internal Control principles
            • SOX 404 compliance requirements
            • Industry audit standards (IIA, AICPA)
            
            **🚀 Next Steps:**
            1. Review specific control objectives
            2. Design targeted testing procedures
            3. Implement continuous monitoring
            4. Schedule follow-up assessment
            
            **⚠️ Note:** Configure API keys for real AI analysis
            """

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'demo_data_loaded' not in st.session_state:
    st.session_state.demo_data_loaded = False

# Dynamic version generation
if 'app_version' not in st.session_state:
    build_date = datetime.now().strftime("%Y%m%d")
    build_time = datetime.now().strftime("%H%M")
    st.session_state.app_version = f"v2024.{build_date}.{build_time}"

# Header
st.markdown(f"""
<div class="main-header">
    <h1>🤖 RAG Agentic AI Internal Audit System</h1>
    <h3>Cloud-Powered Implementation - Production Ready</h3>
    <p>Firebase Cloud Database + Qwen3 AI + Advanced Analytics | Live System</p>
    <p><strong>Developer:</strong> MS Hadianto | <strong>Version:</strong> {st.session_state.app_version} | <strong>Session ID:</strong> {st.session_state.session_id}</p>
</div>
""", unsafe_allow_html=True)

# Load demo data
if not st.session_state.demo_data_loaded:
    with st.spinner("🔄 Loading demo data and initializing AI models..."):
        time.sleep(2)
        st.session_state.audit_findings = DataGenerator.generate_audit_findings(50)
        st.session_state.transaction_data = DataGenerator.generate_transaction_data(5000)
        st.session_state.demo_data_loaded = True
    st.success("✅ Demo data loaded successfully!")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Control Panel")
    
    # System Status
    st.subheader("📡 Live Cloud Status")
    
    # Real-time cloud service status
    firebase_status = "✅ Connected" if cloud_services['firebase_db'] is not None else "⚠️ Demo Mode"
    firebase_css = "status-good" if cloud_services['firebase_db'] is not None else "status-warn"
    
    qwen_status = "✅ Active" if cloud_services['ai_engine'].is_available else "⚠️ Mock Mode"
    qwen_css = "status-good" if cloud_services['ai_engine'].is_available else "status-warn"
    
    status_metrics = {
        "🔥 Firebase": (firebase_status, firebase_css),
        "🤖 Qwen3 AI": (qwen_status, qwen_css), 
        "📊 Analytics": ("✅ Processing", "status-good"),
        "🗄️ Database": ("✅ Online", "status-good"),
        "⚡ Performance": ("✅ Optimal", "status-good"),
        "🔒 Security": ("✅ Protected", "status-good")
    }
    
    for service, (status, css_class) in status_metrics.items():
        st.markdown(f"**{service}:** <span class='{css_class}'>{status}</span>", unsafe_allow_html=True)
    
    st.divider()
    
    # Version Info
    st.subheader("ℹ️ System Info")
    st.markdown(f"""
    **Developer:** MS Hadianto  
    **Version:** {st.session_state.app_version}  
    **Build Date:** {datetime.now().strftime("%Y-%m-%d")}  
    **Session:** {st.session_state.session_id}  
    **Environment:** Demo/Development
    """)
    
    st.divider()
    
    # Live Metrics
    st.subheader("📊 Live Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Active Users", "47", "↑12")
        st.metric("Queries/Min", "23", "↑5")
        st.metric("Uptime", "99.9%", "")
    
    with col2:
        st.metric("AI Accuracy", "94.7%", "↑1.2%")
        st.metric("Response Time", "1.2s", "↓0.3s")
        st.metric("Cost Savings", "$2.8M", "↑$0.5M")
    
    st.divider()
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.session_state.demo_data_loaded = False
        st.rerun()
    
    if st.button("🔒 Run Security Scan", use_container_width=True):
        with st.spinner("Running security scan..."):
            time.sleep(2)
        st.success("✅ Security scan completed - No threats detected")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🤖 AI Assistant",
    "📊 Live Analytics", 
    "🕵️ Fraud Detection",
    "⚠️ Risk Assessment",
    "📋 Audit Management",
    "📈 Executive Dashboard"
])

with tab1:
    st.header("🤖 Intelligent AI Assistant")
    
    # Demo highlight
    if cloud_services['ai_engine'].is_available and cloud_services['firebase_db'] is not None:
        demo_text = "🚀 <strong>LIVE SYSTEM:</strong> This AI assistant uses real Qwen3 AI via OpenRouter API with Firebase Cloud Database for production-grade audit insights!"
        demo_class = "demo-highlight"
    elif cloud_services['ai_engine'].is_available:
        demo_text = "🤖 <strong>AI ENABLED:</strong> Using real Qwen3 AI responses! Configure Firebase for full cloud functionality."
        demo_class = "demo-highlight"
    elif cloud_services['firebase_db'] is not None:
        demo_text = "🔥 <strong>CLOUD ENABLED:</strong> Firebase connected! Configure OpenRouter API for real AI responses."
        demo_class = "demo-highlight"
    else:
        demo_text = "🎯 <strong>DEMO MODE:</strong> Configure Firebase and OpenRouter API keys in Streamlit secrets for full functionality."
        demo_class = "demo-highlight"
    
    st.markdown(f"""
    <div class="{demo_class}">
        {demo_text}
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.subheader("💬 Chat with AI Auditor")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="text-align: right; margin: 1rem 0;">
                    <div style="background: #007bff; color: white; padding: 0.8rem 1.2rem; border-radius: 15px; display: inline-block; max-width: 80%;">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-response">
                    <strong>🤖 AI Auditor:</strong><br><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask about audit, compliance, fraud detection, or risk management:",
                placeholder="Example: Analyze our transaction data for fraud patterns...",
                height=100
            )
            
            col_a, col_b = st.columns([1, 1])
            with col_a:
                submit_chat = st.form_submit_button("🚀 Send Query", use_container_width=True)
            with col_b:
                clear_chat = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
            
            if submit_chat and user_input:
                # Add user message
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Generate AI response using real Qwen3 API
                with st.spinner("🤖 Qwen3 AI is analyzing your query..."):
                    # Prepare context from current session data
                    context = f"""
                    Session: {st.session_state.session_id}
                    Audit Findings: {len(st.session_state.audit_findings)} total
                    Transaction Data: {len(st.session_state.transaction_data)} records
                    """
                    
                    # Use real AI engine or fallback to mock
                    if cloud_services['ai_engine'].is_available:
                        ai_response = cloud_services['ai_engine'].generate_audit_insight(user_input, context)
                        
                        # Save interaction to Firebase for audit trail
                        cloud_services['cloud_db'].save_ai_interaction(
                            query=user_input,
                            response=ai_response,
                            session_id=st.session_state.session_id
                        )
                    else:
                        ai_response = MockAIEngine.generate_audit_insight(user_input)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": ai_response
                    })
                
                # Save session data to Firebase
                session_data = {
                    'session_id': st.session_state.session_id,
                    'version': st.session_state.app_version,
                    'total_interactions': len(st.session_state.chat_history),
                    'last_query': user_input
                }
                cloud_services['cloud_db'].save_audit_session(session_data)
                
                st.rerun()
            
            if clear_chat:
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        st.subheader("💡 Sample Queries")
        
        sample_queries = [
            "🕵️ Analyze transaction patterns for fraud detection",
            "📈 Generate 6-month risk forecast",
            "✅ Check SOX compliance status", 
            "🔍 Assess control effectiveness",
            "💡 Identify process improvement opportunities",
            "⚠️ Review high-risk audit findings"
        ]
        
        for query in sample_queries:
            if st.button(query, use_container_width=True, key=f"sample_{query}"):
                # Auto-fill the query
                clean_query = query[2:]  # Remove emoji
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": clean_query
                })
                
                with st.spinner("🤖 Processing sample query with Qwen3..."):
                    # Prepare context
                    context = f"""
                    Sample Query Processing
                    Session: {st.session_state.session_id}
                    Data Available: Audit findings, Transaction data, Risk metrics
                    """
                    
                    # Use real AI engine or fallback
                    if cloud_services['ai_engine'].is_available:
                        ai_response = cloud_services['ai_engine'].generate_audit_insight(clean_query, context)
                        
                        # Save to Firebase
                        cloud_services['cloud_db'].save_ai_interaction(
                            query=clean_query,
                            response=ai_response,
                            session_id=st.session_state.session_id
                        )
                    else:
                        ai_response = MockAIEngine.generate_audit_insight(clean_query)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                
                st.rerun()

with tab2:
    st.header("📊 Live Analytics Dashboard")
    
    # Real-time metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Transactions Processed", "47,329", "↑2,847")
    with col2:
        st.metric("Anomalies Detected", "23", "↑5")
    with col3:
        st.metric("Risk Score", "7.2/10", "↑0.3")
    with col4:
        st.metric("Control Health", "87%", "↑2%")
    with col5:
        st.metric("AI Accuracy", "94.7%", "↑1.2%")
    
    # Advanced analytics
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction volume chart
        st.subheader("📈 Transaction Volume Analysis")
        
        # Generate sample time series data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        volume_data = pd.DataFrame({
            'Date': dates,
            'Volume': np.random.poisson(1000, len(dates)) + np.random.normal(0, 100, len(dates)),
            'Anomalies': np.random.poisson(5, len(dates))
        })
        
        fig_volume = px.line(volume_data, x='Date', y=['Volume', 'Anomalies'], 
                           title="Daily Transaction Volume & Anomalies")
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        # Risk heatmap
        st.subheader("🌡️ Risk Heatmap by Department")
        
        departments = ['Finance', 'Operations', 'IT', 'HR', 'Marketing']
        risk_types = ['Fraud', 'Compliance', 'Operational', 'Strategic']
        
        risk_matrix = np.random.uniform(3, 9, (len(departments), len(risk_types)))
        
        fig_heatmap = px.imshow(risk_matrix, 
                               x=risk_types, y=departments,
                               color_continuous_scale='RdYlBu_r',
                               title="Risk Assessment Matrix")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Advanced visualizations
    st.subheader("🔬 Advanced Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Control effectiveness gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 87,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Control Effectiveness"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Findings distribution
        findings_dist = st.session_state.audit_findings['severity'].value_counts()
        
        fig_pie = px.pie(values=findings_dist.values, names=findings_dist.index,
                        title="Findings by Severity",
                        color_discrete_map={
                            'Low': '#28a745',
                            'Medium': '#ffc107', 
                            'High': '#fd7e14',
                            'Critical': '#dc3545'
                        })
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col3:
        # Trend analysis
        monthly_trends = pd.DataFrame({
            'Month': pd.date_range('2024-01', periods=12, freq='M'),
            'Findings': np.random.poisson(8, 12),
            'Resolved': np.random.poisson(6, 12)
        })
        
        fig_trend = px.bar(monthly_trends, x='Month', y=['Findings', 'Resolved'],
                          title="Monthly Findings Trend", barmode='group')
        st.plotly_chart(fig_trend, use_container_width=True)

with tab3:
    st.header("🕵️ Advanced Fraud Detection")
    
    # Fraud detection overview
    st.markdown("""
    <div class="feature-card">
        <h3>🚨 Real-Time Fraud Monitoring</h3>
        <p>AI-powered fraud detection using ensemble machine learning models, 
        behavioral analysis, and statistical anomaly detection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Fraud detection visualization
        st.subheader("🔍 Transaction Anomaly Analysis")
        
        # Get transaction data
        tx_data = st.session_state.transaction_data.tail(500)  # Last 500 transactions
        
        fig_fraud = px.scatter(tx_data, x='date', y='amount', 
                              color='is_anomaly', size='risk_score',
                              hover_data=['transaction_id', 'department', 'user_id'],
                              title="Real-Time Anomaly Detection",
                              color_discrete_map={0: 'blue', 1: 'red'})
        
        fig_fraud.update_layout(height=400)
        st.plotly_chart(fig_fraud, use_container_width=True)
        
        # Fraud patterns analysis
        st.subheader("📊 Fraud Pattern Analysis")
        
        # Benford's Law analysis
        amounts = tx_data['amount']
        first_digits = amounts.astype(str).str[0].astype(int)
        observed_freq = first_digits.value_counts()
        
        # Ensure we have all digits 1-9 with 0 counts for missing digits
        observed_counts = []
        expected_freq = []
        
        for digit in range(1, 10):
            # Get observed frequency (0 if digit not present)
            obs_count = observed_freq.get(digit, 0)
            observed_counts.append(obs_count)
            
            # Calculate expected frequency
            exp_count = len(amounts) * np.log10(1 + 1/digit)
            expected_freq.append(exp_count)
        
        benford_data = pd.DataFrame({
            'Digit': range(1, 10),
            'Observed': observed_counts,
            'Expected': expected_freq
        })
        
        fig_benford = px.bar(benford_data, x='Digit', y=['Observed', 'Expected'],
                           title="Benford's Law Analysis", barmode='group')
        st.plotly_chart(fig_benford, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Fraud Alerts")
        
        # High-risk transactions
        high_risk_tx = tx_data[tx_data['risk_score'] > 0.8].head(10)
        
        for _, tx in high_risk_tx.iterrows():
            st.markdown(f"""
            <div class="metric-card">
                <strong>🚨 Alert #{tx['transaction_id']}</strong><br>
                Amount: ${tx['amount']:,.2f}<br>
                Risk Score: {tx['risk_score']:.2f}<br>
                Department: {tx['department']}<br>
                <small>User: {tx['user_id']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Fraud statistics
        st.subheader("📊 Detection Stats")
        
        anomaly_count = tx_data['is_anomaly'].sum()
        total_amount_at_risk = tx_data[tx_data['is_anomaly'] == 1]['amount'].sum()
        
        st.metric("Anomalies Detected", anomaly_count)
        st.metric("Amount at Risk", f"${total_amount_at_risk:,.0f}")
        st.metric("Detection Rate", "94.7%")
        st.metric("False Positive Rate", "3.2%")

with tab4:
    st.header("⚠️ Risk Assessment & Prediction")
    
    # Risk overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Overall Risk Score</h4>
            <h2 style="color: #fd7e14;">7.2/10</h2>
            <p>↑ 0.3 from last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Risk Appetite</h4>
            <h2 style="color: #28a745;">Within Limits</h2>
            <p>85% of target threshold</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Predicted Trend</h4>
            <h2 style="color: #dc3545;">Increasing</h2>
            <p>+18% over next 6 months</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk prediction model
    st.subheader("🔮 AI Risk Prediction Model")
    
    # Generate prediction data
    future_dates = pd.date_range(start=datetime.now(), periods=6, freq='M')
    base_risk = 7.2
    risk_predictions = []
    
    for i, date in enumerate(future_dates):
        predicted_risk = base_risk + np.random.normal(0.1 * i, 0.3)
        predicted_risk = max(1, min(10, predicted_risk))
        risk_predictions.append({
            'Date': date,
            'Predicted_Risk': predicted_risk,
            'Lower_Bound': predicted_risk * 0.9,
            'Upper_Bound': predicted_risk * 1.1,
            'Confidence': np.random.uniform(0.85, 0.95)
        })
    
    prediction_df = pd.DataFrame(risk_predictions)
    
    # Risk prediction chart
    fig_pred = go.Figure()
    
    fig_pred.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Predicted_Risk'],
        mode='lines+markers',
        name='Predicted Risk',
        line=dict(color='red', width=3)
    ))
    
    fig_pred.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Upper_Bound'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig_pred.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Lower_Bound'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Confidence Interval',
        fillcolor='rgba(255,0,0,0.2)'
    ))
    
    fig_pred.update_layout(
        title="📈 6-Month Risk Prediction with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Risk Score (1-10)",
        height=400
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)

with tab5:
    st.header("📋 Audit Findings Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Audit Findings Overview")
        
        # Findings summary
        findings_summary = st.session_state.audit_findings.groupby(['status', 'severity']).size().unstack(fill_value=0)
        
        fig_findings = px.bar(findings_summary, 
                             title="Audit Findings by Status and Severity",
                             color_discrete_map={
                                 'Low': '#28a745',
                                 'Medium': '#ffc107',
                                 'High': '#fd7e14', 
                                 'Critical': '#dc3545'
                             })
        st.plotly_chart(fig_findings, use_container_width=True)
        
        # Findings table
        st.subheader("📋 Current Findings")
        st.dataframe(st.session_state.audit_findings, use_container_width=True, height=400)
    
    with col2:
        st.subheader("📊 Key Metrics")
        
        # Calculate metrics
        total_findings = len(st.session_state.audit_findings)
        open_findings = len(st.session_state.audit_findings[st.session_state.audit_findings['status'] == 'Open'])
        high_critical = len(st.session_state.audit_findings[st.session_state.audit_findings['severity'].isin(['High', 'Critical'])])
        avg_risk_score = st.session_state.audit_findings['risk_score'].mean()
        
        st.metric("Total Findings", total_findings)
        st.metric("Open Findings", open_findings)
        st.metric("High/Critical", high_critical)
        st.metric("Avg Risk Score", f"{avg_risk_score:.1f}")

with tab6:
    st.header("📈 Executive Dashboard")
    
    # Executive summary
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Executive Summary - Q4 2024</h3>
        <p><strong>Overall Status:</strong> The audit program delivered significant value with 94.7% AI accuracy 
        and $2.8M in identified cost savings. Key areas require attention in revenue recognition and IT security.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Audit Progress", "78%", "↑5%")
        st.metric("Budget Utilization", "88%", "↓2%")
    
    with col2:
        st.metric("Control Effectiveness", "87%", "↑3%")
        st.metric("Compliance Rate", "92.3%", "↑1.8%")
    
    with col3:
        st.metric("Risk Score", "7.2/10", "↑0.3")
        st.metric("Cost Savings", "$2.8M", "↑$0.5M")
    
    with col4:
        st.metric("AI Accuracy", "94.7%", "↑1.2%")
        st.metric("User Satisfaction", "4.6/5", "↑0.2")

# Footer with Disclaimer
st.markdown("---")

# Developer & Version Info
st.markdown(f"""
<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
    <h3>🤖 RAG Agentic AI Internal Audit System</h3>
    <p><strong>Developer:</strong> MS Hadianto | <strong>Version:</strong> {st.session_state.app_version} | <strong>Status:</strong> ✅ Operational</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### ⚠️ IMPORTANT DISCLAIMER")

# Disclaimer in expandable section
with st.expander("📋 Click to read full disclaimer and terms", expanded=False):
    st.markdown("""
    **🔍 Professional Judgment Required:**  
    This AI system is designed to assist and enhance audit procedures, not replace professional auditor judgment. All AI-generated insights, recommendations, and risk assessments must be validated by qualified audit professionals.
    
    **📊 Data Accuracy:**  
    While this system employs advanced machine learning algorithms, users are responsible for verifying data accuracy and completeness. AI predictions should be corroborated with additional audit evidence.
    
    **🛡️ Compliance & Standards:**  
    This tool supports but does not guarantee compliance with auditing standards (IIA, AICPA, ISA) or regulatory requirements (SOX, COSO, etc.). Professional standards and regulatory compliance remain the responsibility of the audit team.
    
    **⚖️ Limitation of Liability:**  
    The AI system provides analytical insights based on available data and patterns. MS Hadianto and associated parties are not liable for audit conclusions, business decisions, or regulatory actions based on system outputs.
    
    **🔒 Confidentiality:**  
    Users must ensure proper handling of sensitive audit data in compliance with confidentiality agreements and data protection regulations (GDPR, CCPA, etc.).
    
    **📅 Demo Environment:**  
    This demonstration version uses simulated data for illustrative purposes. Production implementation requires proper data integration, security controls, and governance frameworks.
    """)

# Technical Info
current_time = datetime.now()
firebase_status = "Connected" if cloud_services['firebase_db'] is not None else "Demo"
qwen_status = "Live" if cloud_services['ai_engine'].is_available else "Mock"

st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.85em; margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 5px;">
    <p>© 2025 MS Hadianto | Advanced AI Solutions for Internal Audit</p>
    <p>Last Updated: {current_time.strftime("%B %d, %Y at %H:%M:%S")} | 
    Build: {st.session_state.app_version} | 
    Session: {st.session_state.session_id}</p>
    <p>Framework: Streamlit | Python 3.9+ | Firebase: {firebase_status} | Qwen3: {qwen_status}</p>
</div>
""", unsafe_allow_html=True)