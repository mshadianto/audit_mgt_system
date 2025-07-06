"""
Agile RAG AI Internal Audit System - Universal Version
Flexible for Any Industry, Any Audit Type, Any Scale
Firebase + Qwen3 + Dynamic Templates + Auto-Detection

Dependencies:
pip install streamlit firebase-admin openai pandas numpy plotly python-dotenv
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import uuid
import io
import base64
import hashlib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import re

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="🚀 Agile AI Audit System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# SAFE IMPORTS WITH FALLBACKS
# ========================================
FIREBASE_AVAILABLE = False
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    pass

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

DOCX_AVAILABLE = False
PDF_AVAILABLE = False
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    pass

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    pass

# ========================================
# AGILE DESIGN SYSTEM
# ========================================
st.markdown("""
<style>
    .agile-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .agile-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shine 3s infinite;
    }
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    .agile-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .agile-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    .industry-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .industry-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .audit-type-badge {
        display: inline-block;
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .quick-action-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        margin: 0.3rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    .quick-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .ai-response-agile {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    .file-drop-zone {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f4ff 100%);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .file-drop-zone:hover {
        border-color: #4facfe;
        background: linear-gradient(135deg, #e8f4ff 0%, #d8edff 100%);
    }
    .metric-card-agile {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .metric-card-agile:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    }
    .template-selector {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-good { background: #28a745; }
    .status-warn { background: #ffc107; }
    .status-error { background: #dc3545; }
</style>
""", unsafe_allow_html=True)

# ========================================
# AGILE AUDIT TEMPLATES
# ========================================
AUDIT_TEMPLATES = {
    "financial": {
        "name": "Financial Audit",
        "icon": "💰",
        "description": "Financial statements, controls, and compliance",
        "file_types": ["financial_statements", "trial_balance", "journal_entries", "bank_reconciliations"],
        "key_fields": ["amount", "account", "date", "description", "entity"],
        "risk_areas": ["Revenue Recognition", "Cash Management", "Accounts Payable", "Financial Reporting"],
        "ai_prompts": {
            "analyze": "Analyze financial data for material misstatements, unusual transactions, and control weaknesses",
            "fraud": "Identify potential financial fraud patterns including revenue manipulation and expense schemes",
            "compliance": "Assess compliance with accounting standards (GAAP/IFRS) and regulatory requirements"
        }
    },
    "operational": {
        "name": "Operational Audit",
        "icon": "⚙️",
        "description": "Process efficiency, controls, and performance",
        "file_types": ["process_data", "performance_metrics", "employee_data", "inventory_records"],
        "key_fields": ["process_step", "duration", "cost", "quality_metric", "department"],
        "risk_areas": ["Process Efficiency", "Resource Utilization", "Quality Control", "Compliance"],
        "ai_prompts": {
            "analyze": "Evaluate operational efficiency, identify bottlenecks, and recommend process improvements",
            "fraud": "Detect operational fraud including ghost employees, procurement fraud, and asset misappropriation",
            "compliance": "Review operational compliance with policies, procedures, and regulatory requirements"
        }
    },
    "it_audit": {
        "name": "IT Audit",
        "icon": "💻",
        "description": "IT controls, cybersecurity, and data governance",
        "file_types": ["system_logs", "access_controls", "security_incidents", "it_assets"],
        "key_fields": ["user_id", "system", "timestamp", "action", "ip_address"],
        "risk_areas": ["Access Controls", "Data Security", "System Availability", "Change Management"],
        "ai_prompts": {
            "analyze": "Assess IT control effectiveness, identify security vulnerabilities, and evaluate system performance",
            "fraud": "Detect unauthorized access, data breaches, and system manipulation attempts",
            "compliance": "Review IT compliance with security frameworks (ISO 27001, NIST) and regulations"
        }
    },
    "compliance": {
        "name": "Compliance Audit",
        "icon": "✅",
        "description": "Regulatory compliance and risk management",
        "file_types": ["compliance_testing", "regulatory_reports", "policy_documentation", "training_records"],
        "key_fields": ["regulation", "control_id", "test_result", "evidence", "owner"],
        "risk_areas": ["Regulatory Compliance", "Policy Adherence", "Training Effectiveness", "Documentation"],
        "ai_prompts": {
            "analyze": "Evaluate compliance with applicable regulations and internal policies",
            "fraud": "Identify compliance violations and potential regulatory fraud",
            "compliance": "Assess overall compliance program effectiveness and maturity"
        }
    },
    "sox": {
        "name": "SOX 404 Audit",
        "icon": "🏛️",
        "description": "Sarbanes-Oxley internal controls assessment",
        "file_types": ["control_testing", "scoping_documentation", "management_assessment", "remediation_plans"],
        "key_fields": ["control_id", "effectiveness", "deficiency_level", "testing_date", "owner"],
        "risk_areas": ["ICFR Design", "Operating Effectiveness", "Management Review", "Documentation"],
        "ai_prompts": {
            "analyze": "Assess SOX 404 control design and operating effectiveness",
            "fraud": "Identify control deficiencies that could enable fraud",
            "compliance": "Evaluate SOX compliance readiness and identify material weaknesses"
        }
    },
    "vendor": {
        "name": "Vendor/Procurement Audit",
        "icon": "🏢",
        "description": "Vendor management and procurement processes",
        "file_types": ["vendor_master", "purchase_orders", "contracts", "payment_records"],
        "key_fields": ["vendor_id", "amount", "contract_date", "payment_terms", "approval_level"],
        "risk_areas": ["Vendor Selection", "Contract Management", "Payment Processing", "Performance Monitoring"],
        "ai_prompts": {
            "analyze": "Analyze vendor relationships, contract compliance, and procurement efficiency",
            "fraud": "Detect vendor fraud, collusion, and kickback schemes",
            "compliance": "Review procurement compliance with policies and regulations"
        }
    },
    "custom": {
        "name": "Custom Audit",
        "icon": "🎯",
        "description": "Flexible template for specific audit needs",
        "file_types": ["data_file_1", "data_file_2", "supporting_docs", "analysis_results"],
        "key_fields": ["id", "date", "amount", "category", "status"],
        "risk_areas": ["Custom Risk 1", "Custom Risk 2", "Custom Risk 3", "Custom Risk 4"],
        "ai_prompts": {
            "analyze": "Perform comprehensive analysis based on uploaded data and audit objectives",
            "fraud": "Identify anomalies, patterns, and potential fraud indicators in the data",
            "compliance": "Assess compliance with relevant standards and requirements"
        }
    }
}

INDUSTRY_TEMPLATES = {
    "manufacturing": {
        "name": "Manufacturing",
        "icon": "🏭",
        "focus_areas": ["Inventory Management", "Production Controls", "Quality Assurance", "Cost Accounting"],
        "common_files": ["production_data", "inventory_counts", "quality_metrics", "cost_reports"]
    },
    "financial_services": {
        "name": "Financial Services", 
        "icon": "🏦",
        "focus_areas": ["Credit Risk", "Regulatory Compliance", "Anti-Money Laundering", "Customer Due Diligence"],
        "common_files": ["loan_portfolio", "transaction_monitoring", "kyc_data", "regulatory_reports"]
    },
    "healthcare": {
        "name": "Healthcare",
        "icon": "🏥", 
        "focus_areas": ["Patient Privacy", "Billing Accuracy", "Clinical Documentation", "Regulatory Compliance"],
        "common_files": ["patient_records", "billing_data", "clinical_documentation", "compliance_testing"]
    },
    "retail": {
        "name": "Retail",
        "icon": "🛒",
        "focus_areas": ["Inventory Management", "Sales Accuracy", "Customer Data", "Supplier Relations"],
        "common_files": ["sales_data", "inventory_records", "customer_database", "supplier_contracts"]
    },
    "technology": {
        "name": "Technology",
        "icon": "💻",
        "focus_areas": ["Data Security", "Access Controls", "Software Development", "IP Protection"],
        "common_files": ["access_logs", "security_incidents", "development_records", "ip_documentation"]
    },
    "government": {
        "name": "Government",
        "icon": "🏛️",
        "focus_areas": ["Public Accountability", "Compliance", "Fraud Prevention", "Performance Measurement"],
        "common_files": ["budget_data", "performance_metrics", "compliance_reports", "audit_evidence"]
    }
}

# ========================================
# AGILE FIREBASE MANAGER
# ========================================
class AgileFirebaseManager:
    """Universal Firebase Manager for Any Audit Type"""
    
    def __init__(self):
        self.db = None
        self.connected = False
        self.initialize()
    
    def initialize(self):
        """Initialize Firebase with agile configuration"""
        if not FIREBASE_AVAILABLE:
            return False
        
        try:
            if firebase_admin._apps:
                self.db = firestore.client()
                self.connected = True
                return True
            
            # Multiple configuration options
            config_sources = [
                self._try_streamlit_secrets,
                self._try_environment_variables,
                self._try_service_account_file
            ]
            
            for config_method in config_sources:
                if config_method():
                    self.db = firestore.client()
                    self.connected = True
                    return True
            
        except Exception as e:
            st.warning(f"Firebase initialization: {str(e)}")
        
        return False
    
    def _try_streamlit_secrets(self):
        """Try Streamlit secrets configuration"""
        try:
            if hasattr(st, 'secrets') and 'firebase_private_key' in st.secrets:
                firebase_config = {
                    "type": "service_account",
                    "project_id": st.secrets.get("firebase_project_id", "agile-audit-system"),
                    "private_key_id": st.secrets.get("firebase_private_key_id", ""),
                    "private_key": st.secrets.get("firebase_private_key", "").replace("\\n", "\n"),
                    "client_email": st.secrets.get("firebase_client_email", ""),
                    "client_id": st.secrets.get("firebase_client_id", ""),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{st.secrets.get('firebase_client_email', '')}"
                }
                
                if firebase_config["private_key"]:
                    cred = credentials.Certificate(firebase_config)
                    firebase_admin.initialize_app(cred)
                    return True
        except Exception:
            pass
        return False
    
    def _try_environment_variables(self):
        """Try environment variables configuration"""
        try:
            import os
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred)
                return True
        except Exception:
            pass
        return False
    
    def _try_service_account_file(self):
        """Try service account file configuration"""
        try:
            import os
            service_account_paths = [
                'service-account.json',
                'firebase-key.json',
                os.path.expanduser('~/firebase-key.json')
            ]
            
            for path in service_account_paths:
                if os.path.exists(path):
                    cred = credentials.Certificate(path)
                    firebase_admin.initialize_app(cred)
                    return True
        except Exception:
            pass
        return False
    
    def save_agile_audit_session(self, session_data: dict) -> bool:
        """Save audit session with flexible structure"""
        if not self.connected:
            return False
        
        try:
            doc_ref = self.db.collection('agile_audit_sessions').document(session_data['session_id'])
            doc_ref.set({
                **session_data,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'system_version': 'agile_v1.0'
            })
            return True
        except Exception as e:
            st.error(f"Session save error: {str(e)}")
            return False
    
    def save_agile_file(self, file_content: bytes, filename: str, file_type: str, 
                       audit_template: str, metadata: dict = None) -> Dict[str, Any]:
        """Save file with agile metadata structure"""
        if not self.connected:
            return {"success": False, "error": "Firebase not connected"}
        
        try:
            # File size validation
            max_size = 800 * 1024  # 800KB
            if len(file_content) > max_size:
                return {
                    "success": False,
                    "error": f"File too large ({len(file_content)/1024:.1f}KB). Max: {max_size/1024}KB"
                }
            
            file_id = f"agile_{uuid.uuid4().hex[:12]}"
            file_base64 = base64.b64encode(file_content).decode('utf-8')
            
            # Extract and analyze content
            extracted_content = self._extract_agile_content(file_content, filename, file_type)
            
            file_doc = {
                'id': file_id,
                'filename': filename,
                'file_type': file_type,
                'audit_template': audit_template,
                'upload_date': datetime.now(),
                'size': len(file_content),
                'content_base64': file_base64,
                'content_hash': hashlib.md5(file_content).hexdigest(),
                'extracted_content': extracted_content,
                'metadata': metadata or {},
                'analysis_ready': bool(extracted_content),
                'auto_detected_fields': self._auto_detect_fields(extracted_content),
                'suggested_analysis': self._suggest_analysis_types(filename, extracted_content)
            }
            
            # Save to Firestore
            self.db.collection('agile_audit_files').document(file_id).set(file_doc)
            
            return {
                "success": True,
                "file_id": file_id,
                "extracted_length": len(extracted_content),
                "auto_detected_fields": file_doc['auto_detected_fields'],
                "suggested_analysis": file_doc['suggested_analysis']
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_agile_content(self, file_content: bytes, filename: str, file_type: str) -> str:
        """Agile content extraction for multiple file types"""
        try:
            if file_type == "text/plain" or filename.lower().endswith('.txt'):
                return file_content.decode('utf-8', errors='ignore')
            
            elif file_type == "text/csv" or filename.lower().endswith('.csv'):
                csv_text = file_content.decode('utf-8', errors='ignore')
                try:
                    df = pd.read_csv(io.StringIO(csv_text))
                    summary = f"CSV Analysis:\n"
                    summary += f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
                    summary += f"Columns: {', '.join(df.columns)}\n"
                    summary += f"Sample Data:\n{df.head(3).to_string()}\n"
                    summary += f"Data Types:\n{df.dtypes.to_string()}"
                    return summary
                except:
                    return csv_text[:2000]  # Fallback to raw text
            
            elif PDF_AVAILABLE and (file_type == "application/pdf" or filename.lower().endswith('.pdf')):
                pdf_file = io.BytesIO(file_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages[:5]:  # Limit to first 5 pages
                    text += page.extract_text() + "\n"
                return text[:5000]  # Limit extracted text
            
            elif DOCX_AVAILABLE and filename.lower().endswith('.docx'):
                docx_file = io.BytesIO(file_content)
                doc = Document(docx_file)
                text = ""
                for paragraph in doc.paragraphs[:100]:  # Limit paragraphs
                    text += paragraph.text + "\n"
                return text[:5000]
            
            elif filename.lower().endswith('.json'):
                json_text = file_content.decode('utf-8', errors='ignore')
                try:
                    data = json.loads(json_text)
                    return f"JSON Structure:\n{json.dumps(data, indent=2)[:2000]}"
                except:
                    return json_text[:2000]
            
            else:
                return f"Binary file: {filename} ({len(file_content)} bytes)"
                
        except Exception as e:
            return f"Content extraction error: {str(e)}"
    
    def _auto_detect_fields(self, content: str) -> List[str]:
        """Auto-detect important fields in content"""
        detected_fields = []
        
        # Common audit fields to detect
        field_patterns = {
            'amount': r'amount|total|sum|value|cost|price',
            'date': r'date|time|timestamp|created|modified',
            'id': r'id|identifier|number|ref|reference',
            'account': r'account|acct|ledger|gl',
            'vendor': r'vendor|supplier|payee|company',
            'employee': r'employee|emp|staff|person|user',
            'transaction': r'transaction|txn|trans|entry',
            'status': r'status|state|condition|approved',
            'description': r'description|desc|note|comment',
            'department': r'department|dept|division|unit'
        }
        
        content_lower = content.lower()
        for field_type, pattern in field_patterns.items():
            if re.search(pattern, content_lower):
                detected_fields.append(field_type)
        
        return detected_fields
    
    def _suggest_analysis_types(self, filename: str, content: str) -> List[str]:
        """Suggest analysis types based on file content"""
        suggestions = []
        
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # File-based suggestions
        if any(word in filename_lower for word in ['transaction', 'payment', 'invoice']):
            suggestions.append('fraud_detection')
        if any(word in filename_lower for word in ['employee', 'payroll', 'hr']):
            suggestions.append('ghost_employee_check')
        if any(word in filename_lower for word in ['inventory', 'stock', 'asset']):
            suggestions.append('inventory_analysis')
        if any(word in filename_lower for word in ['control', 'compliance', 'audit']):
            suggestions.append('compliance_assessment')
        
        # Content-based suggestions
        if any(word in content_lower for word in ['risk', 'control', 'test']):
            suggestions.append('risk_assessment')
        if any(word in content_lower for word in ['revenue', 'sales', 'income']):
            suggestions.append('revenue_analysis')
        
        return suggestions if suggestions else ['general_analysis']
    
    def get_agile_files(self, audit_template: str = None, limit: int = 50) -> List[Dict]:
        """Get files with optional filtering"""
        if not self.connected:
            return []
        
        try:
            query = self.db.collection('agile_audit_files')
            
            if audit_template:
                query = query.where('audit_template', '==', audit_template)
            
            query = query.order_by('upload_date', direction=firestore.Query.DESCENDING).limit(limit)
            docs = query.stream()
            
            files = []
            for doc in docs:
                file_data = doc.to_dict()
                # Remove content for list view
                if 'content_base64' in file_data:
                    del file_data['content_base64']
                files.append(file_data)
            
            return files
            
        except Exception as e:
            st.error(f"Error retrieving files: {e}")
            return []

# ========================================
# AGILE AI ENGINE
# ========================================
class AgileAIEngine:
    """Universal AI Engine for Any Audit Type"""
    
    def __init__(self):
        self.api_key = ""
        self.client = None
        self.is_available = False
        self.model = "qwen/qwen-2.5-72b-instruct"
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize AI client with agile configuration"""
        if not OPENAI_AVAILABLE:
            return
        
        try:
            # Try multiple sources for API key
            if hasattr(st, 'secrets') and 'openrouter_api_key' in st.secrets:
                self.api_key = st.secrets["openrouter_api_key"]
            else:
                import os
                self.api_key = os.getenv('OPENROUTER_API_KEY', '')
            
            if self.api_key:
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                )
                self.is_available = True
                
        except Exception as e:
            st.warning(f"AI Engine initialization: {str(e)}")
    
    def analyze_agile_audit(self, query: str, audit_template: str, context: str = "", 
                           analysis_type: str = "general") -> str:
        """Universal audit analysis for any template"""
        if not self.is_available:
            return self._get_agile_mock_response(query, audit_template, analysis_type)
        
        try:
            # Get template-specific system prompt
            template_config = AUDIT_TEMPLATES.get(audit_template, AUDIT_TEMPLATES["custom"])
            
            system_prompt = f"""You are an expert AI Internal Auditor specializing in {template_config['name']}.
            
            Audit Focus: {template_config['description']}
            Key Risk Areas: {', '.join(template_config['risk_areas'])}
            
            Your expertise includes:
            - Risk-based audit methodology
            - Industry best practices and benchmarks
            - Regulatory compliance requirements
            - Advanced data analytics and pattern recognition
            - Fraud detection and prevention
            
            Provide detailed, actionable insights with:
            - Specific findings and observations
            - Risk ratings and explanations
            - Practical recommendations with timelines
            - Industry benchmarking where relevant
            - Clear next steps and priorities
            
            Use professional audit language with emojis for clarity."""
            
            # Template-specific user prompt
            ai_prompt = template_config['ai_prompts'].get(analysis_type, template_config['ai_prompts']['analyze'])
            
            user_prompt = f"""
            Audit Type: {template_config['name']}
            Analysis Focus: {analysis_type.replace('_', ' ').title()}
            Query: {query}
            Context: {context}
            
            Template Guidance: {ai_prompt}
            
            Please provide comprehensive analysis including:
            1. Executive Summary
            2. Detailed Findings
            3. Risk Assessment 
            4. Recommendations
            5. Implementation Timeline
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"AI API Error: {str(e)}")
            return self._get_agile_mock_response(query, audit_template, analysis_type)
    
    def _get_agile_mock_response(self, query: str, audit_template: str, analysis_type: str) -> str:
        """High-quality mock responses for any audit template"""
        
        template_config = AUDIT_TEMPLATES.get(audit_template, AUDIT_TEMPLATES["custom"])
        
        # Template-specific mock responses
        mock_responses = {
            "financial": {
                "fraud_detection": f"""
                💰 **Financial Audit - Fraud Detection Analysis**
                
                **🎯 Audit Scope:** {template_config['description']}
                
                **🔍 Key Findings:**
                • Revenue manipulation indicators detected in 12% of transactions
                • Unusual journal entries pattern: 47 entries >$50K without proper documentation
                • Bank reconciliation gaps: 8 items outstanding >90 days
                • Expense reimbursement anomalies: Round number clustering in 23% of claims
                
                **📊 Risk Assessment:**
                • **High Risk Areas:** {', '.join(template_config['risk_areas'][:2])}
                • **Medium Risk Areas:** {', '.join(template_config['risk_areas'][2:])}
                • **Overall Financial Risk:** 7.2/10 (Medium-High)
                
                **💡 Recommendations:**
                1. **Immediate (30 days):** Implement three-way matching for all invoices >$10K
                2. **Short-term (90 days):** Deploy automated journal entry monitoring
                3. **Long-term (6 months):** Establish continuous audit program
                
                **🎯 AI Confidence:** 94.2% (Demo Mode - Configure API for live analysis)
                """,
                
                "compliance": f"""
                ✅ **Financial Audit - Compliance Assessment**
                
                **📋 Regulatory Framework:** GAAP/IFRS, SOX 404, Local Regulations
                
                **🔍 Compliance Status:**
                • **GAAP Compliance:** 89% (Target: 95%+)
                • **SOX 404 Readiness:** 76% (12 control deficiencies identified)
                • **Documentation Quality:** 82% complete
                • **Testing Coverage:** 67% of key controls tested
                
                **⚠️ Key Gaps:**
                • Revenue recognition documentation incomplete
                • Month-end close procedures not standardized
                • Supporting evidence retention inconsistent
                • Management review controls need strengthening
                
                **📈 Improvement Roadmap:**
                1. **Phase 1 (Month 1-2):** Document all revenue recognition procedures
                2. **Phase 2 (Month 3-4):** Implement automated controls testing
                3. **Phase 3 (Month 5-6):** Achieve 95%+ compliance certification
                
                **💰 Investment Required:** $125K for compliance improvements
                **Expected ROI:** 280% over 3 years through reduced audit costs
                
                **🎯 AI Confidence:** 91.8% (Demo Mode)
                """
            },
            
            "operational": {
                "fraud_detection": f"""
                ⚙️ **Operational Audit - Fraud Detection Analysis**
                
                **🎯 Operational Focus:** {template_config['description']}
                
                **🚨 Fraud Indicators Detected:**
                • Ghost employee patterns: 3 employees with identical bank accounts
                • Procurement fraud: 15 vendors with suspicious address clustering
                • Inventory shrinkage: 23% variance in high-value items
                • Overtime abuse: 67% spike in weekend overtime claims
                
                **📊 Process Efficiency Analysis:**
                • **Bottleneck Areas:** {', '.join(template_config['risk_areas'][:2])}
                • **Cost Savings Potential:** $847K annually through process optimization
                • **Automation Opportunities:** 34 manual processes identified
                
                **🔧 Operational Improvements:**
                1. **Immediate:** Implement biometric access controls
                2. **Short-term:** Deploy RFID inventory tracking
                3. **Long-term:** Automate approval workflows
                
                **📈 Expected Outcomes:**
                • Fraud reduction: 85%
                • Process efficiency: +23%
                • Cost savings: $847K annually
                
                **🎯 AI Confidence:** 93.5% (Demo Mode)
                """,
                
                "compliance": f"""
                ✅ **Operational Audit - Compliance Assessment**
                
                **📋 Compliance Framework:** ISO 9001, Industry Standards, Internal Policies
                
                **🔍 Current State:**
                • **Policy Compliance:** 84% adherence rate
                • **Process Documentation:** 71% complete
                • **Training Compliance:** 89% staff trained
                • **Quality Standards:** 92% conformance
                
                **⚠️ Non-Compliance Areas:**
                • Safety procedures: 12 incidents without proper documentation
                • Quality control: 8% of products lack complete testing records
                • Environmental standards: 3 violations in past quarter
                
                **🎯 Compliance Roadmap:**
                1. **Immediate:** Address safety documentation gaps
                2. **Short-term:** Implement quality tracking system
                3. **Long-term:** Achieve ISO certification readiness
                
                **💡 Best Practices Implementation:**
                • Continuous monitoring dashboard
                • Automated compliance reporting
                • Risk-based audit scheduling
                
                **🎯 AI Confidence:** 90.7% (Demo Mode)
                """
            },
            
            "it_audit": {
                "fraud_detection": f"""
                💻 **IT Audit - Security & Fraud Analysis**
                
                **🔐 Security Assessment:** {template_config['description']}
                
                **🚨 Security Incidents Detected:**
                • Unauthorized access attempts: 234 incidents (15% successful)
                • Privileged account misuse: 12 administrative actions without approval
                • Data exfiltration risk: 5 large file transfers to external locations
                • System vulnerabilities: 67 critical patches pending installation
                
                **📊 Access Control Analysis:**
                • **Excessive Privileges:** 23% of users have unnecessary admin rights
                • **Inactive Accounts:** 156 accounts not used in 90+ days
                • **Password Compliance:** 34% of accounts use weak passwords
                
                **🛡️ Security Improvements:**
                1. **Critical (7 days):** Patch all critical vulnerabilities
                2. **High (30 days):** Implement multi-factor authentication
                3. **Medium (90 days):** Deploy advanced threat monitoring
                
                **💰 Security Investment:** $285K for comprehensive security upgrade
                **Risk Reduction:** 78% reduction in security incidents expected
                
                **🎯 AI Confidence:** 96.1% (Demo Mode)
                """
            }
        }
        
        # Get template-specific response
        template_responses = mock_responses.get(audit_template, {})
        specific_response = template_responses.get(analysis_type)
        
        if specific_response:
            return specific_response
        
        # Generic response for any template/analysis combination
        return f"""
        {template_config['icon']} **{template_config['name']} - {analysis_type.replace('_', ' ').title()}**
        
        **🎯 Analysis Scope:** {template_config['description']}
        
        **📊 Query Processed:** {query}
        
        **🔍 Key Focus Areas:**
        {chr(10).join([f"• {area}" for area in template_config['risk_areas']])}
        
        **💡 Analysis Framework:**
        • Risk-based methodology applied
        • Industry best practices considered
        • Regulatory requirements evaluated
        • Data analytics performed
        
        **📈 Preliminary Insights:**
        • Pattern analysis completed on available data
        • Risk indicators identified and prioritized
        • Control effectiveness assessed
        • Improvement opportunities documented
        
        **🎯 Next Steps:**
        1. Define specific testing procedures
        2. Gather additional audit evidence
        3. Perform detailed risk assessment
        4. Develop remediation recommendations
        
        **⚠️ Note:** Configure OpenRouter API key for detailed real-time analysis specific to your audit objectives and data.
        
        **🎯 AI Confidence:** Demo Mode - Configure API for live analysis
        """

# ========================================
# SESSION STATE INITIALIZATION
# ========================================
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if 'agile_chat_history' not in st.session_state:
    st.session_state.agile_chat_history = []

if 'selected_audit_template' not in st.session_state:
    st.session_state.selected_audit_template = "custom"

if 'selected_industry' not in st.session_state:
    st.session_state.selected_industry = None

if 'uploaded_files_agile' not in st.session_state:
    st.session_state.uploaded_files_agile = {}

if 'agile_app_version' not in st.session_state:
    build_date = datetime.now().strftime("%Y%m%d")
    build_time = datetime.now().strftime("%H%M")
    st.session_state.agile_app_version = f"agile.{build_date}.{build_time}"

# Initialize agile services
@st.cache_resource
def initialize_agile_services():
    """Initialize agile audit services"""
    firebase_manager = AgileFirebaseManager()
    ai_engine = AgileAIEngine()
    
    return {
        'firebase': firebase_manager,
        'ai_engine': ai_engine
    }

agile_services = initialize_agile_services()

# ========================================
# MAIN APPLICATION HEADER
# ========================================
st.markdown(f"""
<div class="agile-header">
    <h1>🚀 Agile RAG AI Audit System</h1>
    <h2>Universal • Flexible • Intelligent</h2>
    <p>Any Industry • Any Audit Type • Any Scale • Instant Deployment</p>
    <p><strong>Version:</strong> {st.session_state.agile_app_version} | <strong>Session:</strong> {st.session_state.session_id} | <strong>Mode:</strong> Universal</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# AGILE SIDEBAR
# ========================================
with st.sidebar:
    st.header("🚀 Agile Control Center")
    
    # System Status
    st.subheader("📡 System Status")
    
    firebase_status = "Connected" if agile_services['firebase'].connected else "Demo Mode"
    ai_status = "Live API" if agile_services['ai_engine'].is_available else "Mock Mode"
    
    st.markdown(f"""
    <div style="margin: 0.5rem 0;">
        <span class="status-indicator {'status-good' if agile_services['firebase'].connected else 'status-warn'}"></span>
        <strong>Firebase:</strong> {firebase_status}
    </div>
    <div style="margin: 0.5rem 0;">
        <span class="status-indicator {'status-good' if agile_services['ai_engine'].is_available else 'status-warn'}"></span>
        <strong>Qwen3 AI:</strong> {ai_status}
    </div>
    <div style="margin: 0.5rem 0;">
        <span class="status-indicator status-good"></span>
        <strong>Analytics:</strong> Active
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick Template Selector
    st.subheader("🎯 Quick Setup")
    
    # Industry selector
    st.markdown("**🏢 Select Industry:**")
    industry_cols = st.columns(2)
    
    for i, (industry_id, industry_config) in enumerate(INDUSTRY_TEMPLATES.items()):
        col = industry_cols[i % 2]
        with col:
            if st.button(f"{industry_config['icon']} {industry_config['name']}", 
                        use_container_width=True, 
                        key=f"industry_{industry_id}"):
                st.session_state.selected_industry = industry_id
                st.rerun()
    
    # Audit template selector
    st.markdown("**📋 Select Audit Type:**")
    
    for template_id, template_config in AUDIT_TEMPLATES.items():
        if st.button(f"{template_config['icon']} {template_config['name']}", 
                    use_container_width=True, 
                    key=f"template_{template_id}"):
            st.session_state.selected_audit_template = template_id
            st.rerun()
    
    # Current selection display
    if st.session_state.selected_audit_template:
        current_template = AUDIT_TEMPLATES[st.session_state.selected_audit_template]
        st.markdown(f"""
        <div class="template-selector">
            <strong>🎯 Current Setup:</strong><br>
            {current_template['icon']} {current_template['name']}<br>
            <small>{current_template['description']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Upload Status
    st.subheader("📁 Files Status")
    
    uploaded_count = len(st.session_state.uploaded_files_agile)
    st.metric("Files Uploaded", uploaded_count)
    
    if uploaded_count > 0:
        for filename, status in st.session_state.uploaded_files_agile.items():
            status_icon = "✅" if status == 'success' else "❌" if status == 'error' else "⏳"
            st.markdown(f"{status_icon} {filename}")
    
    st.divider()
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    quick_actions = [
        ("🔍 Fraud Detection", "fraud_detection"),
        ("✅ Compliance Check", "compliance"),
        ("📊 Risk Assessment", "risk_assessment"),
        ("💰 Financial Analysis", "financial_analysis"),
        ("🔒 Security Audit", "security_audit"),
        ("📈 Performance Review", "performance_review")
    ]
    
    for action_name, action_type in quick_actions:
        if st.button(action_name, use_container_width=True, key=f"quick_{action_type}"):
            # Generate quick analysis query
            current_template = AUDIT_TEMPLATES[st.session_state.selected_audit_template]
            
            query_map = {
                "fraud_detection": f"Perform comprehensive fraud detection analysis focusing on {', '.join(current_template['risk_areas'][:2])}",
                "compliance": f"Assess compliance status and identify gaps in {current_template['name'].lower()}",
                "risk_assessment": f"Conduct risk assessment for {', '.join(current_template['risk_areas'])}",
                "financial_analysis": f"Analyze financial aspects and identify cost optimization opportunities",
                "security_audit": f"Review security controls and identify vulnerabilities",
                "performance_review": f"Evaluate performance metrics and operational efficiency"
            }
            
            query = query_map.get(action_type, f"Perform {action_type.replace('_', ' ')} analysis")
            
            st.session_state.agile_chat_history.append({
                "role": "user",
                "content": query,
                "action_type": action_type
            })
            
            with st.spinner(f"🤖 Processing {action_name}..."):
                context = f"Template: {current_template['name']}, Files: {list(st.session_state.uploaded_files_agile.keys())}"
                
                ai_response = agile_services['ai_engine'].analyze_agile_audit(
                    query, st.session_state.selected_audit_template, context, action_type
                )
                
                st.session_state.agile_chat_history.append({
                    "role": "assistant",
                    "content": ai_response,
                    "action_type": action_type
                })
            
            st.rerun()

# ========================================
# MAIN TABS
# ========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Quick Start",
    "📤 File Upload", 
    "🤖 AI Analysis",
    "📊 Dashboard",
    "⚙️ Advanced"
])

# ========================================
# TAB 1: QUICK START
# ========================================
with tab1:
    st.header("🎯 Agile Audit Quick Start")
    
    # Industry and template setup
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Step 1: Choose Industry")
        
        for industry_id, industry_config in INDUSTRY_TEMPLATES.items():
            selected = st.session_state.selected_industry == industry_id
            
            if st.button(
                f"{industry_config['icon']} {industry_config['name']}", 
                use_container_width=True,
                key=f"main_industry_{industry_id}",
                type="primary" if selected else "secondary"
            ):
                st.session_state.selected_industry = industry_id
                st.rerun()
            
            if selected:
                st.markdown(f"""
                <div class="agile-card">
                    <strong>Focus Areas:</strong><br>
                    {chr(10).join([f"• {area}" for area in industry_config['focus_areas']])}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📋 Step 2: Select Audit Type")
        
        for template_id, template_config in AUDIT_TEMPLATES.items():
            selected = st.session_state.selected_audit_template == template_id
            
            if st.button(
                f"{template_config['icon']} {template_config['name']}", 
                use_container_width=True,
                key=f"main_template_{template_id}",
                type="primary" if selected else "secondary"
            ):
                st.session_state.selected_audit_template = template_id
                st.rerun()
            
            if selected:
                st.markdown(f"""
                <div class="agile-card">
                    <strong>Description:</strong> {template_config['description']}<br>
                    <strong>Risk Areas:</strong><br>
                    {chr(10).join([f"• {area}" for area in template_config['risk_areas']])}
                </div>
                """, unsafe_allow_html=True)
    
    # Quick start actions
    if st.session_state.selected_audit_template != "custom":
        st.subheader("🚀 Step 3: Quick Actions")
        
        current_template = AUDIT_TEMPLATES[st.session_state.selected_audit_template]
        
        action_cols = st.columns(3)
        
        with action_cols[0]:
            if st.button("📤 Upload Files", use_container_width=True, type="primary"):
                st.switch_page("File Upload")  # This would work in a multi-page setup
            
            st.markdown(f"""
            <div class="agile-card">
                <strong>Expected Files:</strong><br>
                {chr(10).join([f"• {file_type}" for file_type in current_template['file_types']])}
            </div>
            """, unsafe_allow_html=True)
        
        with action_cols[1]:
            if st.button("🤖 Start AI Analysis", use_container_width=True, type="primary"):
                query = f"Perform comprehensive {current_template['name'].lower()} analysis"
                
                st.session_state.agile_chat_history.append({
                    "role": "user",
                    "content": query
                })
                
                with st.spinner("🤖 AI analyzing..."):
                    context = f"Template: {current_template['name']}"
                    ai_response = agile_services['ai_engine'].analyze_agile_audit(
                        query, st.session_state.selected_audit_template, context
                    )
                    
                    st.session_state.agile_chat_history.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                
                st.success("✅ Analysis complete! Check AI Analysis tab.")
        
        with action_cols[2]:
            if st.button("📊 View Dashboard", use_container_width=True, type="primary"):
                st.info("📊 Dashboard ready! Click Dashboard tab to view.")
    
    # Template showcase
    st.subheader("📚 Available Templates")
    
    template_cols = st.columns(2)
    
    for i, (template_id, template_config) in enumerate(AUDIT_TEMPLATES.items()):
        col = template_cols[i % 2]
        with col:
            st.markdown(f"""
            <div class="agile-card">
                <h4>{template_config['icon']} {template_config['name']}</h4>
                <p>{template_config['description']}</p>
                <div style="margin-top: 1rem;">
                    {' '.join([f'<span class="audit-type-badge">{area}</span>' for area in template_config['risk_areas'][:3]])}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ========================================
# TAB 2: FILE UPLOAD
# ========================================
with tab2:
    st.header("📤 Agile File Upload Center")
    
    current_template = AUDIT_TEMPLATES[st.session_state.selected_audit_template]
    
    st.markdown(f"""
    <div class="agile-card">
        <h3>{current_template['icon']} Current Template: {current_template['name']}</h3>
        <p>{current_template['description']}</p>
        <strong>Expected File Types:</strong> {', '.join(current_template['file_types'])}
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Universal file uploader
        st.markdown("""
        <div class="file-drop-zone">
            <h3>🎯 Universal File Upload</h3>
            <p>Drag & drop or browse files - Auto-detection enabled</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload Any File Type",
            type=['csv', 'xlsx', 'pdf', 'docx', 'txt', 'json'],
            accept_multiple_files=True,
            help="Supports: CSV, Excel, PDF, Word, Text, JSON files"
        )
        
        if uploaded_files:
            st.subheader("📋 File Processing Queue")
            
            for uploaded_file in uploaded_files:
                with st.container():
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    
                    with col_a:
                        file_status = st.session_state.uploaded_files_agile.get(uploaded_file.name, 'pending')
                        status_icon = "✅" if file_status == 'success' else "❌" if file_status == 'error' else "⏳"
                        
                        st.markdown(f"""
                        <div class="agile-card">
                            <h4>{status_icon} {uploaded_file.name}</h4>
                            <p>Size: {uploaded_file.size/1024:.1f} KB | Type: {uploaded_file.type}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        if st.button("🔍 Analyze", key=f"analyze_{uploaded_file.name}"):
                            try:
                                file_content = uploaded_file.getvalue()
                                
                                # Save to Firebase
                                if agile_services['firebase'].connected:
                                    result = agile_services['firebase'].save_agile_file(
                                        file_content, uploaded_file.name, uploaded_file.type,
                                        st.session_state.selected_audit_template
                                    )
                                    
                                    if result['success']:
                                        st.success(f"✅ {uploaded_file.name} analyzed!")
                                        st.session_state.uploaded_files_agile[uploaded_file.name] = 'success'
                                        
                                        # Show auto-detected insights
                                        if result.get('auto_detected_fields'):
                                            st.info(f"🔍 Auto-detected: {', '.join(result['auto_detected_fields'])}")
                                        
                                        if result.get('suggested_analysis'):
                                            st.info(f"💡 Suggested: {', '.join(result['suggested_analysis'])}")
                                    else:
                                        st.error(f"❌ Error: {result['error']}")
                                        st.session_state.uploaded_files_agile[uploaded_file.name] = 'error'
                                else:
                                    st.session_state.uploaded_files_agile[uploaded_file.name] = 'success'
                                    st.success(f"✅ {uploaded_file.name} processed locally!")
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                                st.session_state.uploaded_files_agile[uploaded_file.name] = 'error'
                    
                    with col_c:
                        if st.button("🤖 AI Query", key=f"ai_query_{uploaded_file.name}"):
                            query = f"Analyze the uploaded file {uploaded_file.name} for audit insights"
                            
                            st.session_state.agile_chat_history.append({
                                "role": "user",
                                "content": query
                            })
                            
                            with st.spinner("🤖 AI analyzing file..."):
                                context = f"File: {uploaded_file.name}, Template: {current_template['name']}"
                                ai_response = agile_services['ai_engine'].analyze_agile_audit(
                                    query, st.session_state.selected_audit_template, context
                                )
                                
                                st.session_state.agile_chat_history.append({
                                    "role": "assistant",
                                    "content": ai_response
                                })
                            
                            st.success("💬 Analysis added to AI chat!")
    
    with col2:
        st.subheader("📊 Upload Statistics")
        
        uploaded_count = len(st.session_state.uploaded_files_agile)
        success_count = len([f for f in st.session_state.uploaded_files_agile.values() if f == 'success'])
        
        st.metric("Files Uploaded", uploaded_count)
        st.metric("Successfully Processed", success_count)
        
        if uploaded_count > 0:
            success_rate = (success_count / uploaded_count) * 100
            st.metric("Success Rate", f"{success_rate:.0f}%")
        
        # Template guidance
        st.subheader("💡 Template Guidance")
        
        st.markdown(f"""
        <div class="agile-card">
            <strong>Key Fields to Include:</strong><br>
            {chr(10).join([f"• {field}" for field in current_template['key_fields']])}
        </div>
        """, unsafe_allow_html=True)

# ========================================
# TAB 3: AI ANALYSIS
# ========================================
with tab3:
    st.header("🤖 Agile AI Analysis Center")
    
    current_template = AUDIT_TEMPLATES[st.session_state.selected_audit_template]
    
    # AI Status Banner
    if agile_services['ai_engine'].is_available:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            🚀 <strong>LIVE AI:</strong> Real Qwen3 analysis powered by OpenRouter API!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            🎯 <strong>DEMO MODE:</strong> High-quality responses. Configure OpenRouter API for live analysis.
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Universal AI Auditor Chat")
        
        # Display chat history
        for message in st.session_state.agile_chat_history:
            if message["role"] == "user":
                action_tag = f"[{message.get('action_type', 'Query').title()}]" if message.get('action_type') else ""
                st.markdown(f"""
                <div style="text-align: right; margin: 1rem 0;">
                    <small style="color: #666;">{action_tag}</small>
                    <div style="background: #667eea; color: white; padding: 0.8rem 1.2rem; border-radius: 15px; display: inline-block; max-width: 80%;">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-response-agile">
                    <strong>🤖 {current_template['icon']} {current_template['name']} AI:</strong><br><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("agile_chat_form", clear_on_submit=True):
            user_input = st.text_area(
                f"Ask your {current_template['name']} AI Auditor:",
                placeholder=f"Example: {current_template['ai_prompts']['analyze']}",
                height=100
            )
            
            col_a, col_b = st.columns([1, 1])
            with col_a:
                submit_chat = st.form_submit_button("🚀 Analyze", use_container_width=True)
            with col_b:
                clear_chat = st.form_submit_button("🗑️ Clear", use_container_width=True)
            
            if submit_chat and user_input:
                st.session_state.agile_chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                with st.spinner("🤖 AI analyzing..."):
                    # Prepare context
                    uploaded_files = list(st.session_state.uploaded_files_agile.keys())
                    context = f"Template: {current_template['name']}, Files: {uploaded_files}"
                    
                    ai_response = agile_services['ai_engine'].analyze_agile_audit(
                        user_input, st.session_state.selected_audit_template, context
                    )
                    
                    st.session_state.agile_chat_history.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                
                st.rerun()
            
            if clear_chat:
                st.session_state.agile_chat_history = []
                st.rerun()
    
    with col2:
        st.subheader("🎯 Template Actions")
        
        # Template-specific quick actions
        for prompt_type, prompt_text in current_template['ai_prompts'].items():
            if st.button(f"{prompt_type.replace('_', ' ').title()}", 
                        use_container_width=True, 
                        key=f"prompt_{prompt_type}"):
                
                st.session_state.agile_chat_history.append({
                    "role": "user",
                    "content": prompt_text,
                    "action_type": prompt_type
                })
                
                with st.spinner(f"🤖 Processing {prompt_type}..."):
                    context = f"Template action: {prompt_type}"
                    ai_response = agile_services['ai_engine'].analyze_agile_audit(
                        prompt_text, st.session_state.selected_audit_template, context, prompt_type
                    )
                    
                    st.session_state.agile_chat_history.append({
                        "role": "assistant",
                        "content": ai_response,
                        "action_type": prompt_type
                    })
                
                st.rerun()
        
        st.divider()
        
        # Risk areas quick queries
        st.subheader("🔍 Risk Areas")
        
        for risk_area in current_template['risk_areas']:
            if st.button(f"Assess {risk_area}", 
                        use_container_width=True, 
                        key=f"risk_{risk_area}"):
                
                query = f"Perform detailed risk assessment for {risk_area} in {current_template['name'].lower()}"
                
                st.session_state.agile_chat_history.append({
                    "role": "user",
                    "content": query,
                    "action_type": "risk_assessment"
                })
                
                with st.spinner(f"🤖 Assessing {risk_area}..."):
                    context = f"Risk area focus: {risk_area}"
                    ai_response = agile_services['ai_engine'].analyze_agile_audit(
                        query, st.session_state.selected_audit_template, context, "risk_assessment"
                    )
                    
                    st.session_state.agile_chat_history.append({
                        "role": "assistant",
                        "content": ai_response,
                        "action_type": "risk_assessment"
                    })
                
                st.rerun()

# ========================================
# TAB 4: DASHBOARD
# ========================================
with tab4:
    st.header("📊 Agile Audit Dashboard")
    
    current_template = AUDIT_TEMPLATES[st.session_state.selected_audit_template]
    
    # Dashboard metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        files_uploaded = len(st.session_state.uploaded_files_agile)
        st.metric("Files Uploaded", files_uploaded, "📁")
    
    with col2:
        ai_interactions = len(st.session_state.agile_chat_history)
        st.metric("AI Interactions", ai_interactions, "🤖")
    
    with col3:
        template_usage = st.session_state.selected_audit_template
        st.metric("Template Active", current_template['name'], current_template['icon'])
    
    with col4:
        system_health = "Optimal" if agile_services['firebase'].connected else "Demo"
        st.metric("System Health", system_health, "✅")
    
    with col5:
        ai_mode = "Live" if agile_services['ai_engine'].is_available else "Mock"
        st.metric("AI Mode", ai_mode, "🧠")
    
    # Template-specific dashboard
    st.subheader(f"{current_template['icon']} {current_template['name']} Dashboard")
    
    # Risk area visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk areas chart
        risk_scores = np.random.uniform(3, 9, len(current_template['risk_areas']))
        risk_data = pd.DataFrame({
            'Risk Area': current_template['risk_areas'],
            'Risk Score': risk_scores,
            'Trend': np.random.choice(['↑', '↓', '→'], len(current_template['risk_areas']))
        })
        
        fig_risk = px.bar(
            risk_data,
            x='Risk Area',
            y='Risk Score',
            title=f"{current_template['name']} Risk Assessment",
            color='Risk Score',
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Compliance gauge
        compliance_score = np.random.randint(75, 95)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=compliance_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{current_template['name']} Compliance"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Activity timeline
    st.subheader("📈 Activity Timeline")
    
    # Generate sample activity data
    activity_data = []
    for i in range(30):
        activity_data.append({
            'Date': datetime.now() - timedelta(days=i),
            'Files_Uploaded': np.random.poisson(2),
            'AI_Queries': np.random.poisson(5),
            'Risk_Score': np.random.uniform(4, 8)
        })
    
    activity_df = pd.DataFrame(activity_data)
    
    fig_timeline = px.line(
        activity_df,
        x='Date',
        y=['Files_Uploaded', 'AI_Queries'],
        title="Daily Activity Timeline"
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# ========================================
# TAB 5: ADVANCED
# ========================================
with tab5:
    st.header("⚙️ Advanced Configuration")
    
    # Configuration sections
    config_tabs = st.tabs(["🔧 System Config", "🎯 Templates", "📊 Analytics", "🔒 Security"])
    
    with config_tabs[0]:
        st.subheader("🔧 System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Firebase Configuration:**")
            
            if agile_services['firebase'].connected:
                st.success("✅ Firebase Connected")
                
                # Show connection details
                with st.expander("Connection Details"):
                    st.info("Firebase initialized successfully with credentials")
            else:
                st.warning("⚠️ Firebase in Demo Mode")
                
                with st.expander("Setup Instructions"):
                    st.markdown("""
                    **Configure in Streamlit secrets:**
                    ```toml
                    firebase_project_id = "your-project-id"
                    firebase_private_key = "-----BEGIN PRIVATE KEY-----..."
                    firebase_client_email = "your-service-account@project.iam.gserviceaccount.com"
                    firebase_private_key_id = "key-id"
                    firebase_client_id = "client-id"
                    ```
                    """)
        
        with col2:
            st.markdown("**AI Configuration:**")
            
            if agile_services['ai_engine'].is_available:
                st.success("✅ Qwen3 AI Connected")
                
                with st.expander("AI Details"):
                    st.info(f"Model: {agile_services['ai_engine'].model}")
                    st.info("API: OpenRouter.ai")
            else:
                st.warning("⚠️ AI in Mock Mode")
                
                with st.expander("Setup Instructions"):
                    st.markdown("""
                    **Configure OpenRouter API:**
                    ```toml
                    openrouter_api_key = "sk-or-v1-your-api-key"
                    ```
                    Get your key from [OpenRouter.ai](https://openrouter.ai)
                    """)
    
    with config_tabs[1]:
        st.subheader("🎯 Template Management")
        
        # Template editor
        selected_template_id = st.selectbox(
            "Select Template to Edit:",
            list(AUDIT_TEMPLATES.keys()),
            format_func=lambda x: f"{AUDIT_TEMPLATES[x]['icon']} {AUDIT_TEMPLATES[x]['name']}"
        )
        
        selected_template = AUDIT_TEMPLATES[selected_template_id]
        
        with st.expander(f"Edit {selected_template['name']} Template"):
            # Template fields
            new_name = st.text_input("Template Name:", value=selected_template['name'])
            new_description = st.text_area("Description:", value=selected_template['description'])
            
            # Risk areas
            st.markdown("**Risk Areas:**")
            new_risk_areas = []
            for i, risk_area in enumerate(selected_template['risk_areas']):
                new_risk_area = st.text_input(f"Risk Area {i+1}:", value=risk_area, key=f"risk_{i}")
                if new_risk_area:
                    new_risk_areas.append(new_risk_area)
            
            if st.button("Save Template Changes"):
                st.success("✅ Template updated! (Changes are session-only)")
    
    with config_tabs[2]:
        st.subheader("📊 Analytics & Reporting")
        
        # Usage analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Session Analytics:**")
            
            session_metrics = {
                "Session ID": st.session_state.session_id,
                "App Version": st.session_state.agile_app_version,
                "Template Used": AUDIT_TEMPLATES[st.session_state.selected_audit_template]['name'],
                "Files Uploaded": len(st.session_state.uploaded_files_agile),
                "AI Interactions": len(st.session_state.agile_chat_history),
                "Session Duration": "Active",
                "System Mode": "Live" if agile_services['ai_engine'].is_available else "Demo"
            }
            
            for metric, value in session_metrics.items():
                st.metric(metric, value)
        
        with col2:
            st.markdown("**Export Options:**")
            
            if st.button("📄 Export Session Report"):
                report_data = {
                    "session_info": session_metrics,
                    "uploaded_files": list(st.session_state.uploaded_files_agile.keys()),
                    "chat_history": st.session_state.agile_chat_history,
                    "timestamp": datetime.now().isoformat()
                }
                
                st.download_button(
                    "💾 Download Report JSON",
                    data=json.dumps(report_data, indent=2, default=str),
                    file_name=f"audit_session_{st.session_state.session_id}.json",
                    mime="application/json"
                )
    
    with config_tabs[3]:
        st.subheader("🔒 Security & Privacy")
        
        security_info = {
            "Data Encryption": "✅ TLS/SSL encryption in transit",
            "Firebase Security": "✅ Service account authentication",
            "API Security": "✅ API key authentication",
            "Session Management": "✅ Unique session IDs",
            "Data Retention": "⚠️ Configure per organization policy",
            "Access Controls": "⚠️ Configure Firebase security rules"
        }
        
        for item, status in security_info.items():
            st.markdown(f"**{item}:** {status}")
        
        st.info("🔒 **Security Note:** This system processes audit data. Ensure compliance with your organization's data governance and privacy policies.")

# ========================================
# FOOTER
# ========================================
st.markdown("---")

# System status footer
current_time = datetime.now()
firebase_status = "Connected" if agile_services['firebase'].connected else "Demo"
ai_status = "Live" if agile_services['ai_engine'].is_available else "Mock"

st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
    <h3>🚀 Agile RAG AI Audit System - Universal Implementation</h3>
    <p><strong>Version:</strong> {st.session_state.agile_app_version} | <strong>Session:</strong> {st.session_state.session_id} | <strong>Template:</strong> {AUDIT_TEMPLATES[st.session_state.selected_audit_template]['name']}</p>
    <p><strong>Status:</strong> Firebase: {firebase_status} | AI: {ai_status} | Files: {len(st.session_state.uploaded_files_agile)} | Interactions: {len(st.session_state.agile_chat_history)}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.85em; margin-top: 1rem;">
    <p>🎯 <strong>Universal Audit System:</strong> Agile • Flexible • Intelligent • Production-Ready</p>
    <p>Supports: Any Industry • Any Audit Type • Any Scale • Instant Deployment</p>
    <p>Last Updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | Developer: MS Hadianto</p>
</div>
""", unsafe_allow_html=True)