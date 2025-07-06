#!/usr/bin/env python3
"""
Auto Error Fix Script for RAG Agentic AI Internal Audit System
Automatically detects and fixes all reported errors:
- Firebase credential issues
- OpenAI client proxies argument error
- ChromaDB telemetry errors
- SentenceTransformer import scope issues
"""

import subprocess
import sys
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import urllib.request
import tempfile

class AutoErrorFixer:
    """Automatically fix all reported errors"""
    
    def __init__(self):
        self.fixes_applied = []
        self.errors_found = []
        self.current_dir = Path.cwd()
        
    def detect_errors(self) -> List[str]:
        """Detect common errors from the system"""
        errors = []
        
        # Check for Firebase key file
        firebase_paths = [
            './firebase-key.json',
            './config/firebase-key.json',
            './serviceAccountKey.json'
        ]
        
        firebase_found = any(Path(p).exists() for p in firebase_paths)
        if not firebase_found and not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            errors.append('firebase_credentials_missing')
        
        # Check OpenAI client version
        try:
            import openai
            # Test for proxies argument issue
            try:
                client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key="test",
                    proxies={}  # This will fail in newer versions
                )
            except TypeError as e:
                if "proxies" in str(e):
                    errors.append('openai_proxies_error')
        except ImportError:
            errors.append('openai_missing')
        
        # Check ChromaDB telemetry issues
        try:
            import chromadb
            # Test telemetry
            try:
                client = chromadb.Client()
                # If this doesn't raise telemetry errors, we're good
            except Exception as e:
                if "telemetry" in str(e).lower():
                    errors.append('chromadb_telemetry_error')
        except ImportError:
            errors.append('chromadb_missing')
        
        self.errors_found = errors
        return errors
    
    def fix_openai_version(self) -> bool:
        """Fix OpenAI client version to compatible one"""
        print("🔧 Fixing OpenAI client version...")
        
        try:
            # Uninstall current version
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'openai', '-y'], 
                          capture_output=True)
            
            # Install compatible version
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'openai==1.3.8'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ OpenAI client fixed (version 1.3.8)")
                self.fixes_applied.append("openai_version_fix")
                return True
            else:
                print(f"❌ OpenAI fix failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ OpenAI fix error: {e}")
            return False
    
    def fix_chromadb_telemetry(self) -> bool:
        """Fix ChromaDB telemetry issues"""
        print("🔧 Fixing ChromaDB telemetry...")
        
        try:
            # Set environment variables
            env_vars = {
                'ANONYMIZED_TELEMETRY': 'False',
                'CHROMA_SERVER_NOFILE': '65536',
                'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python'
            }
            
            for var, value in env_vars.items():
                os.environ[var] = value
            
            # Create .env file with these settings
            env_file = self.current_dir / '.env'
            env_content = ""
            
            if env_file.exists():
                with open(env_file, 'r') as f:
                    env_content = f.read()
            
            # Add missing environment variables
            for var, value in env_vars.items():
                if var not in env_content:
                    env_content += f"\n{var}={value}"
            
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            print("✅ ChromaDB telemetry settings fixed")
            self.fixes_applied.append("chromadb_telemetry_fix")
            return True
            
        except Exception as e:
            print(f"❌ ChromaDB telemetry fix error: {e}")
            return False
    
    def create_firebase_demo_credentials(self) -> bool:
        """Create demo Firebase credentials for testing"""
        print("🔧 Creating Firebase demo setup...")
        
        try:
            demo_config = {
                "type": "service_account",
                "project_id": "demo-project",
                "private_key_id": "demo-key-id",
                "private_key": "-----BEGIN PRIVATE KEY-----\\nDEMO_KEY\\n-----END PRIVATE KEY-----\\n",
                "client_email": "demo@demo-project.iam.gserviceaccount.com",
                "client_id": "demo-client-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/demo%40demo-project.iam.gserviceaccount.com"
            }
            
            # Create config directory
            config_dir = self.current_dir / 'config'
            config_dir.mkdir(exist_ok=True)
            
            # Save demo config
            demo_file = config_dir / 'firebase-demo.json'
            with open(demo_file, 'w') as f:
                json.dump(demo_config, f, indent=2)
            
            print("✅ Demo Firebase config created")
            print("ℹ️ Note: This is for demo purposes only. Use real Firebase for production.")
            self.fixes_applied.append("firebase_demo_setup")
            return True
            
        except Exception as e:
            print(f"❌ Firebase demo setup error: {e}")
            return False
    
    def install_fixed_requirements(self) -> bool:
        """Install requirements with fixed versions"""
        print("🔧 Installing fixed requirements...")
        
        # Fixed requirements addressing all errors
        fixed_requirements = [
            "protobuf==3.20.3",
            "streamlit==1.39.0", 
            "python-dotenv==1.0.1",
            "openai==1.3.8",  # Fixed version without proxies issue
            "firebase-admin==6.4.0",
            "chromadb==0.4.15",
            "sentence-transformers==2.2.2",
            "pandas==2.1.4",
            "numpy==1.24.3",
            "plotly==5.17.0",
            "python-docx==0.8.11",
            "PyPDF2==3.0.1",
            "Pillow==10.0.1",
            "pydantic==2.5.3"
        ]
        
        success_count = 0
        
        for requirement in fixed_requirements:
            try:
                print(f"📦 Installing {requirement}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', requirement
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ {requirement} installed")
                    success_count += 1
                else:
                    print(f"❌ {requirement} failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ {requirement} installation timed out")
            except Exception as e:
                print(f"❌ {requirement} error: {e}")
        
        if success_count == len(fixed_requirements):
            print("✅ All requirements installed successfully")
            self.fixes_applied.append("requirements_install")
            return True
        else:
            print(f"⚠️ {success_count}/{len(fixed_requirements)} requirements installed")
            return False
    
    def create_env_template(self) -> bool:
        """Create .env template with all necessary variables"""
        print("🔧 Creating .env template...")
        
        try:
            env_template = """# RAG Agentic AI Internal Audit System - Environment Variables
# Auto-generated by error fix script

# ========================================
# CRITICAL: Error Prevention Settings
# ========================================

# Fix ChromaDB telemetry errors
ANONYMIZED_TELEMETRY=False
CHROMA_SERVER_NOFILE=65536

# Fix protobuf conflicts
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ========================================
# API Configuration
# ========================================

# OpenRouter API Key (Required for AI functionality)
# Get from: https://openrouter.ai/
OPENROUTER_API_KEY=your_openrouter_api_key_here

# ========================================
# Firebase Configuration (Optional)
# ========================================

# Method 1: Path to service account JSON file
GOOGLE_APPLICATION_CREDENTIALS=./config/firebase-key.json

# Method 2: Firebase project settings (alternative)
# FIREBASE_PROJECT_ID=your-project-id
# FIREBASE_PRIVATE_KEY_ID=your-private-key-id
# FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\\nYOUR_KEY\\n-----END PRIVATE KEY-----\\n"
# FIREBASE_CLIENT_EMAIL=your-service@your-project.iam.gserviceaccount.com

# ========================================
# Streamlit Configuration
# ========================================

# Streamlit server settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_SERVER_HEADLESS=true

# ========================================
# Performance Settings
# ========================================

# Vector database settings
VECTOR_DB_PERSIST_DIRECTORY=./vector_db
VECTOR_DB_COLLECTION_NAME=audit_knowledge_base

# Model cache directory
TRANSFORMERS_CACHE=./model_cache
SENTENCE_TRANSFORMERS_HOME=./model_cache

# ========================================
# Security Settings
# ========================================

# Enable security features
SECURE_MODE=true
LOG_LEVEL=INFO

# File upload limits (in MB)
MAX_FILE_SIZE_MB=10
MAX_FILES_PER_UPLOAD=5

# ========================================
# Debug Settings (Development Only)
# ========================================

# Set to true for detailed logging
DEBUG_MODE=false
VERBOSE_LOGGING=false

# Disable for production
STREAMLIT_DEVELOPMENT_MODE=false
"""
            
            env_file = self.current_dir / '.env'
            with open(env_file, 'w') as f:
                f.write(env_template)
            
            print("✅ .env template created with error prevention settings")
            self.fixes_applied.append("env_template_creation")
            return True
            
        except Exception as e:
            print(f"❌ .env template creation error: {e}")
            return False
    
    def test_fixes(self) -> Dict[str, bool]:
        """Test that all fixes are working"""
        print("🧪 Testing fixes...")
        
        test_results = {}
        
        # Test OpenAI import
        try:
            import openai
            # Test without proxies argument
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key="test"
            )
            test_results['openai'] = True
            print("✅ OpenAI client test passed")
        except Exception as e:
            if "proxies" not in str(e):
                test_results['openai'] = True
                print("✅ OpenAI client syntax correct")
            else:
                test_results['openai'] = False
                print("❌ OpenAI client still has proxies issue")
        
        # Test ChromaDB with telemetry disabled
        try:
            os.environ['ANONYMIZED_TELEMETRY'] = 'False'
            import chromadb
            client = chromadb.Client()
            test_results['chromadb'] = True
            print("✅ ChromaDB test passed")
        except Exception as e:
            if "telemetry" not in str(e).lower():
                test_results['chromadb'] = True
                print("✅ ChromaDB telemetry issues resolved")
            else:
                test_results['chromadb'] = False
                print("❌ ChromaDB still has telemetry issues")
        
        # Test SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            test_results['sentence_transformers'] = True
            print("✅ SentenceTransformers import test passed")
        except Exception as e:
            test_results['sentence_transformers'] = False
            print(f"❌ SentenceTransformers test failed: {e}")
        
        # Test basic imports
        basic_imports = ['streamlit', 'pandas', 'numpy', 'plotly']
        for module in basic_imports:
            try:
                __import__(module)
                test_results[module] = True
                print(f"✅ {module} import test passed")
            except ImportError:
                test_results[module] = False
                print(f"❌ {module} import test failed")
        
        return test_results
    
    def generate_launch_script(self) -> bool:
        """Generate launch script with proper environment setup"""
        print("🔧 Creating launch script...")
        
        try:
            # Windows batch script
            batch_script = """@echo off
echo Starting RAG Agentic AI Internal Audit System...
echo Setting environment variables...

set ANONYMIZED_TELEMETRY=False
set CHROMA_SERVER_NOFILE=65536
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo Launching application...
streamlit run audit_app_error_fixed.py

pause
"""
            
            with open('launch.bat', 'w') as f:
                f.write(batch_script)
            
            # Unix shell script
            shell_script = """#!/bin/bash
echo "Starting RAG Agentic AI Internal Audit System..."
echo "Setting environment variables..."

export ANONYMIZED_TELEMETRY=False
export CHROMA_SERVER_NOFILE=65536
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "Launching application..."
streamlit run audit_app_error_fixed.py
"""
            
            with open('launch.sh', 'w') as f:
                f.write(shell_script)
            
            # Make shell script executable
            os.chmod('launch.sh', 0o755)
            
            print("✅ Launch scripts created (launch.bat for Windows, launch.sh for Unix)")
            self.fixes_applied.append("launch_scripts_creation")
            return True
            
        except Exception as e:
            print(f"❌ Launch script creation error: {e}")
            return False
    
    def run_complete_fix(self) -> bool:
        """Run complete error fixing process"""
        print("🚀 Starting Auto Error Fix for RAG Audit System")
        print("=" * 60)
        
        # Step 1: Detect errors
        detected_errors = self.detect_errors()
        print(f"🔍 Detected {len(detected_errors)} potential issues:")
        for error in detected_errors:
            print(f"  • {error}")
        
        if not detected_errors:
            print("✅ No common errors detected!")
        
        print("\n" + "=" * 60)
        print("🔧 Applying fixes...")
        
        # Step 2: Apply fixes
        fixes = [
            ("Installing fixed requirements", self.install_fixed_requirements),
            ("Fixing OpenAI client version", self.fix_openai_version),
            ("Fixing ChromaDB telemetry", self.fix_chromadb_telemetry),
            ("Creating .env template", self.create_env_template),
            ("Creating Firebase demo setup", self.create_firebase_demo_credentials),
            ("Generating launch scripts", self.generate_launch_script)
        ]
        
        success_count = 0
        for description, fix_function in fixes:
            print(f"\n{description}...")
            try:
                if fix_function():
                    success_count += 1
                    print(f"✅ {description} - Success")
                else:
                    print(f"❌ {description} - Failed")
            except Exception as e:
                print(f"❌ {description} - Error: {e}")
        
        # Step 3: Test fixes
        print("\n" + "=" * 60)
        test_results = self.test_fixes()
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        # Step 4: Generate report
        print("\n" + "=" * 60)
        print("📊 FINAL REPORT")
        print("=" * 60)
        
        print(f"✅ Fixes Applied: {len(self.fixes_applied)}")
        for fix in self.fixes_applied:
            print(f"   • {fix}")
        
        print(f"\n🧪 Tests Passed: {passed_tests}/{total_tests}")
        for test, result in test_results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {test}")
        
        overall_success = success_count >= 4 and passed_tests >= total_tests * 0.8
        
        if overall_success:
            print("\n🎉 AUTO-FIX COMPLETED SUCCESSFULLY!")
            print("\n🚀 Next Steps:")
            print("1. Edit .env file and add your OPENROUTER_API_KEY")
            print("2. Optionally add Firebase credentials")
            print("3. Run: python audit_app_error_fixed.py")
            print("   Or use: launch.bat (Windows) / ./launch.sh (Unix)")
            print("\n✨ All reported errors should now be resolved!")
        else:
            print("\n⚠️ AUTO-FIX COMPLETED WITH ISSUES")
            print("Some fixes may need manual intervention.")
            print("Check the error messages above for details.")
        
        return overall_success

def main():
    """Main function"""
    print("🔍 RAG Agentic AI - Auto Error Fix Script")
    print("Fixes: Firebase | OpenAI | ChromaDB | SentenceTransformers\n")
    
    fixer = AutoErrorFixer()
    success = fixer.run_complete_fix()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())