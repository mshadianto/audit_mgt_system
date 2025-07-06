#!/usr/bin/env python3
"""
RAG Agentic AI Internal Audit System - Diagnostic & Auto-Fix Script
Run this script to automatically detect and fix common installation issues
"""

import subprocess
import sys
import os
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class RAGSystemDiagnostic:
    """Comprehensive diagnostic and auto-fix for RAG Audit System"""
    
    def __init__(self):
        self.results = {
            'python_version': None,
            'pip_version': None,
            'dependencies': {},
            'environment': {},
            'issues': [],
            'fixes_applied': [],
            'status': 'unknown'
        }
        
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        version = sys.version_info
        self.results['python_version'] = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major == 3 and version.minor >= 11:
            print(f"✅ Python {self.results['python_version']} - Compatible")
            return True
        else:
            print(f"❌ Python {self.results['python_version']} - Requires Python 3.11+")
            self.results['issues'].append("Python version incompatible")
            return False
    
    def check_pip_version(self) -> bool:
        """Check pip version and upgrade if needed"""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.results['pip_version'] = result.stdout.strip()
                print(f"✅ Pip available: {self.results['pip_version']}")
                return True
            else:
                print("❌ Pip not available")
                return False
        except Exception as e:
            print(f"❌ Pip check failed: {e}")
            return False
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version"""
        try:
            print("🔄 Upgrading pip...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Pip upgraded successfully")
                self.results['fixes_applied'].append("pip_upgrade")
                return True
            else:
                print(f"❌ Pip upgrade failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Pip upgrade error: {e}")
            return False
    
    def check_dependency(self, package_name: str, import_name: str = None) -> Tuple[bool, Optional[str]]:
        """Check if a dependency is installed and importable"""
        import_name = import_name or package_name
        
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            self.results['dependencies'][package_name] = {
                'installed': True,
                'version': version,
                'importable': True
            }
            return True, version
        except ImportError:
            # Check if installed but not importable
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.results['dependencies'][package_name] = {
                        'installed': True,
                        'version': 'unknown',
                        'importable': False
                    }
                    return False, None
                else:
                    self.results['dependencies'][package_name] = {
                        'installed': False,
                        'version': None,
                        'importable': False
                    }
                    return False, None
            except Exception:
                self.results['dependencies'][package_name] = {
                    'installed': False,
                    'version': None,
                    'importable': False
                }
                return False, None
    
    def fix_protobuf_conflict(self) -> bool:
        """Fix protobuf version conflicts"""
        print("🔧 Fixing protobuf conflicts...")
        
        # Set environment variable
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        
        try:
            # Uninstall current protobuf
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'protobuf', '-y'], 
                          capture_output=True)
            
            # Install specific version
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'protobuf==3.20.3'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Protobuf fixed (version 3.20.3)")
                self.results['fixes_applied'].append("protobuf_fix")
                return True
            else:
                print(f"❌ Protobuf fix failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Protobuf fix error: {e}")
            return False
    
    def install_dependency(self, package_spec: str) -> bool:
        """Install a specific dependency"""
        try:
            print(f"📦 Installing {package_spec}...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package_spec], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package_spec} installed successfully")
                self.results['fixes_applied'].append(f"install_{package_spec}")
                return True
            else:
                print(f"❌ {package_spec} installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Installation error for {package_spec}: {e}")
            return False
    
    def check_all_dependencies(self) -> Dict[str, bool]:
        """Check all required dependencies"""
        dependencies = {
            'streamlit': 'streamlit',
            'firebase-admin': 'firebase_admin', 
            'openai': 'openai',
            'chromadb': 'chromadb',
            'sentence-transformers': 'sentence_transformers',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'plotly': 'plotly',
            'python-docx': 'docx',
            'PyPDF2': 'PyPDF2',
            'python-dotenv': 'dotenv',
            'Pillow': 'PIL',
            'pydantic': 'pydantic'
        }
        
        results = {}
        print("\n🔍 Checking dependencies...")
        
        for package, import_name in dependencies.items():
            available, version = self.check_dependency(package, import_name)
            if available:
                print(f"✅ {package}: {version}")
                results[package] = True
            else:
                print(f"❌ {package}: Not available")
                results[package] = False
                self.results['issues'].append(f"Missing dependency: {package}")
        
        return results
    
    def auto_fix_dependencies(self, missing_deps: List[str]) -> bool:
        """Automatically install missing dependencies in correct order"""
        if not missing_deps:
            return True
        
        print(f"\n🔧 Auto-fixing {len(missing_deps)} missing dependencies...")
        
        # Installation order matters for compatibility
        install_order = [
            'protobuf==3.20.3',
            'streamlit==1.39.0',
            'python-dotenv==1.0.1',
            'firebase-admin==6.5.0',
            'sentence-transformers==2.3.1',
            'chromadb==0.4.24',
            'openai==1.12.0',
            'pandas==2.2.2',
            'numpy==1.26.4',
            'plotly==5.24.1',
            'python-docx==1.1.2',
            'PyPDF2==3.0.1',
            'Pillow==10.2.0',
            'pydantic==2.6.1'
        ]
        
        # Filter to only install missing dependencies
        to_install = []
        for spec in install_order:
            package_name = spec.split('==')[0]
            # Handle different package names
            if package_name in missing_deps or \
               (package_name == 'python-docx' and 'python-docx' in missing_deps) or \
               (package_name == 'Pillow' and 'Pillow' in missing_deps):
                to_install.append(spec)
        
        success_count = 0
        for package_spec in to_install:
            if self.install_dependency(package_spec):
                success_count += 1
        
        print(f"\n📊 Installation Summary: {success_count}/{len(to_install)} successful")
        return success_count == len(to_install)
    
    def check_environment(self) -> Dict[str, str]:
        """Check environment variables and configuration"""
        env_vars = {
            'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY'),
            'GOOGLE_APPLICATION_CREDENTIALS': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': os.getenv('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION')
        }
        
        print("\n🌍 Checking environment...")
        for var, value in env_vars.items():
            if value:
                print(f"✅ {var}: Set")
                self.results['environment'][var] = 'set'
            else:
                print(f"⚠️ {var}: Not set")
                self.results['environment'][var] = 'not_set'
        
        return env_vars
    
    def test_imports(self) -> bool:
        """Test critical imports to verify functionality"""
        print("\n🧪 Testing critical imports...")
        
        critical_tests = [
            ('streamlit', 'import streamlit as st'),
            ('pandas', 'import pandas as pd; df = pd.DataFrame({"test": [1, 2, 3]})'),
            ('numpy', 'import numpy as np; arr = np.array([1, 2, 3])'),
            ('plotly', 'import plotly.express as px'),
            ('sentence_transformers', 'from sentence_transformers import SentenceTransformer'),
        ]
        
        success_count = 0
        for name, test_code in critical_tests:
            try:
                exec(test_code)
                print(f"✅ {name}: Import test passed")
                success_count += 1
            except Exception as e:
                print(f"❌ {name}: Import test failed - {e}")
        
        return success_count == len(critical_tests)
    
    def create_env_file(self) -> bool:
        """Create .env file template if it doesn't exist"""
        env_file = Path('.env')
        
        if not env_file.exists():
            env_template = """# RAG Agentic AI Internal Audit System Environment Variables

# OpenRouter API Key (Required for AI functionality)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Firebase Configuration (Optional - can upload JSON through UI)
GOOGLE_APPLICATION_CREDENTIALS=path/to/firebase-service-account.json

# Vector Database Settings
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Optional: Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
"""
            
            try:
                with open('.env', 'w') as f:
                    f.write(env_template)
                print("✅ Created .env template file")
                self.results['fixes_applied'].append("env_file_created")
                return True
            except Exception as e:
                print(f"❌ Failed to create .env file: {e}")
                return False
        else:
            print("ℹ️ .env file already exists")
            return True
    
    def generate_report(self) -> str:
        """Generate comprehensive diagnostic report"""
        report = f"""
# 🔍 RAG Audit System Diagnostic Report
Generated: {str(subprocess.run(['date'], capture_output=True, text=True).stdout.strip())}

## System Information
- Python Version: {self.results['python_version']}
- Pip Version: {self.results['pip_version']}

## Dependency Status
"""
        
        for dep, info in self.results['dependencies'].items():
            status = "✅" if info['importable'] else "❌"
            version = f" (v{info['version']})" if info['version'] != 'unknown' else ""
            report += f"- {status} {dep}{version}\n"
        
        report += f"""
## Environment Variables
"""
        for var, status in self.results['environment'].items():
            icon = "✅" if status == 'set' else "⚠️"
            report += f"- {icon} {var}: {status}\n"
        
        if self.results['issues']:
            report += f"""
## Issues Found ({len(self.results['issues'])})
"""
            for issue in self.results['issues']:
                report += f"- ❌ {issue}\n"
        
        if self.results['fixes_applied']:
            report += f"""
## Fixes Applied ({len(self.results['fixes_applied'])})
"""
            for fix in self.results['fixes_applied']:
                report += f"- ✅ {fix}\n"
        
        return report
    
    def run_full_diagnostic(self) -> bool:
        """Run complete diagnostic and auto-fix process"""
        print("🚀 Starting RAG Audit System Diagnostic...")
        print("=" * 60)
        
        # Step 1: Basic system checks
        python_ok = self.check_python_version()
        pip_ok = self.check_pip_version()
        
        if not python_ok:
            print("\n❌ Python version incompatible. Please upgrade to Python 3.11+")
            return False
        
        if pip_ok:
            self.upgrade_pip()
        
        # Step 2: Check dependencies
        dep_results = self.check_all_dependencies()
        missing_deps = [dep for dep, available in dep_results.items() if not available]
        
        # Step 3: Fix protobuf conflicts first
        protobuf_available, _ = self.check_dependency('protobuf')
        if not protobuf_available or 'chromadb' in missing_deps:
            self.fix_protobuf_conflict()
        
        # Step 4: Auto-fix missing dependencies
        if missing_deps:
            print(f"\n🔧 Found {len(missing_deps)} missing dependencies")
            auto_fix_success = self.auto_fix_dependencies(missing_deps)
            
            if auto_fix_success:
                print("✅ All dependencies installed successfully")
            else:
                print("⚠️ Some dependencies could not be installed automatically")
        else:
            print("\n✅ All dependencies are available")
        
        # Step 5: Environment setup
        self.check_environment()
        self.create_env_file()
        
        # Step 6: Test imports
        import_success = self.test_imports()
        
        # Step 7: Final status
        overall_success = len(missing_deps) == 0 and import_success
        self.results['status'] = 'success' if overall_success else 'partial'
        
        print("\n" + "=" * 60)
        if overall_success:
            print("🎉 Diagnostic completed successfully! System is ready.")
        else:
            print("⚠️ Diagnostic completed with issues. Check the report above.")
        
        # Generate report
        report = self.generate_report()
        try:
            with open('diagnostic_report.md', 'w') as f:
                f.write(report)
            print("📄 Diagnostic report saved to 'diagnostic_report.md'")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
        
        return overall_success

def main():
    """Main function to run diagnostic"""
    print("🔍 RAG Agentic AI Internal Audit System - Auto Diagnostic")
    print("This script will check and fix common installation issues\n")
    
    diagnostic = RAGSystemDiagnostic()
    success = diagnostic.run_full_diagnostic()
    
    if success:
        print("\n🚀 Next Steps:")
        print("1. Set your OPENROUTER_API_KEY in the .env file")
        print("2. Configure Firebase credentials (optional)")
        print("3. Run: streamlit run audit_app_improved.py")
        return 0
    else:
        print("\n🔧 Manual Steps Required:")
        print("1. Review the diagnostic report above")
        print("2. Install missing dependencies manually")
        print("3. Check Python version compatibility")
        print("4. Consult the installation guide")
        return 1

if __name__ == "__main__":
    sys.exit(main())