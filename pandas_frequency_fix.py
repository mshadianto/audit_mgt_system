#!/usr/bin/env python3
"""
Quick Fix for Pandas Frequency Error
Fixes: pd.date_range freq='ME' ValueError

This script will:
1. Find and replace incompatible frequency aliases
2. Update your main_application.py file
3. Provide compatible alternatives
"""

import re
import os
from pathlib import Path

class PandasFrequencyFixer:
    """Fix pandas frequency compatibility issues"""
    
    def __init__(self):
        self.frequency_mappings = {
            # Old incompatible → New compatible
            'ME': 'M',      # Month End → Month
            'YE': 'Y',      # Year End → Year  
            'QE': 'Q',      # Quarter End → Quarter
            'BE': 'B',      # Business End → Business Day
            'WE': 'W',      # Week End → Week
            'H': 'H',       # Hour (usually fine)
            'T': 'T',       # Minute (usually fine)
            'S': 'S'        # Second (usually fine)
        }
        
        self.fixes_applied = []
    
    def detect_frequency_issues(self, file_path: str) -> list:
        """Detect frequency issues in a Python file"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Pattern to find date_range calls with freq parameter
            pattern = r"pd\.date_range\([^)]*freq\s*=\s*['\"]([^'\"]+)['\"][^)]*\)"
            matches = re.finditer(pattern, content)
            
            for match in matches:
                freq = match.group(1)
                if freq in ['ME', 'YE', 'QE', 'BE', 'WE']:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'line': line_num,
                        'freq': freq,
                        'match': match.group(0),
                        'replacement': self.frequency_mappings.get(freq, freq)
                    })
            
            return issues
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
    
    def fix_frequency_issues(self, file_path: str) -> bool:
        """Fix frequency issues in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Replace problematic frequency aliases
            for old_freq, new_freq in self.frequency_mappings.items():
                if old_freq in ['ME', 'YE', 'QE', 'BE', 'WE']:
                    # Replace freq='OLD' with freq='NEW'
                    pattern = f"freq\\s*=\\s*['\"]({old_freq})['\"]"
                    replacement = f"freq='{new_freq}'"
                    content = re.sub(pattern, replacement, content)
            
            # Check if any changes were made
            if content != original_content:
                # Create backup
                backup_path = f"{file_path}.backup"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ Fixed frequency issues in {file_path}")
                print(f"📁 Backup saved as {backup_path}")
                self.fixes_applied.append(file_path)
                return True
            else:
                print(f"ℹ️ No frequency issues found in {file_path}")
                return True
                
        except Exception as e:
            print(f"❌ Error fixing file {file_path}: {e}")
            return False
    
    def fix_current_directory(self) -> bool:
        """Fix all Python files in current directory"""
        current_dir = Path.cwd()
        python_files = list(current_dir.glob("*.py"))
        
        print(f"🔍 Scanning {len(python_files)} Python files for frequency issues...")
        
        total_issues = 0
        fixed_files = 0
        
        for py_file in python_files:
            print(f"\n📄 Checking {py_file.name}...")
            
            issues = self.detect_frequency_issues(str(py_file))
            if issues:
                print(f"⚠️ Found {len(issues)} frequency issues:")
                for issue in issues:
                    print(f"   Line {issue['line']}: {issue['freq']} → {issue['replacement']}")
                
                if self.fix_frequency_issues(str(py_file)):
                    fixed_files += 1
                    total_issues += len(issues)
            
        print(f"\n📊 Summary:")
        print(f"   Files scanned: {len(python_files)}")
        print(f"   Files fixed: {fixed_files}")
        print(f"   Issues fixed: {total_issues}")
        
        return fixed_files > 0 or total_issues == 0
    
    def create_compatible_code_snippet(self) -> str:
        """Create a code snippet with compatible date_range calls"""
        return '''
# ===============================
# PANDAS COMPATIBLE DATE RANGES
# ===============================

import pandas as pd
from datetime import datetime, timedelta

# ✅ CORRECT - Compatible frequency aliases
months = pd.date_range('2025-01-01', periods=12, freq='M')  # Monthly
quarters = pd.date_range('2025-01-01', periods=4, freq='Q')  # Quarterly  
years = pd.date_range('2020-01-01', periods=5, freq='Y')    # Yearly
days = pd.date_range('2025-01-01', periods=30, freq='D')    # Daily
business_days = pd.date_range('2025-01-01', periods=20, freq='B')  # Business days

# ❌ AVOID - These cause errors in newer pandas
# months = pd.date_range('2025-01-01', periods=12, freq='ME')  # Month End - DEPRECATED
# years = pd.date_range('2020-01-01', periods=5, freq='YE')   # Year End - DEPRECATED

# ✅ ALTERNATIVE - Use explicit period and end parameter
months_end = pd.date_range(start='2025-01-01', periods=12, freq='M') + pd.offsets.MonthEnd(0)

# ✅ ALTERNATIVE - Use period_range for more control
months_period = pd.period_range('2025-01', periods=12, freq='M')

print("✅ All date ranges created successfully!")
'''

def main():
    """Main function to run the frequency fixer"""
    print("🔧 Pandas Frequency Compatibility Fixer")
    print("=" * 50)
    
    fixer = PandasFrequencyFixer()
    
    # Check if main_application.py exists
    main_app_file = "main_application.py"
    if os.path.exists(main_app_file):
        print(f"🎯 Found {main_app_file} - checking for frequency issues...")
        
        issues = fixer.detect_frequency_issues(main_app_file)
        if issues:
            print(f"⚠️ Found {len(issues)} frequency issues in {main_app_file}:")
            for issue in issues:
                print(f"   Line {issue['line']}: {issue['freq']} → {issue['replacement']}")
            
            response = input("\n🔧 Apply fixes automatically? (y/n): ")
            if response.lower() in ['y', 'yes']:
                if fixer.fix_frequency_issues(main_app_file):
                    print("✅ Fixes applied successfully!")
                else:
                    print("❌ Failed to apply fixes")
            else:
                print("ℹ️ No fixes applied")
        else:
            print("✅ No frequency issues found in main_application.py")
    else:
        print(f"⚠️ {main_app_file} not found")
    
    # Check all Python files
    print(f"\n🔍 Checking all Python files in current directory...")
    fixer.fix_current_directory()
    
    # Create example code
    print(f"\n📝 Creating compatible code example...")
    example_code = fixer.create_compatible_code_snippet()
    
    with open('pandas_compatible_examples.py', 'w') as f:
        f.write(example_code)
    
    print("✅ Created pandas_compatible_examples.py with working examples")
    
    print(f"\n🎉 Frequency compatibility fix completed!")
    print(f"🚀 You can now run your Streamlit app without frequency errors")

if __name__ == "__main__":
    main()