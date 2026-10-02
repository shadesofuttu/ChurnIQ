#!/usr/bin/env python
"""Verify Project Setup and Dependencies"""

import sys
from pathlib import Path
import importlib.util

def check_python_version():
    """Check Python version."""
    print("\n" + "="*70)
    print("CHECKING PYTHON VERSION")
    print("="*70)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("[OK] Python version is compatible (>= 3.8)")
        return True
    else:
        print("[ERROR] Python version must be 3.8 or higher")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\n" + "="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib'
    }
    
    missing = []
    installed = []
    
    for module, package in required_packages.items():
        spec = importlib.util.find_spec(module)
        if spec is not None:
            print(f"[OK] {package} is installed")
            installed.append(package)
        else:
            print(f"[ERROR] {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n[WARNING] Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("\n[OK] All required packages are installed")
        return True

def check_directory_structure():
    """Check if required directories exist."""
    print("\n" + "="*70)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*70)
    
    required_dirs = [
        'src/data',
        'src/features',
        'src/models',
        'src/analysis',
        'src/visualization',
        'src/utils',
        'app',
        'app/components',
        'app/pages',
        'app/utils',
        'config',
        'data/raw',
        'data/processed',
        'data/segments',
        'models',
        'figures/eda',
        'figures/model',
        'figures/segmentation',
        'notebooks',
        'scripts',
        'tests',
        'reports'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"[OK] {dir_path}")
        else:
            print(f"[ERROR] {dir_path} (missing)")
            all_exist = False
    
    if all_exist:
        print("\n[OK] All required directories exist")
    else:
        print("\n[WARNING] Some directories are missing")
    
    return all_exist

def check_key_files():
    """Check if key files exist."""
    print("\n" + "="*70)
    print("CHECKING KEY FILES")
    print("="*70)
    
    key_files = [
        'README.md',
        'QUICKSTART.md',
        'requirements.txt',
        'config/settings.py',
        'src/data/loader.py',
        'src/data/preprocessing.py',
        'src/features/engineering.py',
        'src/models/churn_model.py',
        'src/models/segmentation.py',
        'src/analysis/churn_analytics.py',
        'src/visualization/plotting.py',
        'app/main.py',
        'scripts/train_models.py',
        'scripts/generate_sample_data.py',
        'notebooks/01_comprehensive_churn_analysis.ipynb',
        'reports/research_paper.md',
        'reports/executive_summary.md',
        'run.py'
    ]
    
    all_exist = True
    for file_path in key_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"[OK] {file_path} ({size:.1f} KB)")
        else:
            print(f"[ERROR] {file_path} (missing)")
            all_exist = False
    
    if all_exist:
        print("\n[OK] All key files exist")
    else:
        print("\n[WARNING] Some key files are missing")
    
    return all_exist

def check_data():
    """Check if data files exist."""
    print("\n" + "="*70)
    print("CHECKING DATA FILES")
    print("="*70)
    
    raw_data = Path('data/raw')
    if raw_data.exists():
        csv_files = list(raw_data.glob('*.csv'))
        if csv_files:
            print(f"[OK] Found {len(csv_files)} data file(s) in data/raw/")
            for f in csv_files:
                size = f.stat().st_size / 1024
                print(f"  - {f.name} ({size:.1f} KB)")
            return True
        else:
            print("[WARNING] No CSV files found in data/raw/")
            print("\nGenerate sample data with: python scripts/generate_sample_data.py")
            return False
    else:
        print("[ERROR] data/raw/ directory not found")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. If dependencies are missing:")
    print("   pip install -r requirements.txt")
    print("\n2. Generate sample data (if needed):")
    print("   python scripts/generate_sample_data.py")
    print("\n3. Train models and generate analysis:")
    print("   python scripts/train_models.py")
    print("\n4. Launch the dashboard:")
    print("   streamlit run app/main.py")
    print("\n5. Or use the quick launcher:")
    print("   python run.py --full")
    print("\n6. Run tests:")
    print("   python run.py --test")
    print("\n" + "="*70)

def main():
    """Main verification function."""
    print("\n" + "="*70)
    print("CUSTOMER SEGMENTATION & CHURN ANALYTICS")
    print("PROJECT SETUP VERIFICATION")
    print("="*70)
    
    checks = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Directory Structure': check_directory_structure(),
        'Key Files': check_key_files(),
        'Data Files': check_data()
    }
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for check_name, result in checks.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{check_name}: {status}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("*** ALL CHECKS PASSED - PROJECT IS READY! ***")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("[WARNING] SOME CHECKS FAILED - SEE DETAILS ABOVE")
        print("="*70)
    
    print_next_steps()
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
