#!/usr/bin/env python
"""Main Entry Point for Customer Churn Analytics Project"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description='Customer Segmentation & Churn Pattern Analytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --generate-data          # Generate sample data
  python run.py --train                  # Train models
  python run.py --dashboard              # Launch dashboard
  python run.py --test                   # Run tests
  python run.py --full                   # Run complete pipeline
        """
    )
    
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate sample data for testing')
    parser.add_argument('--train', action='store_true',
                       help='Train models and generate analysis')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch Streamlit dashboard')
    parser.add_argument('--test', action='store_true',
                       help='Run test suite')
    parser.add_argument('--notebook', action='store_true',
                       help='Launch Jupyter notebook')
    parser.add_argument('--full', action='store_true',
                       help='Run complete pipeline (generate + train + dashboard)')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("\n" + "="*70)
    print("  CUSTOMER SEGMENTATION & CHURN PATTERN ANALYTICS")
    print("  European Banking - Data-Driven Retention Intelligence")
    print("="*70 + "\n")
    
    # Execute based on arguments
    if args.generate_data or args.full:
        print("\n[STEP 1] Generating sample data...")
        from scripts.generate_sample_data import main as gen_data
        gen_data()
    
    if args.train or args.full:
        print("\n[STEP 2] Training models and generating analysis...")
        from scripts.train_models import main as train
        train()
    
    if args.test:
        print("\n[RUNNING TESTS]")
        import pytest
        pytest.main(['tests/', '-v'])
    
    if args.notebook:
        print("\n[LAUNCHING JUPYTER NOTEBOOK]")
        import subprocess
        subprocess.run(['jupyter', 'notebook', 'notebooks/'])
    
    if args.dashboard or args.full:
        print("\n[STEP 3] Launching Streamlit dashboard...")
        print("Dashboard will open at: http://localhost:8501\n")
        import subprocess
        subprocess.run(['streamlit', 'run', 'app/main.py'])
    
    print("\n" + "="*70)
    print("  COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
