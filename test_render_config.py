"""
Test script to verify Render deployment configuration
Run this before deploying to Render to catch any issues early
"""

import os
import sys
from pathlib import Path
import json

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_render_config():
    """Check Render configuration files"""
    print("=" * 60)
    print("RENDER DEPLOYMENT CONFIGURATION CHECK")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check render.yaml
    print("📋 Checking render.yaml...")
    render_yaml_exists = check_file_exists("render.yaml", "render.yaml")
    if render_yaml_exists:
        with open("render.yaml", "r", encoding="utf-8") as f:
            content = f.read()
            # Check for required services
            if "pcos-api" in content and "pcos-ui" in content:
                print("  ✅ Both services (pcos-api and pcos-ui) found")
            else:
                print("  ❌ Missing services in render.yaml")
                all_checks_passed = False
            
            # Check for gunicorn command
            if "gunicorn" in content:
                print("  ✅ Gunicorn command found")
            else:
                print("  ❌ Gunicorn command not found")
                all_checks_passed = False
            
            # Check for streamlit command
            if "streamlit run" in content:
                print("  ✅ Streamlit command found")
            else:
                print("  ❌ Streamlit command not found")
                all_checks_passed = False
            
            # Check for PORT variable
            if "$PORT" in content:
                print("  ✅ PORT environment variable used")
            else:
                print("  ⚠️  PORT environment variable not found (may use hardcoded port)")
    else:
        print("  ❌ render.yaml not found!")
        all_checks_passed = False
    
    print()
    
    # Check requirements.txt
    print("📦 Checking requirements.txt...")
    req_exists = check_file_exists("requirements.txt", "requirements.txt")
    if req_exists:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            reqs = f.read()
            required_packages = ["flask", "gunicorn", "streamlit", "tensorflow"]
            for pkg in required_packages:
                if pkg.lower() in reqs.lower():
                    print(f"  ✅ {pkg} found in requirements")
                else:
                    print(f"  ❌ {pkg} NOT found in requirements")
                    all_checks_passed = False
    else:
        all_checks_passed = False
    
    print()
    
    # Check runtime.txt
    print("🐍 Checking runtime.txt...")
    runtime_exists = check_file_exists("runtime.txt", "runtime.txt")
    if runtime_exists:
        with open("runtime.txt", "r", encoding="utf-8") as f:
            runtime = f.read().strip()
            if "python" in runtime.lower():
                print(f"  ✅ Python version specified: {runtime}")
            else:
                print(f"  ⚠️  Unexpected runtime: {runtime}")
    else:
        print("  ⚠️  runtime.txt not found (Render will use default Python version)")
    
    print()
    
    # Check Procfile (optional, render.yaml takes precedence)
    print("📄 Checking Procfile (optional)...")
    procfile_exists = check_file_exists("Procfile", "Procfile")
    if procfile_exists:
        print("  ℹ️  Procfile exists (render.yaml takes precedence on Render)")
    
    print()
    
    # Check application files
    print("📁 Checking application files...")
    app_files = [
        ("src/api.py", "Flask API"),
        ("app.py", "Streamlit UI"),
    ]
    
    for filepath, description in app_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    print()
    
    # Check model files (optional - may not exist before first training)
    print("🤖 Checking model files...")
    model_files = [
        "models/pcos_model.h5",
        "models/pcos_model.keras",
        "models/pcos_model.weights.h5"
    ]
    model_found = False
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"  ✅ Found: {model_file}")
            model_found = True
    
    if not model_found:
        print("  ⚠️  No model files found - model will need to be trained first")
        print("     (This is OK if you haven't trained yet)")
    
    print()
    
    # Check API_BASE_URL in render.yaml
    print("🔗 Checking API_BASE_URL configuration...")
    if render_yaml_exists:
        with open("render.yaml", "r", encoding="utf-8") as f:
            content = f.read()
            if "API_BASE_URL" in content:
                if "pcos-api.onrender.com" in content or "$" in content:
                    print("  ✅ API_BASE_URL configured")
                    if "pcos-api.onrender.com" in content:
                        print("  ⚠️  NOTE: Update API_BASE_URL with actual Render URL after deployment")
                else:
                    print("  ⚠️  API_BASE_URL may need to be updated after deployment")
            else:
                print("  ⚠️  API_BASE_URL not found in render.yaml")
    
    print()
    
    # Check PORT handling in API
    print("🔌 Checking PORT environment variable handling...")
    if os.path.exists("src/api.py"):
        try:
            with open("src/api.py", "r", encoding="utf-8") as f:
                api_content = f.read()
                if "os.environ.get('PORT'" in api_content or "$PORT" in api_content:
                    print("  ✅ API handles PORT environment variable")
                else:
                    print("  ⚠️  API may not handle PORT correctly (check src/api.py line 700)")
        except Exception as e:
            print(f"  ⚠️  Could not read src/api.py: {e}")
    
    print()
    
    # Summary
    print("=" * 60)
    if all_checks_passed:
        print("✅ ALL CRITICAL CHECKS PASSED!")
        print()
        print("Next steps:")
        print("1. Push code to GitHub")
        print("2. Deploy API service on Render")
        print("3. Copy the API URL and update API_BASE_URL in render.yaml")
        print("4. Deploy UI service on Render")
        print("5. Test both services")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before deploying")
    print("=" * 60)

def test_local_api():
    """Test if API can be imported and initialized"""
    print()
    print("=" * 60)
    print("TESTING LOCAL API IMPORT")
    print("=" * 60)
    print()
    
    try:
        # Add project root to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Try importing the API
        print("Attempting to import src.api...")
        from src.api import app
        print("✅ API imported successfully")
        
        # Check if app is Flask instance
        from flask import Flask
        if isinstance(app, Flask):
            print("✅ App is a Flask instance")
        else:
            print("❌ App is not a Flask instance")
            return False
        
        # Check routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        required_routes = ["/health", "/predict", "/model_info"]
        print(f"\nFound {len(routes)} routes:")
        for route in routes[:5]:  # Show first 5
            print(f"  - {route}")
        
        missing_routes = [r for r in required_routes if r not in routes]
        if missing_routes:
            print(f"\n❌ Missing routes: {missing_routes}")
            return False
        else:
            print(f"\n✅ All required routes found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing API: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gunicorn_command():
    """Test if gunicorn command is valid"""
    print()
    print("=" * 60)
    print("TESTING GUNICORN COMMAND")
    print("=" * 60)
    print()
    
    command = "gunicorn src.api:app --bind 0.0.0.0:$PORT"
    print(f"Command: {command}")
    print()
    
    # Check if gunicorn is installed
    try:
        import subprocess
        result = subprocess.run(
            ["gunicorn", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✅ Gunicorn installed: {result.stdout.strip()}")
        else:
            print("⚠️  Gunicorn not found in PATH (may need to install)")
    except FileNotFoundError:
        print("⚠️  Gunicorn not found in PATH")
        print("   (This is OK - Render will install it from requirements.txt)")
    except Exception as e:
        print(f"⚠️  Could not check gunicorn: {e}")
    
    # Check if the module path is correct
    if os.path.exists("src/api.py"):
        print("✅ src/api.py exists (module path is correct)")
    else:
        print("❌ src/api.py not found (module path is incorrect)")

if __name__ == "__main__":
    print()
    print("🚀 RENDER DEPLOYMENT READINESS CHECK")
    print()
    
    # Run all checks
    check_render_config()
    test_local_api()
    test_gunicorn_command()
    
    print()
    print("✅ Configuration check complete!")
    print()
    print("💡 TIP: After deploying to Render, test the health endpoint:")
    print("   curl https://your-api-url.onrender.com/health")

