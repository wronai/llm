#!/usr/bin/env python
"""
Wrapper script to run the WronAI Streamlit application.
This resolves import issues by running the app as a module.
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Modify the app.py file to use absolute imports instead of relative imports
    app_path = os.path.join(project_root, "wronai", "web", "app.py")
    
    # Create a modified environment with the project root in PYTHONPATH
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root
    
    # Run the Streamlit app using subprocess
    cmd = ["streamlit", "run", app_path, "--server.headless", "true"]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, env=env)
        sys.exit(process.returncode)
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
