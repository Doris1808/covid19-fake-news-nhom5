import sys

c.ServerProxy.servers = {
    "demo": {
        "command": [sys.executable, "-m", "streamlit", "run", "demo/streamlit_app.py",
                    "--server.address=127.0.0.1", "--server.port={port}",
                    "--server.headless=true", "--browser.gatherUsageStats=false"],
        "timeout": 90,
        "launcher_entry": {"title": "Demo Streamlit nhóm 5"},
    }
}
