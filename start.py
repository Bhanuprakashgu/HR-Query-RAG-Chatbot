#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
import webbrowser

try:
    import requests
except Exception:
    requests = None


def wait_for(url: str, timeout: float = 30.0) -> bool:
    if requests is None:
        # If requests isn't available for some reason, just wait a bit and hope the service comes up
        time.sleep(3)
        return True
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    env = os.environ.copy()
    env.setdefault("API_BASE", "http://localhost:8000")

    processes = []

    try:
        print("[start] Launching FastAPI backend (uvicorn) on http://localhost:8000 …")
        uvicorn_cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
        p_api = subprocess.Popen(uvicorn_cmd, env=env)
        processes.append(p_api)

        # Wait for API health
        if wait_for("http://localhost:8000/health", timeout=60):
            print("[ok] API is up at http://localhost:8000")
        else:
            print("[warn] API health check did not respond in time. Continuing…")

        print("[start] Launching Streamlit UI on http://localhost:8501 …")
        streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.headless", "true"]
        p_ui = subprocess.Popen(streamlit_cmd, env=env)
        processes.append(p_ui)

        # Give Streamlit a moment, then open browser
        time.sleep(2)
        try:
            webbrowser.open("http://localhost:8501")
        except Exception:
            pass

        print("[start] Launching MCP server (Python)")
        mcp_cmd = [sys.executable, "mcp_server.py"]
        p_mcp = subprocess.Popen(mcp_cmd, env=env)
        processes.append(p_mcp)

        print("\nAll services started:\n- Backend:   http://localhost:8000\n- Streamlit: http://localhost:8501\n- MCP:       python offline/mcp_server.py (connected to API_BASE)\n\nPress Ctrl+C to stop all.\n")

        # Wait for any process to exit
        while True:
            exited = [p for p in processes if p.poll() is not None]
            if exited:
                code = exited[0].returncode
                print(f"[exit] A service exited with code {code}. Shutting down…")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[stop] Keyboard interrupt received. Shutting down…")
    finally:
        # Terminate all child processes
        for p in processes:
            try:
                if p.poll() is None:
                    if os.name == "nt":
                        p.terminate()
                    else:
                        p.send_signal(signal.SIGINT)
                        time.sleep(0.5)
                        if p.poll() is None:
                            p.terminate()
                    # Last resort
                    time.sleep(0.5)
                    if p.poll() is None:
                        p.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()