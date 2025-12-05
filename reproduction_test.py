import time
import socketio
import threading
import sys
import subprocess
import os
import requests

# Define colors for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_color(msg, color=Colors.OKBLUE):
    print(f"{color}{msg}{Colors.ENDC}")

# 1. Start Client
sio = socketio.Client()
benchmark_finished = threading.Event()

@sio.event
def connect():
    print_color("Connected to server!", Colors.OKGREEN)
    print_color("Sending 'start_benchmark' event...", Colors.OKCYAN)
    sio.emit('start_benchmark')

@sio.event
def disconnect():
    print_color("Disconnected from server.", Colors.WARNING)

@sio.on('log')
def on_log(data):
    print(f"[SERVER LOG] {data['data']}")

@sio.on('new_result')
def on_new_result(data):
    print_color(f"[RESULT] {data['filename']} processed. Speedup: {data['speedup']:.2f}x", Colors.OKGREEN)

@sio.on('benchmark_finished')
def on_benchmark_finished():
    print_color("Benchmark finished signal received!", Colors.OKGREEN)
    benchmark_finished.set()

def run_client():
    try:
        sio.connect('http://localhost:5000')
        sio.wait()
    except Exception as e:
        print_color(f"Client error: {e}", Colors.FAIL)

# 2. Server Runner Wrapper
def run_test():
    print_color("--- STARTING REPRODUCTION TEST ---", Colors.HEADER)

    # Ensure dependencies
    try:
        import flask_socketio
    except ImportError:
        print_color("Installing dependencies...", Colors.WARNING)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask-socketio", "python-socketio[client]", "requests"])

    # Start Server in Subprocess
    print_color("Starting web_app.py...", Colors.HEADER)
    server_process = subprocess.Popen(
        [sys.executable, "web_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd()
    )

    # Give it time to start
    time.sleep(3)
    
    # Check if server is running
    try:
        requests.get("http://localhost:5000")
    except requests.exceptions.ConnectionError:
        print_color("Server failed to start in time.", Colors.FAIL)
        out, err = server_process.communicate(timeout=1)
        print(out)
        print(err)
        return

    # Start Client Thread
    client_thread = threading.Thread(target=run_client)
    client_thread.daemon = True
    client_thread.start()

    # Wait for completion or timeout
    print_color("Waiting for benchmark to complete (Timeout: 60s)...", Colors.HEADER)
    finished = benchmark_finished.wait(timeout=60)

    if finished:
        print_color("SUCCESS: Benchmark ran and finished.", Colors.OKGREEN)
    else:
        print_color("FAILURE: Benchmark timed out or did not finish.", Colors.FAIL)

    # Cleanup
    sio.disconnect()
    server_process.terminate()
    try:
        server_process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        server_process.kill()

    print_color("--- TEST COMPLETE ---", Colors.HEADER)

if __name__ == "__main__":
    run_test()
