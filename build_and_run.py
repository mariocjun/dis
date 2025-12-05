import os
import subprocess
import sys
import platform

def run_command(command, cwd=None):
    """Runs a shell command and prints its output."""
    print(f"--- Running: {' '.join(command)} ---")
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=False  # Let output flow to stdout/stderr directly
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(e.returncode)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(project_root, "build")
    
    # Determine the executable name based on OS
    exe_name = "UltrasoundBenchmark"
    if platform.system() == "Windows":
        exe_name += ".exe"
    
    # CMake Configure
    print(f"Configuring CMake in: {build_dir}")
    run_command(["cmake", "-S", ".", "-B", "build"], cwd=project_root)
    
    # CMake Build
    print("Building project...")
    # --config Release is important for multi-configuration generators (like Visual Studio)
    run_command(["cmake", "--build", "build", "--config", "Release"], cwd=project_root)
    
    # Find the executable
    # On Windows with MSVC, it might be in build/Release/ or just build/
    # We'll try to find it.
    exe_path = os.path.join(build_dir, exe_name)
    if not os.path.exists(exe_path):
        # Try Release subdirectory (common in Windows/MSVC)
        exe_path_release = os.path.join(build_dir, "Release", exe_name)
        if os.path.exists(exe_path_release):
            exe_path = exe_path_release
        else:
            # Try Debug subdirectory just in case
            exe_path_debug = os.path.join(build_dir, "Debug", exe_name)
            if os.path.exists(exe_path_debug):
                exe_path = exe_path_debug
    
    if not os.path.exists(exe_path):
        print(f"Error: Could not find executable at {exe_path}")
        sys.exit(1)
        
    # Run the executable
    print(f"Executing: {exe_path}")
    # Pass any extra arguments from this script to the executable
    extra_args = sys.argv[1:]
    
    try:
        # On Windows, we might want to just call it directly
        cmd = [exe_path] + extra_args
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Application finished with error code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(130)

if __name__ == "__main__":
    main()
