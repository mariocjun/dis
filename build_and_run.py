import os
import subprocess
import sys
import platform
import datetime
import yaml

def run_command(command, cwd=None):
    """Runs a shell command and prints its output."""
    print(f"--- Running: {' '.join(command)} ---")
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=False
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(e.returncode)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(project_root, "build")
    
    # 1. Parse config to get parameters for folder naming
    config_path = os.path.join(project_root, "config.yaml")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            settings = config.get('settings', {})
            tol = settings.get('epsilon_tolerance', 'NA')
            ite = settings.get('max_iterations', 'NA')
            omp = settings.get('num_omp_threads', 'NA')
            
            # Format numbers for filename (scientific to simplified if needed)
            tol_str = f"{tol:.0e}" if isinstance(tol, float) else str(tol)
            tol_str = tol_str.replace("e-0", "e-").replace(".", "") # simplify 1e-06 -> 1e-6 approx or keep as is
            
            # Timestamp
            now = datetime.datetime.now()
            timestamp = now.strftime("%d_%m_%Y_%H_%M")
            
            # Folder Name: DD_MM_AAAA_HH_MIN_tol_e-4_ite_10_omp_8
            folder_name = f"{timestamp}_tol_{tol}_ite_{ite}_omp_{omp}"
            # Sanitize just in case
            folder_name = folder_name.replace(" ", "_").replace(":", "")
            
            output_base = "output_runs"
            output_path = os.path.join(output_base, folder_name)
            
            print(f"Directory for this run: {output_path}")

    except Exception as e:
        print(f"Warning: Could not parse config for naming. using default. Error: {e}")
        output_path = "output_runs/default_run"

    # 2. Build Project
    exe_name = "UltrasoundBenchmark"
    if platform.system() == "Windows":
        exe_name += ".exe"
    
    print(f"Configuring CMake in: {build_dir}")
    run_command(["cmake", "-S", ".", "-B", "build"], cwd=project_root)
    
    print("Building project...")
    run_command(["cmake", "--build", "build", "--config", "Release"], cwd=project_root)
    
    # Find Executable
    exe_path = os.path.join(build_dir, exe_name)
    if not os.path.exists(exe_path):
        exe_path_release = os.path.join(build_dir, "Release", exe_name)
        if os.path.exists(exe_path_release):
            exe_path = exe_path_release
        else:
            exe_path_debug = os.path.join(build_dir, "Debug", exe_name)
            if os.path.exists(exe_path_debug):
                exe_path = exe_path_debug
    
    if not os.path.exists(exe_path):
        print(f"Error: Could not find executable at {exe_path}")
        sys.exit(1)
        
    # 3. Run Executable with Custom Output Path
    print(f"Executing: {exe_path}")
    # Pass config file AND output directory
    cmd = [exe_path, "config.yaml", output_path]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Application finished with error code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(130)

    # 4. Run Visualization automatically
    print("\n--- Starting Visualization ---")
    visualize_script = os.path.join(project_root, "scripts", "visualize_iterations.py")
    images_dir = os.path.join(output_path, "images")
    
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        cmd_viz = ["python", visualize_script, images_dir]
        try:
             subprocess.run(cmd_viz, check=True)
        except subprocess.CalledProcessError as e:
             print(f"Visualization failed: {e}")
    else:
        print(f"No images found in {images_dir}, skipping visualization.")

if __name__ == "__main__":
    main()
