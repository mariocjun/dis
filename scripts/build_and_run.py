import os
import subprocess
import sys
import platform
import datetime
import yaml
import shutil
import argparse

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
    parser = argparse.ArgumentParser(description="Build and Run Ultrasound Benchmark")
    parser.add_argument("--debug", action="store_true", help="Build in Debug mode")
    parser.add_argument("--release", action="store_true", help="Build in Release mode")
    parser.add_argument("--clean", action="store_true", help="Clean build directory before building")
    args = parser.parse_args()

    # Default to Release if neither or Release is specified
    build_type = "Debug" if args.debug else "Release"
    
    # Fix: Script is in scripts/, so root is one level up
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_root, "build")
    
    build_dir = os.path.join(project_root, "build")
    
    def remove_readonly(func, path, excinfo):
        """Error handler for shutil.rmtree to remove read-only files (like git objects)"""
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if args.clean:
        if os.path.exists(build_dir):
            print(f"Cleaning build directory: {build_dir}")
            try:
                # Use onerror handler to fix permission issues on Windows
                shutil.rmtree(build_dir, onerror=remove_readonly)
            except Exception as e:
                print(f"Warning: Failed to clean build directory: {e}")
        else:
            print("Build directory does not exist, nothing to clean.")

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
            tol_str = tol_str.replace("e-0", "e-").replace(".", "") 
            
            # Timestamp
            now = datetime.datetime.now()
            timestamp = now.strftime("%d_%m_%Y_%H_%M")
            
            # Folder Name: DD_MM_AAAA_HH_MIN_tol_e-4_ite_10_omp_8
            folder_name = f"{timestamp}_tol_{tol}_ite_{ite}_omp_{omp}"
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
    
    print(f"Configuring CMake in: {build_dir} with Build Type: {build_type}")
    
    # Configure command (Ninja is single-config so we must pass CMAKE_BUILD_TYPE here)
    run_command(["cmake", "-S", ".", "-B", "build", f"-DCMAKE_BUILD_TYPE={build_type}"], cwd=project_root)
    
    print(f"Building project ({build_type})...")
    # Build command
    run_command(["cmake", "--build", "build", "--config", build_type], cwd=project_root)
    
    # Find Executable
    exe_path = os.path.join(build_dir, exe_name)
    # Check subfolders for Multi-Config generators (like VS)
    if not os.path.exists(exe_path):
        exe_path_typed = os.path.join(build_dir, build_type, exe_name)
        if os.path.exists(exe_path_typed):
            exe_path = exe_path_typed
    
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

    # 4. Automatic Graph Generation (Metrics)
    print("\n--- Generating Convergence Graphs ---")
    plot_script = os.path.join(project_root, "scripts", "plot_convergence.py")
    metrics_dir = os.path.join(output_path, "metrics")
    
    if os.path.exists(metrics_dir) and len(os.listdir(metrics_dir)) > 0:
        cmd_plot = ["python", plot_script, metrics_dir]
        try:
             subprocess.run(cmd_plot, check=True)
        except subprocess.CalledProcessError as e:
             print(f"Graph generation failed: {e}")
    else:
        print(f"No metrics found in {metrics_dir}, skipping graphs.")

    # 5. Run Visualization automatically (Images/Animations)
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
