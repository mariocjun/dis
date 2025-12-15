"""
Python Ultrasound Reconstruction Server
Flask-based HTTP server with CGNR solver

Endpoints:
- POST /solve - Reconstruct image from signal
- GET /health - Health check
- GET /metrics - Prometheus metrics
"""

import os
import sys
import json
import time
import uuid
import threading
import psutil
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, Optional

import numpy as np
from flask import Flask, request, jsonify, Response
import yaml

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.python_cgnr import cgnr_solve, load_matrix_csv, load_vector_csv

app = Flask(__name__)

# ----- Global State -----
class ServerState:
    """Thread-safe server state"""
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs_total = 0
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.jobs_in_progress = 0
        self.queue_length = 0
        self.start_time = time.time()
        self.config: Dict[str, Any] = {}
        self.datasets: Dict[str, Dict] = {}
        self.output_dir: Path = Path("execs/server_output")
        
    def load_config(self, config_path: str):
        """Load server configuration from YAML"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Build dataset map
        for ds in self.config.get('datasets', []):
            self.datasets[ds['name']] = ds
        
        # Set settings
        settings = self.config.get('settings', {})
        self.epsilon_tolerance = settings.get('epsilon_tolerance', 1e-4)
        self.max_iterations = settings.get('max_iterations', 10)
        self.output_dir = Path(settings.get('output_base_dir', 'execs/server_output'))
        
        # Ensure output directories exist
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'telemetry').mkdir(parents=True, exist_ok=True)

state = ServerState()

# ----- Telemetry -----
def get_process_metrics() -> Dict[str, float]:
    """Get current process metrics"""
    process = psutil.Process()
    return {
        'cpu_percent': process.cpu_percent(),
        'memory_mb': process.memory_info().rss / (1024 * 1024),
        'threads': process.num_threads()
    }


def save_job_metrics(job_id: str, metrics: Dict[str, Any]):
    """Append job metrics to CSV"""
    csv_path = state.output_dir / 'telemetry' / 'job_metrics.csv'
    
    # Write header if file doesn't exist
    if not csv_path.exists():
        header = "job_id,timestamp_start,timestamp_end,server,dataset_id,gain,seed,iterations,final_error,final_epsilon,converged,latency_ms,solver_time_ms,ram_peak_mb,cpu_avg_pct\n"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(header)
    
    # Append row
    row = f"{metrics['job_id']},{metrics['timestamp_start']},{metrics['timestamp_end']},python,{metrics['dataset_id']},{metrics['gain']},{metrics['seed']},{metrics['iterations']},{metrics['final_error']:.6e},{metrics['final_epsilon']:.6e},{metrics['converged']},{metrics['latency_ms']:.2f},{metrics['solver_time_ms']:.2f},{metrics.get('ram_peak_mb', 0):.2f},{metrics.get('cpu_avg_pct', 0):.2f}\n"
    
    with open(csv_path, 'a', encoding='utf-8') as f:
        f.write(row)


def save_image_with_metadata(image: np.ndarray, job_id: str, metadata: Dict[str, Any], rows: int, cols: int):
    """Save reconstructed image as PNG (with overlay), CSV, and JSON metadata"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    images_dir = state.output_dir / 'images'
    
    # Reshape to 2D image
    img_2d = image.reshape(rows, cols)
    
    # Save CSV (raw matrix)
    csv_path = images_dir / f"{job_id}_image.csv"
    np.savetxt(csv_path, img_2d, delimiter=',', fmt='%.10e')
    
    # Save JSON metadata
    json_path = images_dir / f"{job_id}_meta.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save PNG with metadata overlay
    png_path = images_dir / f"{job_id}_image.png"
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use appropriate colormap for ultrasound (dark background)
    im = ax.imshow(img_2d, cmap='gray', aspect='equal')
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Add metadata as title
    title = f"Algorithm: CGNR | Size: {rows}x{cols}\n"
    title += f"Start: {metadata.get('timestamp_start', 'N/A')[:19]} | End: {metadata.get('timestamp_end', 'N/A')[:19]}\n"
    title += f"Iterations: {metadata.get('iterations', 'N/A')} | Error: {metadata.get('final_error', 0):.2e} | ε: {metadata.get('final_epsilon', 0):.2e}"
    ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return {
        'image_csv_path': str(csv_path.relative_to(state.output_dir)),
        'image_png_path': str(png_path.relative_to(state.output_dir)),
        'metadata_json_path': str(json_path.relative_to(state.output_dir))
    }


def save_iteration_images(iteration_images: list, job_id: str, rows: int, cols: int):
    """Save iteration images as CSV files for carousel visualization."""
    images_dir = state.output_dir / 'images'
    
    for i, img_vec in enumerate(iteration_images):
        img_2d = img_vec.reshape(rows, cols)
        csv_path = images_dir / f"{job_id}_iter_{i:02d}.csv"
        np.savetxt(csv_path, img_2d, delimiter=',', fmt='%.10e')


def save_lcurve_csv(job_id: str, residual_history: list, solution_history: list):
    """Save L-curve data to CSV for proper visualization.
    
    Format: Iteration,SolutionNorm,ResidualNorm
    This allows generating proper L-curve plots (residual norm vs solution norm).
    """
    images_dir = state.output_dir / 'images'
    csv_path = images_dir / f"{job_id}_lcurve.csv"
    
    try:
        with open(csv_path, 'w') as f:
            f.write("Iteration,SolutionNorm,ResidualNorm\n")
            for i, (sol_norm, res_norm) in enumerate(zip(solution_history, residual_history)):
                f.write(f"{i+1},{sol_norm:.10e},{res_norm:.10e}\n")
    except Exception as e:
        print(f"[WARN] Failed to save L-curve: {e}")


# ----- API Endpoints -----
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'server': 'python',
        'uptime_seconds': time.time() - state.start_time
    })


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    process = psutil.Process()
    mem_info = process.memory_info()
    
    metrics_text = f"""# HELP jobs_total Total number of jobs received
# TYPE jobs_total counter
jobs_total {state.jobs_total}

# HELP jobs_completed Total number of jobs completed successfully
# TYPE jobs_completed counter  
jobs_completed {state.jobs_completed}

# HELP jobs_failed Total number of jobs failed
# TYPE jobs_failed counter
jobs_failed {state.jobs_failed}

# HELP jobs_in_progress Current jobs being processed
# TYPE jobs_in_progress gauge
jobs_in_progress {state.jobs_in_progress}

# HELP queue_length Current queue length
# TYPE queue_length gauge
queue_length {state.queue_length}

# HELP process_resident_memory_bytes Resident memory size in bytes
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes {mem_info.rss}

# HELP process_cpu_seconds_total Total CPU time spent
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total {process.cpu_times().user + process.cpu_times().system}

# HELP process_threads_active Number of active threads
# TYPE process_threads_active gauge
process_threads_active {process.num_threads()}
"""
    return Response(metrics_text, mimetype='text/plain')


@app.route('/solve', methods=['POST'])
def solve():
    """
    Reconstruct image from signal using CGNR
    
    Request JSON:
    {
        "job_id": "optional-uuid",
        "dataset_id": "30x30_g1",
        "gain": 1.0,                    # optional, default 1.0
        "seed": 42,                     # optional
        "epsilon_tolerance": 1e-4,      # optional
        "max_iterations": 10            # optional
    }
    """
    timestamp_start = datetime.utcnow().isoformat() + 'Z'
    start_time = time.perf_counter()
    
    with state.lock:
        state.jobs_total += 1
        state.jobs_in_progress += 1
    
    try:
        data = request.get_json()
        
        # Parse request
        job_id = data.get('job_id', str(uuid.uuid4()))
        dataset_id = data.get('dataset_id')
        gain = float(data.get('gain', 1.0))
        seed = data.get('seed', None)
        epsilon_tolerance = float(data.get('epsilon_tolerance', state.epsilon_tolerance))
        max_iterations = int(data.get('max_iterations', state.max_iterations))
        
        # Validate dataset
        if dataset_id not in state.datasets:
            return jsonify({
                'error': f"Unknown dataset_id: {dataset_id}",
                'available': list(state.datasets.keys())
            }), 400
        
        ds = state.datasets[dataset_id]
        
        # Resolve paths relative to config location
        config_dir = Path(state.config.get('_config_path', '.')).parent
        h_path = config_dir / ds['h_matrix_csv']
        g_path = config_dir / ds['g_signal_csv']
        
        if not h_path.exists():
            return jsonify({'error': f"H matrix file not found: {h_path}"}), 500
        if not g_path.exists():
            return jsonify({'error': f"G signal file not found: {g_path}"}), 500
        
        # Load data
        H = load_matrix_csv(str(h_path))
        g = load_vector_csv(str(g_path))
        
        # Apply gain
        g = g * gain
        
        # Get process metrics before solve
        ram_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Run CGNR solver
        result = cgnr_solve(
            H, g,
            max_iterations=max_iterations,
            epsilon_tolerance=epsilon_tolerance,
            normalize=True
        )
        
        # Get process metrics after solve
        ram_after = psutil.Process().memory_info().rss / (1024 * 1024)
        ram_peak = max(ram_before, ram_after)
        
        timestamp_end = datetime.utcnow().isoformat() + 'Z'
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Build metadata
        metadata = {
            'job_id': job_id,
            'algorithm': 'CGNR',
            'dataset_id': dataset_id,
            'image_size': {'rows': ds['image_rows'], 'cols': ds['image_cols']},
            'gain': gain,
            'seed': seed,
            'iterations': result.iterations,
            'final_error': float(result.final_error),
            'final_epsilon': float(result.final_epsilon),
            'converged': result.converged,
            'timestamp_start': timestamp_start,
            'timestamp_end': timestamp_end,
            'solver_time_ms': result.execution_time_ms,
            'server': 'python'
        }
        
        # Save image with metadata
        image_paths = save_image_with_metadata(
            result.image, job_id, metadata,
            ds['image_rows'], ds['image_cols']
        )
        metadata.update(image_paths)
        
        # Save iteration images for carousel visualization
        if hasattr(result, 'iteration_images') and result.iteration_images:
            save_iteration_images(
                result.iteration_images, job_id,
                ds['image_rows'], ds['image_cols']
            )
        
        # Save L-curve data for proper L-curve visualization
        if hasattr(result, 'residual_history') and hasattr(result, 'solution_history'):
            save_lcurve_csv(job_id, result.residual_history, result.solution_history)
        
        # Save telemetry
        telemetry = {
            **metadata,
            'latency_ms': latency_ms,
            'ram_peak_mb': ram_peak,
            'cpu_avg_pct': psutil.Process().cpu_percent()
        }
        save_job_metrics(job_id, telemetry)
        
        with state.lock:
            state.jobs_completed += 1
            state.jobs_in_progress -= 1
        
        # Response
        response = {
            'job_id': job_id,
            'status': 'completed',
            **metadata
        }
        
        return jsonify(response)
        
    except Exception as e:
        with state.lock:
            state.jobs_failed += 1
            state.jobs_in_progress -= 1
        
        return jsonify({
            'error': str(e),
            'job_id': data.get('job_id', 'unknown') if 'data' in dir() else 'unknown'
        }), 500


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Python Ultrasound Reconstruction Server')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--port', type=int, default=5001, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script
        config_path = Path(__file__).parent.parent / 'config.yaml'
    
    if config_path.exists():
        state.load_config(str(config_path))
        state.config['_config_path'] = str(config_path)
        print(f"[INFO] Loaded config from: {config_path}")
        print(f"[INFO] Available datasets: {list(state.datasets.keys())}")
    else:
        print(f"[WARN] Config not found: {args.config}")
    
    print(f"[INFO] Starting Python server on {args.host}:{args.port}")
    print(f"[INFO] Output directory: {state.output_dir}")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
