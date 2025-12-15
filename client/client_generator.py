"""
Client Generator for Ultrasound Reconstruction Benchmark
Sends requests to both Python and C++ servers with identical sequences

Features:
- Seed-based reproducibility
- Random intervals between requests
- Random gain and dataset selection
- Collects results for comparison
"""

import os
import sys
import json
import time
import random
import requests
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import csv
import yaml


@dataclass
class JobRequest:
    """Job request specification"""
    job_id: str
    dataset_id: str
    gain: float
    seed: int
    epsilon_tolerance: float
    max_iterations: int
    interval_before_ms: float


@dataclass
class JobResult:
    """Result from a server"""
    job_id: str
    server: str
    status: str
    dataset_id: str
    gain: float
    iterations: int
    final_error: float
    final_epsilon: float
    converged: bool
    solver_time_ms: float
    latency_ms: float
    timestamp_start: str
    timestamp_end: str
    error_message: Optional[str] = None


class ClientGenerator:
    """Generates and sends requests to reconstruction servers"""

    def __init__(
            self,
            python_url: str = "http://localhost:5001",
            cpp_url: str = "http://localhost:5002",
            seed: int = 42,
            output_dir: str = "execs/benchmark"
    ):
        self.python_url = python_url
        self.cpp_url = cpp_url
        self.seed = seed
        self.rng = random.Random(seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[JobResult] = []
        self.requests: List[JobRequest] = []

    def generate_requests(
            self,
            num_jobs: int,
            datasets: List[str],
            gain_range: tuple = (0.5, 2.0),
            interval_range_ms: tuple = (100, 500),
            epsilon_tolerance: float = 1e-4,
            max_iterations: int = 10
    ) -> List[JobRequest]:
        """Generate a sequence of job requests with random parameters"""

        requests = []
        for i in range(num_jobs):
            job_id = f"job_{i:04d}_{self.seed}"
            dataset_id = self.rng.choice(datasets)
            gain = self.rng.uniform(*gain_range)
            interval_before = self.rng.uniform(*interval_range_ms)

            req = JobRequest(
                job_id=job_id,
                dataset_id=dataset_id,
                gain=gain,
                seed=self.seed,
                epsilon_tolerance=epsilon_tolerance,
                max_iterations=max_iterations,
                interval_before_ms=interval_before
            )
            requests.append(req)

        self.requests = requests
        return requests

    def send_request(self, server_url: str, server_name: str, req: JobRequest) -> JobResult:
        """Send a single request to a server"""
        start_time = time.perf_counter()

        try:
            payload = {
                'job_id': req.job_id,
                'dataset_id': req.dataset_id,
                'gain': req.gain,
                'seed': req.seed,
                'epsilon_tolerance': req.epsilon_tolerance,
                'max_iterations': req.max_iterations
            }

            # AUMENTADO O TIMEOUT PARA 1200 SEGUNDOS (20 MINUTOS)
            # O Python é muito lento com 3 clientes simultâneos em matrizes 60x60
            response = requests.post(
                f"{server_url}/solve",
                json=payload,
                timeout=1200
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return JobResult(
                    job_id=req.job_id,
                    server=server_name,
                    status='completed',
                    dataset_id=req.dataset_id,
                    gain=req.gain,
                    iterations=data.get('iterations', 0),
                    final_error=data.get('final_error', 0),
                    final_epsilon=data.get('final_epsilon', 0),
                    converged=data.get('converged', False),
                    solver_time_ms=data.get('solver_time_ms', 0),
                    latency_ms=latency_ms,
                    timestamp_start=data.get('timestamp_start', ''),
                    timestamp_end=data.get('timestamp_end', '')
                )
            else:
                return JobResult(
                    job_id=req.job_id,
                    server=server_name,
                    status='failed',
                    dataset_id=req.dataset_id,
                    gain=req.gain,
                    iterations=0,
                    final_error=0,
                    final_epsilon=0,
                    converged=False,
                    solver_time_ms=0,
                    latency_ms=latency_ms,
                    timestamp_start='',
                    timestamp_end='',
                    error_message=f"Status {response.status_code}: {response.text[:200]}"
                )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            print(f"  [ERROR] Request failed: {e}")
            return JobResult(
                job_id=req.job_id,
                server=server_name,
                status='error',
                dataset_id=req.dataset_id,
                gain=req.gain,
                iterations=0,
                final_error=0,
                final_epsilon=0,
                converged=False,
                solver_time_ms=0,
                latency_ms=latency_ms,
                timestamp_start='',
                timestamp_end='',
                error_message=str(e)[:200]
            )

    def run_sequential(self, server_url: str, server_name: str) -> List[JobResult]:
        """Run all requests sequentially to one server"""
        results = []

        for i, req in enumerate(self.requests):
            print(f"[{server_name}] Job {i+1}/{len(self.requests)}: {req.dataset_id} (gain={req.gain:.2f})")

            # Wait before sending (simulates random arrival)
            if i > 0:
                time.sleep(req.interval_before_ms / 1000)

            result = self.send_request(server_url, server_name, req)
            results.append(result)

            status = "✓" if result.status == 'completed' else "✗"
            print(f"  {status} {result.iterations} iter, {result.solver_time_ms:.1f}ms solver, {result.latency_ms:.1f}ms total")

            # Save partial results to avoid data loss on crash
            self.append_result_to_csv(result)

        return results

    def append_result_to_csv(self, result: JobResult):
        """Appends a single result to the CSV file immediately"""
        csv_path = self.output_dir / f"benchmark_results_partial.csv"
        file_exists = csv_path.exists()

        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(result).keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(asdict(result))
        except Exception as e:
            print(f"[WARN] Failed to save partial result: {e}")

    def run_benchmark(self, python_only: bool = False, cpp_only: bool = False) -> Dict[str, List[JobResult]]:
        """Run benchmark against both servers"""
        results = {}

        if not cpp_only:
            print("\n" + "="*60)
            print("Running benchmark against PYTHON server")
            print("="*60)
            try:
                # Check health first
                resp = requests.get(f"{self.python_url}/health", timeout=5)
                if resp.status_code == 200:
                    results['python'] = self.run_sequential(self.python_url, 'python')
                else:
                    print(f"[ERROR] Python server health check failed: {resp.status_code}")
            except Exception as e:
                print(f"[ERROR] Python server not available: {e}")

        if not python_only:
            print("\n" + "="*60)
            print("Running benchmark against C++ server")
            print("="*60)
            try:
                resp = requests.get(f"{self.cpp_url}/health", timeout=5)
                if resp.status_code == 200:
                    results['cpp'] = self.run_sequential(self.cpp_url, 'cpp')
                else:
                    print(f"[ERROR] C++ server health check failed: {resp.status_code}")
            except Exception as e:
                print(f"[ERROR] C++ server not available: {e}")

        self.results = results
        return results

    def save_results(self):
        """Save results to CSV and JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save all results to single CSV
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        all_results = []
        for server, server_results in self.results.items():
            all_results.extend(server_results)

        if all_results:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(all_results[0]).keys())
                writer.writeheader()
                for r in all_results:
                    writer.writerow(asdict(r))
            print(f"\n[INFO] Results saved to: {csv_path}")

        # Save requests (for reproducibility)
        requests_path = self.output_dir / f"benchmark_requests_{timestamp}.json"
        with open(requests_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.requests], f, indent=2)
        print(f"[INFO] Requests saved to: {requests_path}")

        # Generate summary
        self.print_summary()

    def print_summary(self):
        """Print comparison summary"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        for server, server_results in self.results.items():
            completed = [r for r in server_results if r.status == 'completed']

            if completed:
                avg_solver = sum(r.solver_time_ms for r in completed) / len(completed)
                avg_latency = sum(r.latency_ms for r in completed) / len(completed)
                total_solver = sum(r.solver_time_ms for r in completed)

                print(f"\n{server.upper()} Server:")
                print(f"  Completed: {len(completed)}/{len(server_results)}")
                print(f"  Avg Solver Time: {avg_solver:.2f} ms")
                print(f"  Avg Latency: {avg_latency:.2f} ms")
                print(f"  Total Solver Time: {total_solver:.2f} ms")
                print(f"  Throughput: {len(completed) / (total_solver/1000):.2f} jobs/sec")


def main():
    parser = argparse.ArgumentParser(description='Ultrasound Reconstruction Benchmark Client')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num-jobs', type=int, default=10, help='Number of jobs to send')
    parser.add_argument('--python-url', default='http://localhost:5001', help='Python server URL')
    parser.add_argument('--cpp-url', default='http://localhost:5002', help='C++ server URL')
    parser.add_argument('--output-dir', default='execs/benchmark', help='Output directory')
    parser.add_argument('--python-only', action='store_true', help='Only test Python server')
    parser.add_argument('--cpp-only', action='store_true', help='Only test C++ server')
    parser.add_argument('--datasets', nargs='+', default=['30x30_g1', '30x30_g2', '60x60_G1'],
                        help='Datasets to use')
    parser.add_argument('--gain-min', type=float, default=0.5, help='Minimum gain')
    parser.add_argument('--gain-max', type=float, default=2.0, help='Maximum gain')
    parser.add_argument('--interval-min', type=float, default=100, help='Min interval (ms)')
    parser.add_argument('--interval-max', type=float, default=500, help='Max interval (ms)')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Epsilon tolerance')
    parser.add_argument('--max-iter', type=int, default=10, help='Max iterations')

    args = parser.parse_args()

    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] Jobs: {args.num_jobs}")
    print(f"[INFO] Datasets: {args.datasets}")

    client = ClientGenerator(
        python_url=args.python_url,
        cpp_url=args.cpp_url,
        seed=args.seed,
        output_dir=args.output_dir
    )

    client.generate_requests(
        num_jobs=args.num_jobs,
        datasets=args.datasets,
        gain_range=(args.gain_min, args.gain_max),
        interval_range_ms=(args.interval_min, args.interval_max),
        epsilon_tolerance=args.epsilon,
        max_iterations=args.max_iter
    )

    client.run_benchmark(python_only=args.python_only, cpp_only=args.cpp_only)
    client.save_results()


if __name__ == '__main__':
    main()