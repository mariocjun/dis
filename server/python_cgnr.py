"""
Python CGNR Solver - Pure NumPy Implementation
CGNR: Conjugate Gradient for Normal Equations Residual
Solves: min ||Hx - g||^2 via H^T H x = H^T g

Stop rule: abs(||r_new|| - ||r_old||) < epsilon_tolerance OR iterations >= max_iterations
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time


@dataclass
class CGNRResult:
    """Result container for CGNR solver"""
    image: np.ndarray           # Reconstructed image vector
    iterations: int             # Number of iterations executed
    final_error: float          # Final ||r|| (residual norm)
    final_epsilon: float        # Final abs(||r_new|| - ||r_old||)
    converged: bool             # True if stopped by tolerance, False if max_iter
    execution_time_ms: float    # Solver time in milliseconds
    residual_history: List[float]   # History of ||r_i||
    solution_history: List[float]   # History of ||x_i||


def cgnr_solve(
    H: np.ndarray,
    g: np.ndarray,
    max_iterations: int = 10,
    epsilon_tolerance: float = 1e-4,
    normalize: bool = True
) -> CGNRResult:
    """
    CGNR (Conjugate Gradient for Normal Equations Residual)
    
    Solves the least squares problem: min ||Hx - g||^2
    by solving the normal equations: H^T H x = H^T g
    
    Args:
        H: System matrix (m x n)
        g: Measurement signal vector (m,)
        max_iterations: Maximum number of iterations
        epsilon_tolerance: Stop when abs(||r_new|| - ||r_old||) < epsilon_tolerance
        normalize: If True, normalize H and g by row norms before solving
    
    Returns:
        CGNRResult with reconstructed image and metrics
    """
    start_time = time.perf_counter()
    
    # Ensure proper shapes
    H = np.asarray(H, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64).flatten()
    
    m, n = H.shape
    
    # Optional row normalization (same as C++ implementation)
    if normalize:
        H, g = normalize_system_rows(H, g)
    
    # Initialize
    x = np.zeros(n, dtype=np.float64)  # Initial guess: zero vector
    
    # r = g - H @ x = g (since x=0)
    r = g.copy()
    
    # z = H^T @ r
    z = H.T @ r
    
    # p = z (search direction)
    p = z.copy()
    
    # Tracking
    residual_history = []
    solution_history = []
    
    r_norm_prev = np.linalg.norm(r)
    residual_history.append(r_norm_prev)
    solution_history.append(np.linalg.norm(x))
    
    converged = False
    final_epsilon = float('inf')
    
    for k in range(max_iterations):
        # w = H @ p
        w = H @ p
        
        # alpha = ||z||^2 / ||w||^2
        z_norm_sq = np.dot(z, z)
        w_norm_sq = np.dot(w, w)
        
        if w_norm_sq < 1e-30:
            # Avoid division by zero - already converged
            converged = True
            break
        
        alpha = z_norm_sq / w_norm_sq
        
        # x = x + alpha * p
        x = x + alpha * p
        
        # r = r - alpha * w
        r = r - alpha * w
        
        # z_new = H^T @ r
        z_new = H.T @ r
        
        # beta = ||z_new||^2 / ||z||^2
        z_new_norm_sq = np.dot(z_new, z_new)
        
        if z_norm_sq < 1e-30:
            converged = True
            break
            
        beta = z_new_norm_sq / z_norm_sq
        
        # p = z_new + beta * p
        p = z_new + beta * p
        
        # Update z
        z = z_new
        
        # Track norms
        r_norm = np.linalg.norm(r)
        x_norm = np.linalg.norm(x)
        
        residual_history.append(r_norm)
        solution_history.append(x_norm)
        
        # Calculate epsilon (same as C++: abs difference of residual norms)
        epsilon = abs(r_norm - r_norm_prev)
        final_epsilon = epsilon
        
        # Check convergence
        if epsilon < epsilon_tolerance:
            converged = True
            break
        
        r_norm_prev = r_norm
    
    execution_time_ms = (time.perf_counter() - start_time) * 1000
    
    return CGNRResult(
        image=x,
        iterations=len(residual_history) - 1,  # -1 because first entry is initial
        final_error=residual_history[-1] if residual_history else 0.0,
        final_epsilon=final_epsilon,
        converged=converged,
        execution_time_ms=execution_time_ms,
        residual_history=residual_history,
        solution_history=solution_history
    )


def normalize_system_rows(H: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize each row of H and corresponding element of g by the row norm.
    Same logic as C++ normalize_system_rows.
    
    Args:
        H: System matrix (m x n)
        g: Measurement vector (m,)
    
    Returns:
        Tuple of (H_normalized, g_normalized)
    """
    H = H.copy()
    g = g.copy()
    
    m = H.shape[0]
    
    for i in range(m):
        row_norm = np.linalg.norm(H[i, :])
        if row_norm > 1e-12:
            H[i, :] /= row_norm
            g[i] /= row_norm
    
    return H, g


def load_matrix_csv(filepath: str) -> np.ndarray:
    """Load a matrix from CSV file"""
    return np.loadtxt(filepath, delimiter=',', dtype=np.float64)


def load_vector_csv(filepath: str) -> np.ndarray:
    """Load a vector from CSV file (can be multi-line or comma-separated)"""
    try:
        # Try loading as 2D and flatten
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float64)
        return data.flatten()
    except:
        # Try loading as 1D
        return np.loadtxt(filepath, dtype=np.float64)


if __name__ == "__main__":
    # Quick test with small synthetic problem
    print("Testing CGNR solver...")
    
    # Create a simple test problem: identity matrix
    n = 10
    H = np.eye(n)
    x_true = np.random.randn(n)
    g = H @ x_true
    
    result = cgnr_solve(H, g, max_iterations=10, epsilon_tolerance=1e-4, normalize=False)
    
    print(f"True solution norm: {np.linalg.norm(x_true):.6f}")
    print(f"Reconstructed norm: {np.linalg.norm(result.image):.6f}")
    print(f"Error norm: {np.linalg.norm(result.image - x_true):.6f}")
    print(f"Iterations: {result.iterations}")
    print(f"Final error: {result.final_error:.6e}")
    print(f"Final epsilon: {result.final_epsilon:.6e}")
    print(f"Converged: {result.converged}")
    print(f"Time: {result.execution_time_ms:.2f} ms")
    print("CGNR test PASSED!" if np.linalg.norm(result.image - x_true) < 1e-6 else "CGNR test FAILED!")
