"""
Unit tests for Python CGNR solver
Tests stop rule, edge cases, and correctness
"""

import numpy as np
import pytest
from python_cgnr import cgnr_solve, normalize_system_rows, CGNRResult


class TestCGNRSolver:
    """Test suite for CGNR solver"""
    
    def test_identity_matrix_exact_solution(self):
        """CGNR should solve identity system exactly in 1 iteration"""
        n = 10
        H = np.eye(n)
        x_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        g = H @ x_true
        
        result = cgnr_solve(H, g, max_iterations=10, epsilon_tolerance=1e-4, normalize=False)
        
        assert np.allclose(result.image, x_true, atol=1e-10)
        assert result.iterations <= 2  # Should converge very fast
        assert result.converged == True
    
    def test_overdetermined_system(self):
        """CGNR should handle overdetermined systems (m > n)"""
        np.random.seed(42)
        m, n = 100, 50
        H = np.random.randn(m, n)
        x_true = np.random.randn(n)
        g = H @ x_true + 0.01 * np.random.randn(m)  # Add small noise
        
        result = cgnr_solve(H, g, max_iterations=10, epsilon_tolerance=1e-4)
        
        assert result.image.shape == (n,)
        assert len(result.residual_history) > 0
        assert result.execution_time_ms > 0
    
    def test_stop_by_tolerance(self):
        """Test that solver stops when epsilon < tolerance"""
        np.random.seed(42)
        n = 20
        # Use a more challenging matrix that doesn't converge instantly
        H = np.random.randn(30, n)
        x_true = np.random.randn(n)
        g = H @ x_true
        
        result = cgnr_solve(H, g, max_iterations=100, epsilon_tolerance=1e-4, normalize=True)
        
        # Should converge before max iterations
        assert result.iterations < 100
        # Either converged by tolerance or very early (like identity)
        assert result.converged == True or result.iterations <= 2
    
    def test_stop_by_max_iterations(self):
        """Test that solver stops at max_iterations if not converged"""
        np.random.seed(42)
        m, n = 100, 50
        H = np.random.randn(m, n)
        g = np.random.randn(m)
        
        result = cgnr_solve(H, g, max_iterations=3, epsilon_tolerance=1e-20)
        
        # With very tight tolerance, should stop by max_iter
        assert result.iterations <= 3
    
    def test_zero_signal(self):
        """CGNR should return zero solution for zero signal"""
        n = 10
        H = np.eye(n)
        g = np.zeros(n)
        
        result = cgnr_solve(H, g, max_iterations=10, epsilon_tolerance=1e-4, normalize=False)
        
        assert np.allclose(result.image, np.zeros(n), atol=1e-10)
    
    def test_epsilon_definition(self):
        """Verify epsilon is abs(||r_new|| - ||r_old||)"""
        np.random.seed(42)
        n = 20
        H = np.random.randn(30, n)
        g = np.random.randn(30)
        
        result = cgnr_solve(H, g, max_iterations=5, epsilon_tolerance=1e-10)
        
        # Check epsilon matches the definition
        if len(result.residual_history) >= 2:
            computed_epsilon = abs(result.residual_history[-1] - result.residual_history[-2])
            assert np.isclose(result.final_epsilon, computed_epsilon, rtol=1e-10)
    
    def test_residual_history_decreasing(self):
        """Residual norm should generally decrease (may plateau)"""
        np.random.seed(42)
        n = 20
        H = np.eye(n) + 0.1 * np.random.randn(n, n)
        H = H.T @ H  # Make it SPD-like
        x_true = np.random.randn(n)
        g = H @ x_true
        
        result = cgnr_solve(H, g, max_iterations=10, epsilon_tolerance=1e-10, normalize=False)
        
        # Residual should decrease or stay same (not increase significantly)
        for i in range(1, len(result.residual_history)):
            # Allow small numerical tolerance for increase
            assert result.residual_history[i] <= result.residual_history[i-1] * 1.01


class TestNormalization:
    """Test row normalization function"""
    
    def test_normalization_preserves_solution(self):
        """Normalization should not change the least squares solution"""
        np.random.seed(42)
        m, n = 30, 20
        H = np.random.randn(m, n)
        x_true = np.random.randn(n)
        g = H @ x_true
        
        # Solve without normalization
        result1 = cgnr_solve(H, g, max_iterations=20, epsilon_tolerance=1e-8, normalize=False)
        
        # Solve with normalization
        result2 = cgnr_solve(H, g, max_iterations=20, epsilon_tolerance=1e-8, normalize=True)
        
        # Solutions should be similar (normalization is just preconditioning)
        assert np.allclose(result1.image, result2.image, rtol=0.1)
    
    def test_normalize_handles_zero_rows(self):
        """Normalization should handle zero rows gracefully"""
        H = np.array([[1, 2, 3], [0, 0, 0], [4, 5, 6]], dtype=np.float64)
        g = np.array([1, 0, 2], dtype=np.float64)
        
        H_norm, g_norm = normalize_system_rows(H, g)
        
        # Zero row should remain zero
        assert np.allclose(H_norm[1, :], [0, 0, 0])
        assert g_norm[1] == 0


class TestResultDataclass:
    """Test CGNRResult dataclass"""
    
    def test_result_fields(self):
        """Verify all fields are populated correctly"""
        np.random.seed(42)
        n = 10
        H = np.eye(n)
        g = np.ones(n)
        
        result = cgnr_solve(H, g, max_iterations=5, epsilon_tolerance=1e-4)
        
        assert hasattr(result, 'image')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'final_error')
        assert hasattr(result, 'final_epsilon')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'execution_time_ms')
        assert hasattr(result, 'residual_history')
        assert hasattr(result, 'solution_history')
        
        assert isinstance(result.image, np.ndarray)
        assert isinstance(result.iterations, int)
        assert isinstance(result.converged, bool)
        assert result.execution_time_ms >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
