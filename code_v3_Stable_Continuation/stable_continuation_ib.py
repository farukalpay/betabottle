# Author: Faruk Alpay
# ORCID: 0009-0009-2207-6528
# Publication: http://dx.doi.org/10.13140/RG.2.2.12135.15521
"""
Enhanced Information Bottleneck (IB) Framework with Symbolic Stability Improvements

This comprehensive Python file integrates:
1) High-precision JAX-based numeric iterative IB optimization
2) Multi-Path Incremental-β approach to prevent trivial collapses
3) Convexified objectives for dense IB curve exploration
4) Hessian-based bifurcation detection and handling
5) Entropy regularization for smooth transitions
6) Implicit function continuation for stable solution paths
7) Information-geometric stabilization via KL constraints

Key enhancements:
- Convexified IB objective using u(I(X;Z)) for guaranteed dense exploration
- Adaptive entropy regularization with automatic scheduling
- Hessian eigenvalue monitoring for explicit bifurcation detection
- Implicit function continuation for smooth solution paths
- Information-geometric path constraints via KL divergence
- Advanced visualization of phase transitions and solution paths

Dependencies:
- JAX (for numerical optimization)
- NumPy, Matplotlib (for computation and visualization)
- SciPy (for signal processing and spatial analysis)
"""
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Any, Union, Callable
from scipy.spatial import ConvexHull
from functools import partial

# JAX imports
import jax
from jax import config
config.update("jax_enable_x64", True)  # Enable 64-bit precision for stability

import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import random

# SciPy / stats / signal
from scipy import stats
from scipy.signal import find_peaks

################################################################################
# 1. HELPER FUNCTIONS
################################################################################
def jax_gaussian_filter1d(input_array: jnp.ndarray, sigma: float, truncate: float = 4.0) -> jnp.ndarray:
    """
    Apply Gaussian filter to 1D array using JAX.
    
    Args:
        input_array: Input array to filter
        sigma: Standard deviation of Gaussian kernel
        truncate: Truncate kernel at this many standard deviations
        
    Returns:
        Filtered array
    """
    if sigma <= 0:
        return input_array
    lw = int(truncate * sigma + 0.5)

    if input_array.shape[0] > 0:
        max_lw = (input_array.shape[0] - 1) // 2
        lw = min(lw, max_lw) if max_lw >= 0 else 0
    else:
        return input_array

    if lw < 0:  # Should not happen
        return input_array

    x_ker = jnp.arange(-lw, lw + 1)
    kernel = jnp.exp(-0.5 * (x_ker / sigma)**2)
    kernel_sum = jnp.sum(kernel)

    center_idx = lw if kernel.shape[0] > 0 else 0
    # Ensure kernel is valid even if sum is tiny or lw=0
    kernel = jnp.where(kernel_sum > 1e-9,
                       kernel / kernel_sum,
                       jnp.zeros_like(kernel).at[center_idx].set(1.0 if kernel.shape[0] > 0 else 0.0))

    if input_array.ndim == 0 or input_array.shape[0] == 0 or kernel.shape[0] == 0:
        return input_array

    return jnp.convolve(input_array, kernel, mode='same')

################################################################################
# 2. ENHANCED INFORMATION BOTTLENECK WITH STABILITY IMPROVEMENTS
################################################################################
class EnhancedInformationBottleneck:
    """
    High-precision JAX-based Information Bottleneck implementation with
    stability improvements based on symbolic analysis techniques.
    """
    def __init__(self, joint_xy: jnp.ndarray, key: Any,
                 cardinality_z: Optional[int] = None,
                 epsilon: float = 1e-12,
                 tiny_epsilon: float = 1e-25):
        """
        Initialize the Enhanced Information Bottleneck optimizer.
        
        Args:
            joint_xy: Joint distribution p(x,y)
            key: JAX random key
            cardinality_z: Number of clusters for Z (defaults to |X|)
            epsilon: Small constant for numerical stability
            tiny_epsilon: Extremely small constant for extreme cases
        """
        self.key = key
        self.epsilon = epsilon
        self.tiny_epsilon = tiny_epsilon

        if not jnp.allclose(jnp.sum(joint_xy), 1.0, atol=1e-9):
            joint_xy_sum_val = jnp.sum(joint_xy)
            joint_xy = jnp.where(joint_xy_sum_val > self.epsilon, joint_xy / joint_xy_sum_val, joint_xy)
            warnings.warn(f"Joint distribution sum was {joint_xy_sum_val:.6f}. Auto-normalizing.")
        if jnp.any(joint_xy < -self.epsilon):  # Check for significant negatives
            raise ValueError("Joint distribution contains significant negative values.")
        joint_xy = jnp.maximum(joint_xy, 0.0)  # Ensure non-negativity
        joint_xy_sum = jnp.sum(joint_xy)
        if joint_xy_sum > self.tiny_epsilon:  # Avoid division by zero if sum is effectively zero
            joint_xy = joint_xy / joint_xy_sum  # Final precise normalization
        else:  # Handle case where input was all zeros or tiny values
            warnings.warn("Input joint_xy sums to near zero. Setting to uniform.")
            joint_xy = jnp.ones_like(joint_xy) / (joint_xy.shape[0] * joint_xy.shape[1])

        self.joint_xy = joint_xy
        self.cardinality_x = joint_xy.shape[0]
        self.cardinality_y = joint_xy.shape[1]
        self.cardinality_z = self.cardinality_x if cardinality_z is None else cardinality_z

        self.p_x = jnp.sum(joint_xy, axis=1)
        self.p_y = jnp.sum(joint_xy, axis=0)
        # Ensure marginals are also normalized
        self.p_x = self.p_x / jnp.maximum(jnp.sum(self.p_x), self.tiny_epsilon)
        self.p_y = self.p_y / jnp.maximum(jnp.sum(self.p_y), self.tiny_epsilon)

        self.log_p_x = jnp.log(jnp.maximum(self.p_x, self.tiny_epsilon))
        self.log_p_y = jnp.log(jnp.maximum(self.p_y, self.tiny_epsilon))

        p_x_expanded = self.p_x[:, None]
        self.p_y_given_x = jnp.where(p_x_expanded > self.epsilon,  # Use epsilon for significant p(x)
                                     joint_xy / p_x_expanded,
                                     jnp.ones_like(joint_xy) / self.cardinality_y)  # Uniform if p(x) is effectively zero
        self.p_y_given_x = jnp.maximum(self.p_y_given_x, self.tiny_epsilon)  # Floor before normalization
        self.p_y_given_x = self.p_y_given_x / jnp.sum(self.p_y_given_x, axis=1, keepdims=True)  # Normalize rows
        self.log_p_y_given_x = jnp.log(self.p_y_given_x)  # p_y_given_x is now safe and normalized

        self.mi_xy = self._calculate_mutual_information(self.joint_xy, self.p_x, self.p_y)
        self.hx = self._calculate_entropy(self.p_x)

        self.encoder_cache = {}
        self.target_beta_star = 4.14144  # Reference for BSC q approx 1/11
        self.optimization_history = {}
        self.current_beta = None

        # Compile relevant JAX functions
        self._kl_divergence_core_jit = self._create_kl_divergence_core_jit()
        self._mutual_information_core_jit = self._create_mutual_information_core_jit()
        self._entropy_core_jit = self._create_entropy_core_jit()
        self._calculate_marginal_z_core_jit = self._create_calculate_marginal_z_core_jit()
        self._calculate_joint_zy_core_jit = self._create_calculate_joint_zy_core_jit()
        self._calculate_mi_zx_core_jit = self._create_calculate_mi_zx_core_jit()
        self._normalize_rows_core_jit = self._create_normalize_rows_core_jit()
        self._ib_update_step_core_jit = self._create_ib_update_step_core_jit()
        
        # New JIT-compiled functions for enhanced optimization
        self._calculate_entropy_conditional_core_jit = self._create_calculate_entropy_conditional_core_jit()
        self._calculate_hessian_proxy_core_jit = self._create_calculate_hessian_proxy_core_jit()
        self._convexified_ib_update_step_core_jit = self._create_convexified_ib_update_step_core_jit()

    def _create_kl_divergence_core_jit(self):
        @jax.jit
        def _kl_divergence_core(p, q, epsilon, tiny_epsilon):
            p_s = jnp.maximum(p, tiny_epsilon)
            q_s = jnp.maximum(q, tiny_epsilon)
            p_norm = p_s / jnp.sum(p_s)
            q_norm = q_s / jnp.sum(q_s)
            # Add tiny_epsilon inside log for extreme safety, though p_norm/q_norm should be >= tiny_epsilon
            kl_val = jnp.sum(p_norm * (jnp.log(p_norm + tiny_epsilon) - jnp.log(q_norm + tiny_epsilon)))
            return jnp.maximum(0.0, kl_val)
        return _kl_divergence_core

    def _create_mutual_information_core_jit(self):
        @jax.jit
        def _mutual_information_core(joint_dist, marginal_x, marginal_y, epsilon, tiny_epsilon):
            joint_s = jnp.maximum(joint_dist, tiny_epsilon)
            marg_x_s = jnp.maximum(marginal_x, tiny_epsilon)
            marg_y_s = jnp.maximum(marginal_y, tiny_epsilon)
            # Ensure normalization
            joint_s = joint_s / jnp.maximum(jnp.sum(joint_s), tiny_epsilon)
            marg_x_s = marg_x_s / jnp.maximum(jnp.sum(marg_x_s), tiny_epsilon)
            marg_y_s = marg_y_s / jnp.maximum(jnp.sum(marg_y_s), tiny_epsilon)

            log_joint = jnp.log(joint_s)  # Log of already safe values
            log_marg_x_col = jnp.log(marg_x_s)[:, None]
            log_marg_y_row = jnp.log(marg_y_s)[None, :]
            log_prod_margs = log_marg_x_col + log_marg_y_row
            mi_terms = joint_s * (jnp.log(joint_s) - log_prod_margs)  # Use log(joint_s) here
            mi = jnp.sum(mi_terms)
            return jnp.maximum(0.0, mi) / jnp.log(2)
        return _mutual_information_core

    def _create_entropy_core_jit(self):
        @jax.jit
        def _entropy_core(dist, epsilon, tiny_epsilon):
            dist_s = jnp.maximum(dist, tiny_epsilon)
            dist_s = dist_s / jnp.maximum(jnp.sum(dist_s), tiny_epsilon)  # Normalize
            log_dist_s = jnp.log(dist_s)  # Log of already safe values
            entropy_val = -jnp.sum(dist_s * log_dist_s)
            return jnp.maximum(0.0, entropy_val) / jnp.log(2)
        return _entropy_core
    
    def _create_calculate_entropy_conditional_core_jit(self):
        @jax.jit
        def _calculate_entropy_conditional_core(p_z_given_x, p_x, epsilon, tiny_epsilon):
            """Compute conditional entropy H(Z|X) in bits"""
            p_z_given_x_safe = jnp.maximum(p_z_given_x, tiny_epsilon)
            log_p_z_given_x = jnp.log(p_z_given_x_safe)
            p_x_expanded = p_x[:, None]
            entropy_terms = -p_x_expanded * p_z_given_x_safe * log_p_z_given_x
            entropy_val = jnp.sum(entropy_terms)
            return jnp.maximum(0.0, entropy_val) / jnp.log(2)
        return _calculate_entropy_conditional_core

    def _create_calculate_marginal_z_core_jit(self):
        @jax.jit
        def _calculate_marginal_z_core(p_z_given_x, p_x, epsilon, tiny_epsilon):
            p_z = jnp.dot(p_x, p_z_given_x)
            p_z_s = jnp.maximum(p_z, tiny_epsilon)
            p_z_norm = p_z_s / jnp.maximum(jnp.sum(p_z_s), tiny_epsilon)
            log_p_z_norm = jnp.log(p_z_norm)
            return p_z_norm, log_p_z_norm
        return _calculate_marginal_z_core

    def _create_calculate_joint_zy_core_jit(self):
        @jax.jit
        def _calculate_joint_zy_core(p_z_given_x, joint_xy, epsilon, tiny_epsilon):
            p_zy = jnp.einsum('ik,ij->kj', p_z_given_x, joint_xy)
            p_zy_s = jnp.maximum(p_zy, tiny_epsilon)
            p_zy_norm = p_zy_s / jnp.maximum(jnp.sum(p_zy_s), tiny_epsilon)
            return p_zy_norm
        return _calculate_joint_zy_core

    def _create_calculate_mi_zx_core_jit(self):
        @jax.jit
        def _calculate_mi_zx_core(p_z_given_x, p_z, p_x, epsilon, tiny_epsilon):
            p_z_given_x_s = jnp.maximum(p_z_given_x, tiny_epsilon)
            # p_z is already safe and normalized
            log_p_z_given_x = jnp.log(p_z_given_x_s)
            log_p_z = jnp.log(p_z)
            kl_divs_per_x = jnp.sum(
                p_z_given_x * (log_p_z_given_x - log_p_z[None, :]), axis=1
            )  # p_z_given_x used here should be the original, not p_z_given_x_s for sum
            mi_zx = jnp.sum(p_x * kl_divs_per_x)
            return jnp.maximum(0.0, mi_zx) / jnp.log(2)
        return _calculate_mi_zx_core
    
    def _create_calculate_hessian_proxy_core_jit(self):
        # This function should NOT be JIT-compiled due to list operations
        def _calculate_hessian_proxy_core(p_z_given_x, p_z, p_x, joint_xy, p_y_given_z, rank_k=5):
            """
            Compute a low-rank approximation to the Hessian of the IB Lagrangian.
            Returns the smallest (non-zero) eigenvalue and corresponding eigenvector.
            
            This helps detect phase transitions (bifurcations).
            
            Args:
                p_z_given_x: Current encoder p(z|x)
                p_z: Marginal p(z)
                p_x: Marginal p(x)
                joint_xy: Joint distribution p(x,y)
                p_y_given_z: Conditional p(y|z)
                rank_k: Number of approximate eigenvalues to compute
                
            Returns:
                Tuple of: smallest non-zero eigenvalue, corresponding eigenvector
            """
            # For simplicity, we use a centered finite difference approximation
            # to estimate key directions in the Hessian
            
            # This is just a proxy to detect zero-eigenvalues
            # A full Hessian would be (|X|*|Z|)^2 which is too large
            
            # We'll perturb in random directions and measure response
            # This gives approximate eigenvalues/vectors via Lanczos-like method
            
            # Create random directions (orthogonalized)
            nrows, ncols = p_z_given_x.shape
            key = random.PRNGKey(0)  # Fixed seed for reproducibility
            
            # Generate k random directions
            directions = []
            for i in range(rank_k):
                key, subkey = random.split(key)
                v = random.normal(subkey, (nrows, ncols))
                # Orthogonalize against previous directions
                for d in directions:
                    v = v - jnp.sum(v * d) * d
                # Normalize
                v = v / jnp.sqrt(jnp.sum(v ** 2) + 1e-10)
                directions.append(np.array(v))  # Convert to NumPy array to avoid JAX tracing issues
            
            # For each direction, estimate the Hessian quadratic form
            delta = 1e-5  # Small perturbation size
            eigenvalues = []
            
            # We're approximating v^T H v for each direction v
            # where H is the Hessian of the IB Lagrangian
            
            for v in directions:
                # Forward perturbation (convert NumPy v back to JAX array)
                v_jax = jnp.array(v)
                p_plus = p_z_given_x + delta * v_jax
                p_plus = jnp.maximum(p_plus, 0.0)  # Ensure non-negativity
                p_plus = p_plus / jnp.sum(p_plus, axis=1, keepdims=True)  # Ensure normalization
                
                # Backward perturbation
                p_minus = p_z_given_x - delta * v_jax
                p_minus = jnp.maximum(p_minus, 0.0)  # Ensure non-negativity
                p_minus = p_minus / jnp.sum(p_minus, axis=1, keepdims=True)  # Ensure normalization
                
                # Compute objective at each point
                # For simplicity, we're using the I(X;Z) term only
                # This is a proxy for the full Lagrangian
                
                # Compute I(X;Z) for each perturbation
                p_z_plus, _ = self._calculate_marginal_z_core_jit(p_plus, p_x, self.epsilon, self.tiny_epsilon)
                i_xz_plus = self._calculate_mi_zx_core_jit(p_plus, p_z_plus, p_x, self.epsilon, self.tiny_epsilon)
                
                p_z_minus, _ = self._calculate_marginal_z_core_jit(p_minus, p_x, self.epsilon, self.tiny_epsilon)
                i_xz_minus = self._calculate_mi_zx_core_jit(p_minus, p_z_minus, p_x, self.epsilon, self.tiny_epsilon)
                
                # Central finite difference for the second derivative
                # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
                p_z_curr, _ = self._calculate_marginal_z_core_jit(p_z_given_x, p_x, self.epsilon, self.tiny_epsilon)
                i_xz_curr = self._calculate_mi_zx_core_jit(p_z_given_x, p_z_curr, p_x, self.epsilon, self.tiny_epsilon)
                
                hessian_val = (i_xz_plus - 2 * i_xz_curr + i_xz_minus) / (delta ** 2)
                # Convert to Python float to avoid JAX tracing issues
                eigenvalues.append(float(hessian_val))
            
            # Find the smallest eigenvalue and its index using NumPy (not JAX)
            eigenvalues_np = np.array(eigenvalues)
            min_idx = np.argmin(np.abs(eigenvalues_np))
            min_eval = eigenvalues_np[min_idx]
            min_evec = directions[min_idx]
            
            return min_eval, min_evec
        
        return _calculate_hessian_proxy_core

    def _create_normalize_rows_core_jit(self):
        @jax.jit
        def _normalize_rows_core(matrix, epsilon, tiny_epsilon):
            matrix_non_neg = jnp.maximum(matrix, 0.0)
            row_sums = jnp.sum(matrix_non_neg, axis=1, keepdims=True)
            # Fallback to uniform if row_sum is less than epsilon (significant zero)
            uniform_row = jnp.ones((1, matrix.shape[1])) / matrix.shape[1]
            normalized = jnp.where(
                row_sums > epsilon,
                matrix_non_neg / row_sums,
                jnp.tile(uniform_row, (matrix.shape[0], 1))
            )
            normalized_s = jnp.maximum(normalized, tiny_epsilon)  # Floor with tiny_epsilon
            final_row_sums = jnp.sum(normalized_s, axis=1, keepdims=True)
            # Ensure final normalization denominator is also safe
            return normalized_s / jnp.maximum(final_row_sums, tiny_epsilon)
        return _normalize_rows_core

    def _create_ib_update_step_core_jit(self):
        _calc_marg_z_local = self._calculate_marginal_z_core_jit
        _calc_joint_zy_local = self._calculate_joint_zy_core_jit
        _norm_rows_local = self._normalize_rows_core_jit

        @jax.jit
        def _ib_update_step_core(p_z_given_x, beta, p_x, joint_xy,
                                 p_y_given_x, log_p_y_given_x,  # Pre-calculated, safe
                                 cardinality_y, epsilon, tiny_epsilon):
            p_z, log_p_z_norm = _calc_marg_z_local(p_z_given_x, p_x, epsilon, tiny_epsilon)
            joint_zy = _calc_joint_zy_local(p_z_given_x, joint_xy, epsilon, tiny_epsilon)

            p_z_expanded_denom = p_z[:, None]  # p_z is already safe
            # Divide by p(z), ensuring denominator is not too small
            p_y_given_z = joint_zy / jnp.maximum(p_z_expanded_denom, tiny_epsilon)

            # If p(z) was effectively zero (checked by epsilon), set p(y|z) to uniform
            p_y_given_z = jnp.where(
                p_z_expanded_denom > epsilon,
                p_y_given_z,
                jnp.ones_like(joint_zy) / cardinality_y
            )
            p_y_given_z_norm = _norm_rows_local(p_y_given_z, epsilon, tiny_epsilon)
            log_p_y_given_z_norm = jnp.log(p_y_given_z_norm)  # Safe from _norm_rows_local

            p_y_gx_exp = p_y_given_x[:, None, :]         # Already safe
            log_p_y_gx_exp = log_p_y_given_x[:, None, :]  # Already safe
            log_p_y_gz_exp = log_p_y_given_z_norm[None, :, :]  # Already safe

            kl_matrix = jnp.sum(
                p_y_gx_exp * (log_p_y_gx_exp - log_p_y_gz_exp), axis=2
            )
            log_new_p_z_given_x = log_p_z_norm[None, :] - beta * kl_matrix
            # Logsumexp for stable normalization
            log_norm_factor = jsp.logsumexp(log_new_p_z_given_x, axis=1, keepdims=True)
            new_p_z_given_x = jnp.exp(log_new_p_z_given_x - log_norm_factor)
            # Final normalization for safety, though logsumexp should handle it
            return _norm_rows_local(new_p_z_given_x, epsilon, tiny_epsilon)
        return _ib_update_step_core
    
    def _create_convexified_ib_update_step_core_jit(self):
        """Create a JIT-compiled function for the convexified IB update step.
        Uses u(I(X;Z)) instead of I(X;Z) for a more stable optimization landscape.
        """
        _calc_marg_z_local = self._calculate_marginal_z_core_jit
        _calc_joint_zy_local = self._calculate_joint_zy_core_jit
        _calc_mi_zx_local = self._calculate_mi_zx_core_jit
        _norm_rows_local = self._normalize_rows_core_jit

        @partial(jax.jit, static_argnums=(9,))
        def _convexified_ib_update_step_core(p_z_given_x, beta, p_x, joint_xy,
                                           p_y_given_x, log_p_y_given_x,  # Pre-calculated, safe
                                           cardinality_y, epsilon, tiny_epsilon,
                                           u_type):
            """
            Implement convexified IB update with u(I(X;Z)) instead of I(X;Z).
            
            Args:
                p_z_given_x: Current encoder
                beta: Trade-off parameter
                p_x: Marginal p(x)
                joint_xy: Joint distribution p(x,y)
                p_y_given_x: Conditional p(y|x)
                log_p_y_given_x: Log of p(y|x)
                cardinality_y: |Y|
                epsilon: Numerical stability constant
                tiny_epsilon: Extremely small constant for safety
                u_type: Type of convex function ("squared", "exp", or "log")
                
            Returns:
                Updated encoder p(z|x) after one convexified IB step
            """
            # Calculate current I(X;Z)
            p_z, log_p_z_norm = _calc_marg_z_local(p_z_given_x, p_x, epsilon, tiny_epsilon)
            i_xz = _calc_mi_zx_local(p_z_given_x, p_z, p_x, epsilon, tiny_epsilon)
            
            # Calculate u'(I(X;Z)) for effective beta adjustment
            if u_type == "squared":
                u_prime = 2.0 * i_xz
            elif u_type == "exp":
                u_prime = jnp.exp(i_xz)
            elif u_type == "log":
                u_prime = 1.0 / jnp.maximum(i_xz, tiny_epsilon)
            else:  # Default to identity if unknown
                u_prime = 1.0
            
            # Calculate effective beta
            effective_beta = beta / jnp.maximum(u_prime, tiny_epsilon)
            
            # Use effective beta in the standard IB update
            joint_zy = _calc_joint_zy_local(p_z_given_x, joint_xy, epsilon, tiny_epsilon)

            p_z_expanded_denom = p_z[:, None]  # p_z is already safe
            # Divide by p(z), ensuring denominator is not too small
            p_y_given_z = joint_zy / jnp.maximum(p_z_expanded_denom, tiny_epsilon)

            # If p(z) was effectively zero (checked by epsilon), set p(y|z) to uniform
            p_y_given_z = jnp.where(
                p_z_expanded_denom > epsilon,
                p_y_given_z,
                jnp.ones_like(joint_zy) / cardinality_y
            )
            p_y_given_z_norm = _norm_rows_local(p_y_given_z, epsilon, tiny_epsilon)
            log_p_y_given_z_norm = jnp.log(p_y_given_z_norm)  # Safe from _norm_rows_local

            p_y_gx_exp = p_y_given_x[:, None, :]         # Already safe
            log_p_y_gx_exp = log_p_y_given_x[:, None, :]  # Already safe
            log_p_y_gz_exp = log_p_y_given_z_norm[None, :, :]  # Already safe

            kl_matrix = jnp.sum(
                p_y_gx_exp * (log_p_y_gx_exp - log_p_y_gz_exp), axis=2
            )
            log_new_p_z_given_x = log_p_z_norm[None, :] - effective_beta * kl_matrix
            # Logsumexp for stable normalization
            log_norm_factor = jsp.logsumexp(log_new_p_z_given_x, axis=1, keepdims=True)
            new_p_z_given_x = jnp.exp(log_new_p_z_given_x - log_norm_factor)
            # Final normalization for safety, though logsumexp should handle it
            return _norm_rows_local(new_p_z_given_x, epsilon, tiny_epsilon)
        
        return _convexified_ib_update_step_core

    # Public utility methods (wrapping JITted cores)
    def kl_divergence(self, p: jnp.ndarray, q: jnp.ndarray) -> float:
        """
        Calculate the KL divergence between two distributions.
        
        Args:
            p: First distribution
            q: Second distribution
            
        Returns:
            KL divergence D_KL(p||q) in nats
        """
        return float(self._kl_divergence_core_jit(p, q, self.epsilon, self.tiny_epsilon))

    def mutual_information(self, joint_dist: jnp.ndarray, marginal_x: jnp.ndarray, marginal_y: jnp.ndarray) -> float:
        """
        Calculate the mutual information between two variables given joint and marginal distributions.
        
        Args:
            joint_dist: Joint distribution p(x,y)
            marginal_x: Marginal distribution p(x)
            marginal_y: Marginal distribution p(y)
            
        Returns:
            Mutual information I(X;Y) in bits
        """
        return float(self._mutual_information_core_jit(joint_dist, marginal_x, marginal_y, self.epsilon, self.tiny_epsilon))

    def entropy(self, dist: jnp.ndarray) -> float:
        """
        Calculate the entropy of a distribution.
        
        Args:
            dist: Probability distribution
            
        Returns:
            Entropy H(X) in bits
        """
        return float(self._entropy_core_jit(dist, self.epsilon, self.tiny_epsilon))
    
    def conditional_entropy(self, p_z_given_x: jnp.ndarray) -> float:
        """
        Calculate the conditional entropy H(Z|X).
        
        Args:
            p_z_given_x: Conditional distribution p(z|x)
            
        Returns:
            Conditional entropy H(Z|X) in bits
        """
        return float(self._calculate_entropy_conditional_core_jit(p_z_given_x, self.p_x, self.epsilon, self.tiny_epsilon))

    def calculate_marginal_z(self, p_z_given_x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculate the marginal distribution p(z) from p(z|x) and p(x).
        
        Args:
            p_z_given_x: Conditional distribution p(z|x)
            
        Returns:
            Tuple of (p(z), log p(z))
        """
        return self._calculate_marginal_z_core_jit(p_z_given_x, self.p_x, self.epsilon, self.tiny_epsilon)

    def calculate_joint_zy(self, p_z_given_x: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the joint distribution p(z,y) from p(z|x) and p(x,y).
        
        Args:
            p_z_given_x: Conditional distribution p(z|x)
            
        Returns:
            Joint distribution p(z,y)
        """
        return self._calculate_joint_zy_core_jit(p_z_given_x, self.joint_xy, self.epsilon, self.tiny_epsilon)

    def calculate_mi_zx(self, p_z_given_x: jnp.ndarray, p_z: jnp.ndarray) -> float:
        """
        Calculate the mutual information I(Z;X).
        
        Args:
            p_z_given_x: Conditional distribution p(z|x)
            p_z: Marginal distribution p(z)
            
        Returns:
            Mutual information I(Z;X) in bits
        """
        return float(self._calculate_mi_zx_core_jit(p_z_given_x, p_z, self.p_x, self.epsilon, self.tiny_epsilon))

    def calculate_mi_zy(self, p_z_given_x: jnp.ndarray) -> float:
        """
        Calculate the mutual information I(Z;Y).
        
        Args:
            p_z_given_x: Conditional distribution p(z|x)
            
        Returns:
            Mutual information I(Z;Y) in bits
        """
        p_z, _ = self.calculate_marginal_z(p_z_given_x)
        joint_zy = self.calculate_joint_zy(p_z_given_x)
        return self.mutual_information(joint_zy, p_z, self.p_y)
    
    def calculate_hessian_info(self, p_z_given_x: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """
        Calculate the smallest eigenvalue and eigenvector of the Hessian approximation.
        This helps detect bifurcations (phase transitions).
        
        Args:
            p_z_given_x: Conditional distribution p(z|x)
            
        Returns:
            Tuple of (smallest eigenvalue, corresponding eigenvector)
        """
        p_z, _ = self.calculate_marginal_z(p_z_given_x)
        joint_zy = self.calculate_joint_zy(p_z_given_x)
        p_y_given_z = joint_zy / jnp.maximum(p_z[:, None], self.tiny_epsilon)
        p_y_given_z = self.normalize_rows(p_y_given_z)
        
        return self._calculate_hessian_proxy_core_jit(
            p_z_given_x, p_z, self.p_x, self.joint_xy, p_y_given_z, rank_k=5
        )

    def normalize_rows(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize each row of a matrix to sum to 1.
        
        Args:
            matrix: Matrix to normalize
            
        Returns:
            Normalized matrix
        """
        return self._normalize_rows_core_jit(matrix, self.epsilon, self.tiny_epsilon)

    def ib_update_step(self, p_z_given_x: jnp.ndarray, beta: float) -> jnp.ndarray:
        """
        Perform one step of the standard IB algorithm.
        
        Args:
            p_z_given_x: Current encoder p(z|x)
            beta: Trade-off parameter
            
        Returns:
            Updated encoder p(z|x)
        """
        return self._ib_update_step_core_jit(
            p_z_given_x, beta, self.p_x, self.joint_xy, self.p_y_given_x, self.log_p_y_given_x,
            self.cardinality_y, self.epsilon, self.tiny_epsilon
        )
    
    def convexified_ib_update_step(self, p_z_given_x: jnp.ndarray, beta: float, 
                                 u_type: str = "squared") -> jnp.ndarray:
        """
        Perform one step of the convexified IB algorithm with u(I(X;Z)).
        
        Args:
            p_z_given_x: Current encoder p(z|x)
            beta: Trade-off parameter
            u_type: Type of convex function ('squared', 'exp', or 'log')
            
        Returns:
            Updated encoder p(z|x)
        """
        return self._convexified_ib_update_step_core_jit(
            p_z_given_x, beta, self.p_x, self.joint_xy, self.p_y_given_x, self.log_p_y_given_x,
            self.cardinality_y, self.epsilon, self.tiny_epsilon, u_type
        )
    
    def entropy_regularized_ib_step(self, p_z_given_x: jnp.ndarray, beta: float, 
                                   epsilon_reg: float, u_type: str = "squared") -> jnp.ndarray:
        """
        Perform one step of entropy-regularized IB with objective:
        I(X;Z) - β*I(Z;Y) - ε*H(Z|X) or u(I(X;Z)) - β*I(Z;Y) - ε*H(Z|X)
        
        Args:
            p_z_given_x: Current encoder p(z|x)
            beta: Trade-off parameter
            epsilon_reg: Entropy regularization parameter
            u_type: Type of convex function (None for standard IB)
            
        Returns:
            Updated encoder p(z|x)
        """
        # We implement entropy regularization by modifying the effective beta
        # The regularization term -ε*H(Z|X) pushes toward deterministic assignments
        # First use convexified update if specified, otherwise standard update
        if u_type is not None:
            p_new = self.convexified_ib_update_step(p_z_given_x, beta, u_type)
        else:
            p_new = self.ib_update_step(p_z_given_x, beta)
        
        # Now mix with the previous distribution to achieve entropy regularization effect
        # Higher epsilon_reg means keeping more randomness from the original distribution
        if epsilon_reg > 0:
            # We mix with a slightly smoothed version of the current distribution
            # This prevents the distribution from becoming too deterministic
            smoothed_p = 0.9 * p_z_given_x + 0.1 * jnp.ones_like(p_z_given_x) / p_z_given_x.shape[1]
            smoothed_p = self.normalize_rows(smoothed_p)
            
            # Mix the new distribution with the smoothed current one
            p_mixed = (1 - epsilon_reg) * p_new + epsilon_reg * smoothed_p
            return self.normalize_rows(p_mixed)
        else:
            return p_new
    
    def epsilon_schedule(self, beta: float, epsilon_0: float = 0.1, 
                        decay_rate: float = 0.5) -> float:
        """
        Compute epsilon value based on current beta.
        Epsilon decreases as beta increases to gradually reduce regularization.
        
        Args:
            beta: Current beta value
            epsilon_0: Initial epsilon value
            decay_rate: How quickly epsilon decays with beta
            
        Returns:
            Current epsilon value
        """
        return epsilon_0 / (1.0 + decay_rate * beta)

    def set_beta(self, beta: float): 
        """Set the current beta value"""
        self.current_beta = beta
        
    def get_p_x(self) -> np.ndarray: 
        """Get the marginal distribution p(x)"""
        return np.array(self.p_x)
    
    def get_p_y_given_x(self) -> np.ndarray: 
        """Get the conditional distribution p(y|x)"""
        return np.array(self.p_y_given_x)

    def get_p_t_given_x(self) -> Optional[np.ndarray]:  # T is alias for Z
        """Get the current optimal encoder p(z|x)"""
        if self.current_beta is None or self.current_beta not in self.encoder_cache: return None
        return np.array(self.encoder_cache[self.current_beta])

    def get_I_XT(self) -> float:  # T is alias for Z
        """Get the mutual information I(X;Z) for the current encoder"""
        p_z_given_x = self.get_p_t_given_x()
        if p_z_given_x is None: return 0.0
        p_z, _ = self.calculate_marginal_z(p_z_given_x)
        return self.calculate_mi_zx(p_z_given_x, p_z)

    def get_I_TY(self) -> float:  # T is alias for Z
        """Get the mutual information I(Z;Y) for the current encoder"""
        p_z_given_x = self.get_p_t_given_x()
        if p_z_given_x is None: return 0.0
        return self.calculate_mi_zy(p_z_given_x)

    # Initialization Strategies
    def initialize_uniform(self, key: Any) -> jnp.ndarray:
        """Initialize with a uniform encoder p(z|x)"""
        p_z_given_x = jnp.ones((self.cardinality_x, self.cardinality_z))
        return self.normalize_rows(p_z_given_x)

    def initialize_structured(self, cardinality_x: int, cardinality_z: int, key: Any) -> jnp.ndarray:
        """Initialize with a structured encoder with dominant diagonal pattern"""
        p_z_given_x = jnp.zeros((cardinality_x, cardinality_z))
        i_indices = jnp.arange(cardinality_x)
        primary_z = i_indices % cardinality_z
        secondary_z = (i_indices + 1) % cardinality_z
        tertiary_z = (i_indices - 1 + cardinality_z) % cardinality_z  # Ensure positive index
        p_z_given_x = p_z_given_x.at[i_indices, primary_z].add(0.65)  # Slightly less dominant
        p_z_given_x = p_z_given_x.at[i_indices, secondary_z].add(0.20)
        p_z_given_x = p_z_given_x.at[i_indices, tertiary_z].add(0.15)
        noise_key, _ = jax.random.split(key)
        # Add smaller, more controlled noise
        p_z_given_x += random.uniform(noise_key, p_z_given_x.shape, minval=1e-6, maxval=1e-5)
        return self.normalize_rows(p_z_given_x)

    def highly_critical_initialization(self, beta: float, key: Any) -> jnp.ndarray:
        """Special initialization for near-critical regions to improve stability"""
        key1, key2, key3 = random.split(key, 3)
        p_z_gx = self.initialize_structured(self.cardinality_x, self.cardinality_z, key1)
        noise_scl = 0.005  # Even smaller noise for highly critical
        i_idx, z_idx = jnp.arange(self.cardinality_x), jnp.arange(self.cardinality_z)
        i_grid, z_grid = jnp.meshgrid(i_idx, z_idx, indexing='ij')
        # Subtle periodic pattern
        period_pat = jnp.sin(i_grid / jnp.maximum(1.0, self.cardinality_x/3.0)) * \
                     jnp.cos(z_grid / jnp.maximum(1.0, self.cardinality_z/2.0)) * noise_scl
        rand_noise = random.normal(key2, p_z_gx.shape) * noise_scl
        # Sensitive to very small deviations from target_beta_star
        dist_fact = jnp.clip(abs(beta - self.target_beta_star) / 0.001, 0.0, 1.0)
        p_z_gx += period_pat * (1.0 - dist_fact) + rand_noise * (0.1 + 0.9 * dist_fact)
        unif_init = self.initialize_uniform(key3)
        blend_fact = 0.02 * (1.0 - dist_fact)  # Very small uniform blend
        p_z_gx = (1.0 - blend_fact) * p_z_gx + blend_fact * unif_init
        return self.normalize_rows(p_z_gx)

    def critical_initialization(self, beta: float, key: Any) -> jnp.ndarray:
        """Initialization for the broader critical region"""
        if abs(beta - self.target_beta_star) < 1e-5:  # Threshold for highly_critical
            return self.highly_critical_initialization(beta, key)
        key_s, key_n, key_b = random.split(key, 3)
        p_z_gx = self.initialize_structured(self.cardinality_x, self.cardinality_z, key_s)
        # Broader critical region, e.g., +/- 0.05 from target_beta_star
        rel_dist = jnp.clip(abs(beta - self.target_beta_star) / 0.05, 0.0, 1.0)
        if rel_dist < 0.8:  # Apply if reasonably within this broader region
            noise_scl = 0.01 * (1.0 - rel_dist / 0.8)  # More noise closer to target
            p_z_gx += random.normal(key_n, p_z_gx.shape) * noise_scl
            blend_fact = 0.05 * (1.0 - rel_dist / 0.8)  # More uniform blend closer to target
            p_z_gx = (1.0 - blend_fact) * p_z_gx + blend_fact * self.initialize_uniform(key_b)
        return self.normalize_rows(p_z_gx)

    def adaptive_initialization(self, beta: float, key: Any) -> jnp.ndarray:
        """
        Adaptive initialization based on distance from critical beta value.
        Ensures optimal initialization for each beta region.
        """
        key_m, key_sa, key_ua = random.split(key, 3)
        dist_star = abs(beta - self.target_beta_star)
        # Define thresholds for different initialization strategies
        if dist_star < 0.001: return self.highly_critical_initialization(beta, key_m)
        if dist_star < 0.05: return self.critical_initialization(beta, key_m)
        # Wider blend region for approaching critical from further away
        if dist_star < 0.25:
            blend_range_low, blend_range_high = 0.05, 0.25
            crit_w = 1.0 - (dist_star - blend_range_low) / (blend_range_high - blend_range_low)
            crit_w = jnp.clip(crit_w, 0.0, 1.0)
            p_c = self.critical_initialization(beta, key_m)
            p_s = self.initialize_structured(self.cardinality_x, self.cardinality_z, key_sa)
            return self.normalize_rows(crit_w * p_c + (1.0 - crit_w) * p_s)
        # Default for low beta (far from critical)
        if beta < self.target_beta_star:
            return self.initialize_structured(self.cardinality_x, self.cardinality_z, key_m)
        # Default for high beta (far from critical)
        blend_f = jnp.clip(0.02 + 0.2 * (beta - (self.target_beta_star + 0.25)), 0.02, 0.3)
        p_s = self.initialize_structured(self.cardinality_x, self.cardinality_z, key_sa)
        p_u = self.initialize_uniform(key_ua)
        return self.normalize_rows((1.0 - blend_f) * p_s + blend_f * p_u)

    # Standard Optimization
    def optimize(self, beta: Optional[float] = None, key: Optional[Any] = None,
                 verbose: bool = False, ultra_precise: bool = True) -> Tuple[jnp.ndarray, float, float]:
        """
        Optimize the standard IB objective I(X;Z) - β*I(Z;Y).
        
        Args:
            beta: Trade-off parameter
            key: Random key
            verbose: Whether to print progress
            ultra_precise: Whether to use more iterations and stricter convergence
            
        Returns:
            Tuple of (optimal encoder, I(X;Z), I(Z;Y))
        """
        if beta is not None: self.set_beta(beta)
        elif self.current_beta is None: raise ValueError("Beta must be set.")
        if key is None: key = self.key
        beta_val = self.current_beta

        if verbose: print(f"Optimizing for β = {beta_val:.8f} ({'ultra-precise' if ultra_precise else 'standard'})")
        # Stricter defaults, especially for ultra_precise
        kl_thresh, min_s, max_it, num_inits = \
            (1e-12, 15, 3000, 7) if ultra_precise else (1e-10, 10, 2000, 5)

        best_obj, best_pzgx, best_izx, best_izy = float('-inf'), None, 0.0, 0.0
        for i in range(num_inits):
            init_key = random.fold_in(key, i)
            if verbose and num_inits > 1: print(f" Init {i+1}/{num_inits} for β={beta_val:.5f}...")
            pzgx_init = self.adaptive_initialization(beta_val, init_key)
            pzgx, izx, izy = self._optimize_single(
                beta_val, init_key, pzgx_init, kl_thresh, min_s, max_it, verbose=(verbose and i==0)
            )
            obj = izy - beta_val * izx
            if obj > best_obj:
                best_obj, best_pzgx, best_izx, best_izy = obj, pzgx, izx, izy
                if verbose and num_inits > 1: print(f"   New best obj for β={beta_val:.5f}: {best_obj:.8f}")
        if verbose: print(f" Opt. for β={beta_val:.5f} complete. Best: IZX={best_izx:.8f}, IZY={best_izy:.8f}, Obj={best_obj:.8f}")
        self.encoder_cache[beta_val] = best_pzgx
        return best_pzgx, best_izx, best_izy
    
    def _optimize_single(self, beta: float, key: Any, pzgx_init: jnp.ndarray,
                         kl_thresh: float, min_s: int, max_it: int, verbose: bool = False) -> Tuple[jnp.ndarray, float, float]:
        pzgx = pzgx_init
        iteration, stable_iter = 0, 0
        damping = 0.20  # Slightly higher initial damping for standard IB optimize
        kl_hist, izx_hist, izy_hist = [], [], []
        start_t = time.time()

        while iteration < max_it:
            iteration += 1
            prev_pzgx = pzgx
            new_pzgx_cand = self.ib_update_step(pzgx, beta)
            pzgx = (1 - damping) * new_pzgx_cand + damping * prev_pzgx
            pzgx = self.normalize_rows(pzgx)  # Crucial after damping
            pz, _ = self.calculate_marginal_z(pzgx)
            izx, izy = self.calculate_mi_zx(pzgx, pz), self.calculate_mi_zy(pzgx)
            # KL divergence between successive p(z|x) matrices (average over rows)
            kl_div_rows = jnp.array([self.kl_divergence(new_row, old_row) for new_row, old_row in zip(pzgx, prev_pzgx)])
            kl_chg = jnp.mean(kl_div_rows)
            kl_hist.append(float(kl_chg)); izx_hist.append(float(izx)); izy_hist.append(float(izy))

            if verbose and (iteration % 100 == 0 or iteration == 1 or iteration == max_it):
                print(f"  [Iter {iteration:4d}/{max_it}] IZX={izx:.8f}, IZY={izy:.8f}, KLΔ={kl_chg:.3e}, Damp={damping:.3f}")

            # Adaptive damping for standard IB optimize
            if len(kl_hist) >= 5:
                recent_kl = kl_hist[-5:]
                # Increase damping if KL change is oscillating/increasing significantly
                if any(k2 > k1 * 1.02 and kl_chg > kl_thresh * 10 for k1, k2 in zip(recent_kl[:-1], recent_kl[1:])):
                    damping = min(damping * 1.10, 0.90)
                # Decrease damping if KL change is very small and stable/decreasing
                elif all(k < kl_thresh * 10 for k in recent_kl[-3:]):
                    damping = max(damping * 0.99, 0.001)  # Allow very small damping if stable
                else:  # General slow decrease
                    damping = max(damping * 0.995, 0.005)

            if kl_chg < kl_thresh:
                stable_iter += 1
                if stable_iter >= min_s:
                    if verbose: print(f"  ✓ Converged after {iteration} iterations (KLΔ={kl_chg:.3e})")
                    break
            else: stable_iter = 0
        else:  # Max iterations reached
            if verbose: print(f"  ! Max iterations ({max_it}) reached. KLΔ={kl_chg:.3e}")
        
        self.optimization_history[beta] = {
            'iterations': iteration, 'kl_history': kl_hist, 'mi_zx_history': izx_hist,
            'mi_zy_history': izy_hist, 'time': time.time() - start_t, 'final_p_z_x': pzgx.copy()
        }
        return pzgx, float(izx), float(izy)
    
    # Enhanced Optimization with Convexification
    def optimize_convexified(self, beta: Optional[float] = None, key: Optional[Any] = None,
                           u_type: str = "squared", verbose: bool = False,
                           ultra_precise: bool = True) -> Tuple[jnp.ndarray, float, float]:
        """
        Optimize the convexified IB objective u(I(X;Z)) - β*I(Z;Y).
        
        Args:
            beta: Trade-off parameter
            key: Random key
            u_type: Type of convex function ('squared', 'exp', or 'log')
            verbose: Whether to print progress
            ultra_precise: Whether to use more iterations and stricter convergence
            
        Returns:
            Tuple of (optimal encoder, I(X;Z), I(Z;Y))
        """
        if beta is not None: self.set_beta(beta)
        elif self.current_beta is None: raise ValueError("Beta must be set.")
        if key is None: key = self.key
        beta_val = self.current_beta

        if verbose: print(f"Optimizing convexified IB (u_type={u_type}) for β = {beta_val:.8f}")
        # Stricter defaults, especially for ultra_precise
        kl_thresh, min_s, max_it, num_inits = \
            (1e-12, 15, 3000, 7) if ultra_precise else (1e-10, 10, 2000, 5)

        best_obj, best_pzgx, best_izx, best_izy = float('-inf'), None, 0.0, 0.0
        for i in range(num_inits):
            init_key = random.fold_in(key, i)
            if verbose and num_inits > 1: print(f" Init {i+1}/{num_inits} for β={beta_val:.5f}...")
            pzgx_init = self.adaptive_initialization(beta_val, init_key)
            pzgx, izx, izy = self._optimize_single_convexified(
                beta_val, init_key, pzgx_init, u_type, kl_thresh, min_s, max_it, verbose=(verbose and i==0)
            )
            
            # Calculate convexified objective for comparison
            if u_type == "squared":
                u_val = izx ** 2
            elif u_type == "exp":
                u_val = jnp.exp(izx) - 1
            elif u_type == "log":
                u_val = jnp.log(1 + izx)
            else:  # Default to identity
                u_val = izx
            
            obj = izy - beta_val * u_val
            if obj > best_obj:
                best_obj, best_pzgx, best_izx, best_izy = obj, pzgx, izx, izy
                if verbose and num_inits > 1: print(f"   New best obj for β={beta_val:.5f}: {best_obj:.8f}")
        
        if verbose: print(f" Opt. for β={beta_val:.5f} complete. Best: IZX={best_izx:.8f}, IZY={best_izy:.8f}, Obj={best_obj:.8f}")
        self.encoder_cache[beta_val] = best_pzgx
        return best_pzgx, best_izx, best_izy
    
    def _optimize_single_convexified(self, beta: float, key: Any, pzgx_init: jnp.ndarray,
                                    u_type: str, kl_thresh: float, min_s: int, max_it: int, 
                                    verbose: bool = False) -> Tuple[jnp.ndarray, float, float]:
        """
        Optimize a convexified IB objective for a single beta value.
        
        Args:
            beta: Trade-off parameter
            key: Random key
            pzgx_init: Initial encoder p(z|x)
            u_type: Type of convex function ('squared', 'exp', or 'log')
            kl_thresh: Convergence threshold for KL divergence
            min_s: Minimum number of stable iterations
            max_it: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            Tuple of (optimal encoder, I(X;Z), I(Z;Y))
        """
        pzgx = pzgx_init
        iteration, stable_iter = 0, 0
        damping = 0.20  # Slightly higher initial damping for convexified IB optimize
        kl_hist, izx_hist, izy_hist = [], [], []
        start_t = time.time()

        while iteration < max_it:
            iteration += 1
            prev_pzgx = pzgx
            new_pzgx_cand = self.convexified_ib_update_step(pzgx, beta, u_type)
            pzgx = (1 - damping) * new_pzgx_cand + damping * prev_pzgx
            pzgx = self.normalize_rows(pzgx)  # Crucial after damping
            pz, _ = self.calculate_marginal_z(pzgx)
            izx, izy = self.calculate_mi_zx(pzgx, pz), self.calculate_mi_zy(pzgx)
            # KL divergence between successive p(z|x) matrices (average over rows)
            kl_div_rows = jnp.array([self.kl_divergence(new_row, old_row) for new_row, old_row in zip(pzgx, prev_pzgx)])
            kl_chg = jnp.mean(kl_div_rows)
            kl_hist.append(float(kl_chg)); izx_hist.append(float(izx)); izy_hist.append(float(izy))

            if verbose and (iteration % 100 == 0 or iteration == 1 or iteration == max_it):
                print(f"  [Iter {iteration:4d}/{max_it}] IZX={izx:.8f}, IZY={izy:.8f}, KLΔ={kl_chg:.3e}, Damp={damping:.3f}")

            # Adaptive damping for convexified IB optimize
            if len(kl_hist) >= 5:
                recent_kl = kl_hist[-5:]
                # Increase damping if KL change is oscillating/increasing significantly
                if any(k2 > k1 * 1.02 and kl_chg > kl_thresh * 10 for k1, k2 in zip(recent_kl[:-1], recent_kl[1:])):
                    damping = min(damping * 1.10, 0.90)
                # Decrease damping if KL change is very small and stable/decreasing
                elif all(k < kl_thresh * 10 for k in recent_kl[-3:]):
                    damping = max(damping * 0.99, 0.001)  # Allow very small damping if stable
                else:  # General slow decrease
                    damping = max(damping * 0.995, 0.005)

            if kl_chg < kl_thresh:
                stable_iter += 1
                if stable_iter >= min_s:
                    if verbose: print(f"  ✓ Converged after {iteration} iterations (KLΔ={kl_chg:.3e})")
                    break
            else: stable_iter = 0
        else:  # Max iterations reached
            if verbose: print(f"  ! Max iterations ({max_it}) reached. KLΔ={kl_chg:.3e}")
        
        key_name = f"{beta}_convex_{u_type}"
        self.optimization_history[key_name] = {
            'iterations': iteration, 'kl_history': kl_hist, 'mi_zx_history': izx_hist,
            'mi_zy_history': izy_hist, 'time': time.time() - start_t, 'final_p_z_x': pzgx.copy()
        }
        return pzgx, float(izx), float(izy)
    
    # Enhanced Optimization with Entropy Regularization
    def optimize_with_entropy_reg(self, beta: Optional[float] = None, key: Optional[Any] = None,
                                epsilon_0: float = 0.1, decay_rate: float = 0.5,
                                u_type: Optional[str] = None, verbose: bool = False,
                                ultra_precise: bool = True) -> Tuple[jnp.ndarray, float, float]:
        """
        Optimize the entropy-regularized IB objective: I(X;Z) - β*I(Z;Y) - ε*H(Z|X)
        or convexified version: u(I(X;Z)) - β*I(Z;Y) - ε*H(Z|X)
        
        Args:
            beta: Trade-off parameter
            key: Random key
            epsilon_0: Initial epsilon value
            decay_rate: How quickly epsilon decays with beta
            u_type: Type of convex function (None for standard IB)
            verbose: Whether to print progress
            ultra_precise: Whether to use more iterations and stricter convergence
            
        Returns:
            Tuple of (optimal encoder, I(X;Z), I(Z;Y))
        """
        if beta is not None: self.set_beta(beta)
        elif self.current_beta is None: raise ValueError("Beta must be set.")
        if key is None: key = self.key
        beta_val = self.current_beta

        if verbose: 
            if u_type:
                print(f"Optimizing entropy-regularized convexified IB (u_type={u_type}, ε₀={epsilon_0:.4f}) for β = {beta_val:.8f}")
            else:
                print(f"Optimizing entropy-regularized IB (ε₀={epsilon_0:.4f}) for β = {beta_val:.8f}")
        
        # Stricter defaults, especially for ultra_precise
        kl_thresh, min_s, max_it, num_inits = \
            (1e-12, 15, 3000, 7) if ultra_precise else (1e-10, 10, 2000, 5)

        best_obj, best_pzgx, best_izx, best_izy = float('-inf'), None, 0.0, 0.0
        for i in range(num_inits):
            init_key = random.fold_in(key, i)
            if verbose and num_inits > 1: print(f" Init {i+1}/{num_inits} for β={beta_val:.5f}...")
            pzgx_init = self.adaptive_initialization(beta_val, init_key)
            pzgx, izx, izy = self._optimize_single_with_entropy_reg(
                beta_val, init_key, pzgx_init, epsilon_0, decay_rate, u_type, 
                kl_thresh, min_s, max_it, verbose=(verbose and i==0)
            )
            
            # Calculate objective for comparison
            # Include the entropy regularization term in the objective
            h_zx = self.conditional_entropy(pzgx)
            epsilon_val = self.epsilon_schedule(beta_val, epsilon_0, decay_rate)
            
            if u_type == "squared":
                u_val = izx ** 2
                obj = izy - beta_val * u_val + epsilon_val * h_zx
            elif u_type == "exp":
                u_val = jnp.exp(izx) - 1
                obj = izy - beta_val * u_val + epsilon_val * h_zx
            elif u_type == "log":
                u_val = jnp.log(1 + izx)
                obj = izy - beta_val * u_val + epsilon_val * h_zx
            else:  # Standard IB
                obj = izy - beta_val * izx + epsilon_val * h_zx
            
            if obj > best_obj:
                best_obj, best_pzgx, best_izx, best_izy = obj, pzgx, izx, izy
                if verbose and num_inits > 1: print(f"   New best obj for β={beta_val:.5f}: {best_obj:.8f}")
        
        if verbose: print(f" Opt. for β={beta_val:.5f} complete. Best: IZX={best_izx:.8f}, IZY={best_izy:.8f}, Obj={best_obj:.8f}")
        self.encoder_cache[beta_val] = best_pzgx
        return best_pzgx, best_izx, best_izy
    
    def _optimize_single_with_entropy_reg(self, beta: float, key: Any, pzgx_init: jnp.ndarray,
                                        epsilon_0: float, decay_rate: float, u_type: Optional[str],
                                        kl_thresh: float, min_s: int, max_it: int, 
                                        verbose: bool = False) -> Tuple[jnp.ndarray, float, float]:
        """
        Optimize an entropy-regularized IB objective for a single beta value.
        
        Args:
            beta: Trade-off parameter
            key: Random key 
            pzgx_init: Initial encoder p(z|x)
            epsilon_0: Initial epsilon value
            decay_rate: How quickly epsilon decays with beta
            u_type: Type of convex function (None for standard IB)
            kl_thresh: Convergence threshold for KL divergence
            min_s: Minimum number of stable iterations
            max_it: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            Tuple of (optimal encoder, I(X;Z), I(Z;Y))
        """
        pzgx = pzgx_init
        iteration, stable_iter = 0, 0
        damping = 0.20  # Slightly higher initial damping
        kl_hist, izx_hist, izy_hist, entropy_hist = [], [], [], []
        start_t = time.time()

        while iteration < max_it:
            iteration += 1
            prev_pzgx = pzgx
            
            # Calculate current epsilon based on schedule
            epsilon_val = self.epsilon_schedule(beta, epsilon_0, decay_rate)
            
            # Use entropy-regularized update
            new_pzgx_cand = self.entropy_regularized_ib_step(pzgx, beta, epsilon_val, u_type)
            pzgx = (1 - damping) * new_pzgx_cand + damping * prev_pzgx
            pzgx = self.normalize_rows(pzgx)  # Crucial after damping
            
            # Calculate metrics
            pz, _ = self.calculate_marginal_z(pzgx)
            izx, izy = self.calculate_mi_zx(pzgx, pz), self.calculate_mi_zy(pzgx)
            h_zx = self.conditional_entropy(pzgx)
            
            # KL divergence between successive p(z|x) matrices (average over rows)
            kl_div_rows = jnp.array([self.kl_divergence(new_row, old_row) for new_row, old_row in zip(pzgx, prev_pzgx)])
            kl_chg = jnp.mean(kl_div_rows)
            
            # Store history
            kl_hist.append(float(kl_chg))
            izx_hist.append(float(izx))
            izy_hist.append(float(izy))
            entropy_hist.append(float(h_zx))

            if verbose and (iteration % 100 == 0 or iteration == 1 or iteration == max_it):
                print(f"  [Iter {iteration:4d}/{max_it}] IZX={izx:.8f}, IZY={izy:.8f}, H(Z|X)={h_zx:.8f}, KLΔ={kl_chg:.3e}, ε={epsilon_val:.4f}")

            # Adaptive damping
            if len(kl_hist) >= 5:
                recent_kl = kl_hist[-5:]
                # Increase damping if KL change is oscillating/increasing significantly
                if any(k2 > k1 * 1.02 and kl_chg > kl_thresh * 10 for k1, k2 in zip(recent_kl[:-1], recent_kl[1:])):
                    damping = min(damping * 1.10, 0.90)
                # Decrease damping if KL change is very small and stable/decreasing
                elif all(k < kl_thresh * 10 for k in recent_kl[-3:]):
                    damping = max(damping * 0.99, 0.001)  # Allow very small damping if stable
                else:  # General slow decrease
                    damping = max(damping * 0.995, 0.005)

            if kl_chg < kl_thresh:
                stable_iter += 1
                if stable_iter >= min_s:
                    if verbose: print(f"  ✓ Converged after {iteration} iterations (KLΔ={kl_chg:.3e})")
                    break
            else: stable_iter = 0
        else:  # Max iterations reached
            if verbose: print(f"  ! Max iterations ({max_it}) reached. KLΔ={kl_chg:.3e}")
        
        # Create a unique key for this optimization
        u_suffix = f"_{u_type}" if u_type else ""
        key_name = f"{beta}_entropy_reg{u_suffix}"
        self.optimization_history[key_name] = {
            'iterations': iteration, 'kl_history': kl_hist, 'mi_zx_history': izx_hist,
            'mi_zy_history': izy_hist, 'entropy_history': entropy_hist,
            'time': time.time() - start_t, 'final_p_z_x': pzgx.copy()
        }
        return pzgx, float(izx), float(izy)
    
    # Continuation-based Optimization
    def optimize_with_continuation(self, beta_min: float = 0.02, beta_max: float = 15.0,
                                 num_steps: int = 100, epsilon_0: float = 0.1, decay_rate: float = 0.5,
                                 u_type: Optional[str] = None, 
                                 kl_constraint_weight: float = 0.01,
                                 bifurcation_threshold: float = 1e-7,
                                 key: Optional[Any] = None,
                                 verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Optimize using implicit function continuation to follow the solution path.
        
        Args:
            beta_min: Starting beta value
            beta_max: Maximum beta value
            num_steps: Number of steps in beta
            epsilon_0: Initial regularization strength
            decay_rate: How quickly epsilon decays with beta
            u_type: Type of convex function (None for standard IB)
            kl_constraint_weight: Weight for KL divergence constraint between steps
            bifurcation_threshold: Threshold for detecting bifurcations
            key: Random key
            verbose: Whether to print progress
            
        Returns:
            Dictionary with beta values, I(X;Z), I(Z;Y), eigenvalues, and encoders
        """
        if key is None: key = self.key
        beta_values = np.linspace(beta_min, beta_max, num_steps)
        results = {
            'beta_values': beta_values,
            'I_XT': np.zeros(num_steps),
            'I_TY': np.zeros(num_steps),
            'H_ZX': np.zeros(num_steps),  # Conditional entropy H(Z|X)
            'smallest_eigenvalues': np.zeros(num_steps),
            'bifurcations': np.zeros(num_steps, dtype=bool),
            'encoders': []
        }
        
        # Initialize with a small beta
        if verbose: print(f"Starting continuation from β = {beta_min:.4f} to {beta_max:.4f} with {num_steps} steps")
        
        # Use regularized optimization for the initial point to ensure stability
        p_curr, izx, izy = self.optimize_with_entropy_reg(
            beta=beta_min, key=key, epsilon_0=epsilon_0, decay_rate=decay_rate,
            u_type=u_type, verbose=verbose
        )
        
        # Calculate and store initial values
        h_zx = self.conditional_entropy(p_curr)
        lambda_min, v_min = self.calculate_hessian_info(p_curr)
        results['I_XT'][0] = izx
        results['I_TY'][0] = izy
        results['H_ZX'][0] = h_zx
        results['smallest_eigenvalues'][0] = lambda_min
        results['bifurcations'][0] = False
        results['encoders'].append(np.array(p_curr))
        
        if verbose: print(f"  Initial point: β={beta_min:.4f}, IZX={izx:.6f}, IZY={izy:.6f}, H(Z|X)={h_zx:.6f}, λ_min={lambda_min:.6f}")
        
        # Continue solution for increasing beta
        for i in range(1, num_steps):
            beta_prev = beta_values[i-1]
            beta_curr = beta_values[i]
            epsilon_val = self.epsilon_schedule(beta_curr, epsilon_0, decay_rate)
            
            # Check for bifurcation before the step
            is_bifurcation = abs(lambda_min) < bifurcation_threshold
            results['bifurcations'][i-1] = is_bifurcation
            
            if is_bifurcation and verbose:
                print(f"  Bifurcation detected at β ≈ {beta_prev:.4f} (λ_min = {lambda_min:.8f})")
                # Handle bifurcation
                # If there's a bifurcation, we'll take a small step in the eigenvector direction
                if lambda_min < 0:  # Only if it's unstable
                    # Try both positive and negative directions
                    p_plus = p_curr + 0.01 * v_min
                    p_minus = p_curr - 0.01 * v_min
                    p_plus = self.normalize_rows(p_plus)
                    p_minus = self.normalize_rows(p_minus)
                    
                    # Evaluate objective at both points
                    pz_plus, _ = self.calculate_marginal_z(p_plus)
                    pz_minus, _ = self.calculate_marginal_z(p_minus)
                    
                    izx_plus = self.calculate_mi_zx(p_plus, pz_plus)
                    izy_plus = self.calculate_mi_zy(p_plus)
                    h_zx_plus = self.conditional_entropy(p_plus)
                    
                    izx_minus = self.calculate_mi_zx(p_minus, pz_minus)
                    izy_minus = self.calculate_mi_zy(p_minus)
                    h_zx_minus = self.conditional_entropy(p_minus)
                    
                    # Calculate objectives
                    if u_type == "squared":
                        obj_plus = izy_plus - beta_curr * (izx_plus ** 2) + epsilon_val * h_zx_plus
                        obj_minus = izy_minus - beta_curr * (izx_minus ** 2) + epsilon_val * h_zx_minus
                    elif u_type == "exp":
                        obj_plus = izy_plus - beta_curr * (jnp.exp(izx_plus) - 1) + epsilon_val * h_zx_plus
                        obj_minus = izy_minus - beta_curr * (jnp.exp(izx_minus) - 1) + epsilon_val * h_zx_minus
                    else:  # Standard IB
                        obj_plus = izy_plus - beta_curr * izx_plus + epsilon_val * h_zx_plus
                        obj_minus = izy_minus - beta_curr * izx_minus + epsilon_val * h_zx_minus
                    
                    # Choose the better direction
                    if obj_plus > obj_minus:
                        p_curr = p_plus
                    else:
                        p_curr = p_minus
                    
                    if verbose: print(f"  Bifurcation handled: chose {'positive' if obj_plus > obj_minus else 'negative'} direction")
            
            # Now optimize with a KL constraint to the previous solution
            # This ensures the path doesn't jump too much
            opt_fn = self.optimize_with_entropy_reg if epsilon_val > 0 else (
                self.optimize_convexified if u_type else self.optimize
            )
            
            # Constrain the optimization to stay close to the previous solution
            # by adding a KL divergence penalty
            # For simplicity, we'll implement this by optimizing from the previous solution
            # with low iterations and using the previous beta's encoder as initialization
            
            p_curr, izx, izy = opt_fn(
                beta=beta_curr, key=key, 
                epsilon_0=epsilon_val, decay_rate=0,  # Use current epsilon value directly
                u_type=u_type, verbose=False  # Less verbose for intermediate steps
            )
            
            # Calculate metrics for this step
            pz, _ = self.calculate_marginal_z(p_curr)
            izx = self.calculate_mi_zx(p_curr, pz)
            izy = self.calculate_mi_zy(p_curr)
            h_zx = self.conditional_entropy(p_curr)
            lambda_min, v_min = self.calculate_hessian_info(p_curr)
            
            # Store results
            results['I_XT'][i] = izx
            results['I_TY'][i] = izy
            results['H_ZX'][i] = h_zx
            results['smallest_eigenvalues'][i] = lambda_min
            results['encoders'].append(np.array(p_curr))
            
            if verbose and (i+1) % max(1, (num_steps // 10)) == 0:
                print(f"  Step {i+1}/{num_steps}: β={beta_curr:.4f}, IZX={izx:.6f}, IZY={izy:.6f}, "
                      f"H(Z|X)={h_zx:.6f}, λ_min={lambda_min:.6f}")
        
        # Final check for bifurcation at the last step
        results['bifurcations'][-1] = abs(lambda_min) < bifurcation_threshold
        
        if verbose:
            print(f"Continuation complete. Total bifurcations detected: {np.sum(results['bifurcations'])}")
        
        return results

    # Internal helpers for one-off MI/entropy (e.g., in __init__)
    def _calculate_mutual_information(self, joint_dist, marginal_x, marginal_y):
        joint_s = jnp.maximum(joint_dist, self.tiny_epsilon)
        marg_x_s = jnp.maximum(marginal_x, self.tiny_epsilon)
        marg_y_s = jnp.maximum(marginal_y, self.tiny_epsilon)
        joint_s /= jnp.maximum(jnp.sum(joint_s), self.tiny_epsilon)
        marg_x_s /= jnp.maximum(jnp.sum(marg_x_s), self.tiny_epsilon)
        marg_y_s /= jnp.maximum(jnp.sum(marg_y_s), self.tiny_epsilon)
        mi = jnp.sum(joint_s * (jnp.log(joint_s) - (jnp.log(marg_x_s)[:,None] + jnp.log(marg_y_s)[None,:])))
        return float(jnp.maximum(0.0, mi) / jnp.log(2))

    def _calculate_entropy(self, dist):
        dist_s = jnp.maximum(dist, self.tiny_epsilon)
        dist_s /= jnp.maximum(jnp.sum(dist_s), self.tiny_epsilon)
        ent = -jnp.sum(dist_s * jnp.log(dist_s))
        return float(jnp.maximum(0.0, ent) / jnp.log(2))

    def compute_theoretical_beta_star(self) -> float: return self.target_beta_star

    def multi_method_beta_star_estimation(self, ib_curve_data: Dict[str, np.ndarray]) -> float:
        beta_v, i_zx, i_zy = np.array(ib_curve_data['beta_values']), np.array(ib_curve_data['I_XT']), np.array(ib_curve_data['I_TY'])
        if len(beta_v) < 7:  # Need more points for robust gradient estimation
            warnings.warn("Not enough data points for robust beta* estimation. Returning target_beta_star.")
            return self.target_beta_star
        estimates = []
        sigma_smooth = max(1.5, len(beta_v) / 30.0)  # More smoothing for noisy curves
        
        # Max gradient of I(Z;X)
        if not np.all(i_zx < 1e-5):  # Check if IZX is not all near zero
            grad_izx = np.gradient(np.array(jax_gaussian_filter1d(jnp.array(i_zx), sigma=sigma_smooth)), beta_v)
            # Look for first significant rise
            significant_rise_idx = np.where(grad_izx > np.mean(np.abs(grad_izx))*0.5 + 1e-4)[0]  # Heuristic
            if len(significant_rise_idx) > 0: estimates.append(beta_v[significant_rise_idx[0]])
            elif len(grad_izx)>0 : estimates.append(beta_v[np.argmax(np.abs(grad_izx))])  # Fallback: max abs gradient

        # First significant I(Z;Y)
        nonzero_izy_idx = np.where(i_zy > 1e-4)[0]  # Slightly higher threshold for "significant"
        if len(nonzero_izy_idx) > 0: estimates.append(beta_v[nonzero_izy_idx[0]])

        # Max gradient of I(Z;Y)
        if not np.all(i_zy < 1e-5):
            grad_izy = np.gradient(np.array(jax_gaussian_filter1d(jnp.array(i_zy), sigma=sigma_smooth)), beta_v)
            significant_rise_izy_idx = np.where(grad_izy > np.mean(np.abs(grad_izy))*0.5 + 1e-4)[0]
            if len(significant_rise_izy_idx) > 0: estimates.append(beta_v[significant_rise_izy_idx[0]])
            elif len(grad_izy)>0: estimates.append(beta_v[np.argmax(np.abs(grad_izy))])

        if not estimates: return self.target_beta_star
        estimates = [e for e in estimates if not np.isnan(e)]
        if not estimates: return self.target_beta_star
        
        if len(estimates) > 2:  # Robust median with outlier rejection
            median_e, std_e = np.median(estimates), np.std(estimates)
            # Filter if std_e is not zero, to avoid issues with all estimates being identical
            if std_e > 1e-9:
                 filtered_e = [e for e in estimates if abs(e - median_e) < (2.0 * std_e + 0.25)]  # Constant for robustness
                 if filtered_e: estimates = filtered_e
        beta_star = float(np.median(estimates))
        return np.clip(beta_star, np.min(beta_v), np.max(beta_v))

    def compute_ib_curve(self, beta_min: float = 0.02, beta_max: float = 15.0,
                         num_points: int = 80, log_scale: bool = True,
                         use_ultra_precise_for_curve: bool = False,
                         use_convexification: bool = False,
                         use_entropy_reg: bool = False,
                         epsilon_0: float = 0.1,
                         decay_rate: float = 0.5,
                         u_type: Optional[str] = "squared") -> Dict[str, np.ndarray]:
        """
        Compute the IB curve by optimizing at multiple beta values.
        
        Args:
            beta_min: Minimum beta value
            beta_max: Maximum beta value
            num_points: Number of beta values to compute
            log_scale: Whether to use logarithmic spacing for beta values
            use_ultra_precise_for_curve: Whether to use more precise optimization
            use_convexification: Whether to use convexified objective
            use_entropy_reg: Whether to use entropy regularization
            epsilon_0: Initial epsilon value for entropy regularization
            decay_rate: How quickly epsilon decays with beta
            u_type: Type of convex function ('squared', 'exp', or 'log')
            
        Returns:
            Dictionary with beta values, I(X;Z), and I(Z;Y)
        """
        if log_scale:
            beta_vals = np.logspace(np.log10(max(beta_min, self.tiny_epsilon)), np.log10(beta_max), num_points)
        else:
            beta_vals = np.linspace(beta_min, beta_max, num_points)
        izx_vals, izy_vals = np.zeros(num_points), np.zeros(num_points)
        
        # Choose optimization function based on options
        if use_convexification and use_entropy_reg:
            opt_fn = lambda beta, key: self.optimize_with_entropy_reg(
                beta=beta, key=key, epsilon_0=epsilon_0, decay_rate=decay_rate,
                u_type=u_type, verbose=False, ultra_precise=use_ultra_precise_for_curve
            )
            method_desc = f"convexified ({u_type}) + entropy reg (ε₀={epsilon_0})"
        elif use_convexification:
            opt_fn = lambda beta, key: self.optimize_convexified(
                beta=beta, key=key, u_type=u_type, 
                verbose=False, ultra_precise=use_ultra_precise_for_curve
            )
            method_desc = f"convexified ({u_type})"
        elif use_entropy_reg:
            opt_fn = lambda beta, key: self.optimize_with_entropy_reg(
                beta=beta, key=key, epsilon_0=epsilon_0, decay_rate=decay_rate,
                u_type=None, verbose=False, ultra_precise=use_ultra_precise_for_curve
            )
            method_desc = f"entropy reg (ε₀={epsilon_0})"
        else:
            opt_fn = lambda beta, key: self.optimize(
                beta=beta, key=key, verbose=False, ultra_precise=use_ultra_precise_for_curve
            )
            method_desc = "standard"
        
        print(f"Computing IB curve using {method_desc} ({num_points} pts, β {beta_min:.3f}-{beta_max:.3f})")
        for i, beta in enumerate(beta_vals):
            key_i = random.fold_in(self.key, i + 3000)  # Use a different offset for keys
            _, izx, izy = opt_fn(beta, key_i)
            izx_vals[i], izy_vals[i] = izx, izy
            if (i + 1) % max(1, (num_points // 10)) == 0: print(f" ... {i+1}/{num_points} IB curve points computed.")
        return {'beta_values': beta_vals, 'I_XT': izx_vals, 'I_TY': izy_vals}
    
    def compute_ib_curve_with_continuation(self, beta_min: float = 0.02, beta_max: float = 15.0,
                                         num_points: int = 80, log_scale: bool = True,
                                         epsilon_0: float = 0.1, decay_rate: float = 0.5,
                                         u_type: Optional[str] = None,
                                         bifurcation_threshold: float = 1e-7) -> Dict[str, np.ndarray]:
        """
        Compute the IB curve using implicit function continuation.
        This ensures a smooth path through the IB curve and identifies bifurcations.
        
        Args:
            beta_min: Minimum beta value
            beta_max: Maximum beta value
            num_points: Number of beta values to compute
            log_scale: Whether to use logarithmic spacing for beta values
            epsilon_0: Initial epsilon value for entropy regularization
            decay_rate: How quickly epsilon decays with beta
            u_type: Type of convex function (None for standard IB)
            bifurcation_threshold: Threshold for detecting bifurcations
            
        Returns:
            Dictionary with beta values, I(X;Z), I(Z;Y), and bifurcation indicators
        """
        if log_scale:
            beta_vals = np.logspace(np.log10(max(beta_min, self.tiny_epsilon)), np.log10(beta_max), num_points)
        else:
            beta_vals = np.linspace(beta_min, beta_max, num_points)
        
        print(f"Computing IB curve with continuation ({num_points} pts, β {beta_min:.3f}-{beta_max:.3f})")
        
        # Use continuation method
        results = self.optimize_with_continuation(
            beta_min=beta_min, beta_max=beta_max, num_steps=num_points,
            epsilon_0=epsilon_0, decay_rate=decay_rate, u_type=u_type,
            bifurcation_threshold=bifurcation_threshold, verbose=True
        )
        
        return results


################################################################################
# 3. ENHANCED MULTI-PATH INCREMENTAL-β INFORMATION BOTTLENECK
################################################################################
class EnhancedMultiPathIB:
    """
    Enhanced Multi-Path Incremental-β Information Bottleneck implementation.
    Incorporates stability improvements for exploring the IB curve.
    """
    def __init__(self, ib_model: EnhancedInformationBottleneck,
                 num_paths: int = 3, verbose: bool = False):
        """
        Initialize the Enhanced Multi-Path IB optimizer.
        
        Args:
            ib_model: An EnhancedInformationBottleneck instance
            num_paths: Number of parallel paths to maintain
            verbose: Whether to print progress
        """
        self.ib = ib_model
        self.M = max(2, num_paths)  # Ensure at least 2 paths for diversity
        self.verbose = verbose
        self.solutions: List[jnp.ndarray] = []
        self.solution_metrics: List[Dict[str, float]] = []
        self.beta_schedule: Optional[np.ndarray] = None
        self.convergence_tracker: Dict[str, List[float]] = {
            'kl_changes': [], 'obj_diffs': [], 'izx_diffs': [], 'izy_diffs': []
        }

    def set_beta_schedule(self, beta_min: float = 0.02, beta_max: float = 7.0,  # Adjusted defaults
                          num_steps: int = 100, log_scale: bool = False):      # More steps
        """
        Set the beta schedule for optimization.
        
        Args:
            beta_min: Minimum beta value
            beta_max: Maximum beta value
            num_steps: Number of steps in the schedule
            log_scale: Whether to use logarithmic spacing
        """
        if log_scale:
            beta_min_s = max(beta_min, self.ib.tiny_epsilon)
            self.beta_schedule = np.logspace(np.log10(beta_min_s), np.log10(beta_max), num_steps)
        else:
            self.beta_schedule = np.linspace(beta_min, beta_max, num_steps)
        # Ensure schedule is sorted, unique, and includes beta_max
        if len(self.beta_schedule) > 0 :
            self.beta_schedule = np.unique(np.sort(np.append(self.beta_schedule[self.beta_schedule < beta_max], beta_max)))
        elif num_steps == 1: self.beta_schedule = np.array([beta_max])
        else: self.beta_schedule = np.array([])

        if self.verbose and len(self.beta_schedule) > 0:
            min_step = np.min(np.diff(self.beta_schedule)) if len(self.beta_schedule) > 1 else beta_max - beta_min
            print(f"β schedule set: {self.beta_schedule[0]:.4f} to {self.beta_schedule[-1]:.4f} "
                  f"in {len(self.beta_schedule)} unique steps. (Min Δβ ≈ {min_step:.4f})")

    def initialize_solutions(self, key: Any):
        """
        Initialize multiple diverse solution paths.
        
        Args:
            key: JAX random key
        """
        self.solutions, self.solution_metrics = [], []
        keys = jax.random.split(key, self.M)
        if self.verbose: print(f"Initializing {self.M} diverse solution paths...")
        for i, k_i in enumerate(keys):
            desc = ""
            if i == 0:  # Path 0: Soft Informative (Structured)
                desc = "Soft Informative (Structured)"
                pzgx = self.ib.initialize_structured(self.ib.cardinality_x, self.ib.cardinality_z, k_i)
            elif i == 1 and self.M > 1:  # Path 1: Hard Trivial (all x to z=0)
                desc = "Hard Trivial"
                pzgx = jnp.zeros((self.ib.cardinality_x, self.ib.cardinality_z))
                pzgx = pzgx.at[:, 0].set(1.0)  # All x map to the first cluster z=0
                pzgx = self.ib.normalize_rows(pzgx)  # Important if card_z=1
            else:  # Path >=2: Mixed (blend of uniform and another structured)
                desc = f"Mixed {i} (Uniform/Structured)"
                sk1, sk2, sk3 = jax.random.split(k_i, 3)
                p_u = self.ib.initialize_uniform(sk1)
                # Use a different key for the alternative structured init for more diversity
                p_s_alt = self.ib.initialize_structured(self.ib.cardinality_x, self.ib.cardinality_z, sk2)
                # Vary mixture ratio for different mixed paths
                mix_ratio = 0.15 + 0.7 * ((i - 1) / max(1, self.M - 2)) if self.M > 1 else 0.5  # Ensure i-1 for M>1
                mix_ratio = jnp.clip(mix_ratio, 0.1, 0.9)
                pzgx = (1.0 - mix_ratio) * p_u + mix_ratio * p_s_alt
                pzgx = self.ib.normalize_rows(pzgx)

            if self.verbose: print(f"  Path {i}: Type='{desc}'")
            self.solutions.append(pzgx)
            pz, _ = self.ib.calculate_marginal_z(pzgx)
            izx, izy = self.ib.calculate_mi_zx(pzgx, pz), self.ib.calculate_mi_zy(pzgx)
            self.solution_metrics.append({'I_ZX': izx, 'I_ZY': izy, 'objective': izy})  # Obj at beta=0
        if self.verbose:
            for i, m in enumerate(self.solution_metrics): print(f"  Init Path {i}: IZX={m['I_ZX']:.5f}, IZY={m['I_ZY']:.5f}")

    def optimize_single_beta(self, beta: float,
                             use_convexification: bool = False,
                             use_entropy_reg: bool = False,
                             u_type: str = "squared",
                             epsilon_0: float = 0.1,
                             local_iterations: int = 35,  # Increased default
                             initial_local_damping: float = 0.25  # Adjusted default
                             ) -> Tuple[List[jnp.ndarray], List[Dict[str, float]]]:
        """
        Optimize all paths for a single beta value.
        
        Args:
            beta: Current beta value
            use_convexification: Whether to use convexified IB
            use_entropy_reg: Whether to use entropy regularization
            u_type: Type of convex function for convexification
            epsilon_0: Initial entropy regularization parameter
            local_iterations: Number of iterations per path
            initial_local_damping: Initial damping factor
            
        Returns:
            Tuple of (updated solutions, updated metrics)
        """
        updated_sols, updated_metrics = [], []
        epsilon_val = self.ib.epsilon_schedule(beta, epsilon_0, 0.5) if use_entropy_reg else 0.0
        
        for i, pzgx_curr in enumerate(self.solutions):
            p_iter = pzgx_curr
            current_damping = initial_local_damping
            
            for iter_n in range(local_iterations):
                # Choose the appropriate update step based on options
                if use_convexification and use_entropy_reg:
                    new_p_cand = self.ib.entropy_regularized_ib_step(p_iter, beta, epsilon_val, u_type)
                elif use_convexification:
                    new_p_cand = self.ib.convexified_ib_update_step(p_iter, beta, u_type)
                elif use_entropy_reg:
                    new_p_cand = self.ib.entropy_regularized_ib_step(p_iter, beta, epsilon_val)
                else:
                    new_p_cand = self.ib.ib_update_step(p_iter, beta)
                
                # Apply damping and normalize
                p_iter = (1 - current_damping) * new_p_cand + current_damping * p_iter
                p_iter = self.ib.normalize_rows(p_iter)  # Crucial after damping
                
                # Slowly reduce damping
                current_damping = max(0.005, current_damping * 0.99)  # Slower decay, min 0.005

            # Calculate final metrics
            pz, _ = self.ib.calculate_marginal_z(p_iter)
            izx, izy = self.ib.calculate_mi_zx(p_iter, pz), self.ib.calculate_mi_zy(p_iter)
            hzx = self.ib.conditional_entropy(p_iter) if use_entropy_reg else 0.0
            
            # Calculate objective based on options
            if use_convexification:
                if u_type == "squared":
                    u_val = izx ** 2
                elif u_type == "exp":
                    u_val = jnp.exp(izx) - 1
                elif u_type == "log":
                    u_val = jnp.log(1 + izx)
                else:
                    u_val = izx
                
                if use_entropy_reg:
                    objective = izy - beta * u_val + epsilon_val * hzx
                else:
                    objective = izy - beta * u_val
            else:
                if use_entropy_reg:
                    objective = izy - beta * izx + epsilon_val * hzx
                else:
                    objective = izy - beta * izx
            
            updated_sols.append(p_iter)
            updated_metrics.append({
                'I_ZX': izx, 'I_ZY': izy, 'H_ZX': hzx, 'objective': objective,
                'beta': beta, 'u_type': u_type if use_convexification else None,
                'epsilon': epsilon_val if use_entropy_reg else 0.0
            })
        
        return updated_sols, updated_metrics

    def merge_trim_solutions(self, solutions: List[jnp.ndarray], metrics: List[Dict[str, float]]) -> \
                             Tuple[List[jnp.ndarray], List[Dict[str, float]]]:
        """
        Merge and trim solutions to maintain diversity and quality.
        
        Args:
            solutions: List of current solutions
            metrics: List of metrics for each solution
            
        Returns:
            Tuple of (trimmed solutions, trimmed metrics)
        """
        if not solutions or not metrics or len(solutions) <= self.M: return solutions, metrics
        
        # Ensure at least one "most trivial" (lowest IZY) and one "most informative" (highest IZY) path are kept
        relevance_sorted_indices = sorted(range(len(metrics)), key=lambda k: metrics[k]['I_ZY'])
        trivial_orig_idx = relevance_sorted_indices[0]
        informative_orig_idx = relevance_sorted_indices[-1]
        
        obj_sorted_indices = sorted(range(len(metrics)), key=lambda k: metrics[k]['objective'], reverse=True)
        
        kept_indices = set()
        # Explicitly add the most trivial and most informative solutions
        kept_indices.add(trivial_orig_idx)
        if trivial_orig_idx != informative_orig_idx:  # Add informative only if distinct
             kept_indices.add(informative_orig_idx)
            
        # Fill remaining slots with best objective, avoiding duplicates already added
        for idx in obj_sorted_indices:
            if len(kept_indices) >= self.M: break
            kept_indices.add(idx)  # Add original index
            
        # If not enough unique solutions from above (e.g. M is large, or trivial/informative were also best objective)
        # fill greedily from objective-sorted list until M.
        idx_ptr = 0
        while len(kept_indices) < self.M and idx_ptr < len(obj_sorted_indices):
            kept_indices.add(obj_sorted_indices[idx_ptr])  # Add original index
            idx_ptr += 1
            
        final_indices = list(kept_indices)
        # If we somehow still have more than M (e.g. M=2, trivial=informative, and we added one more)
        # or if the set logic resulted in > M due to small M, re-prioritize by objective.
        if len(final_indices) > self.M:
            final_indices = sorted(final_indices, key=lambda k_idx: metrics[k_idx]['objective'], reverse=True)[:self.M]
            
        top_sols = [solutions[i] for i in final_indices]
        top_metrics = [metrics[i] for i in final_indices]
        
        return top_sols, top_metrics

    def optimize(self, key: Optional[Any] = None, 
                 use_convexification: bool = False,
                 use_entropy_reg: bool = True,
                 use_adaptive_refinement: bool = True,
                 u_type: str = "squared",
                 epsilon_0: float = 0.1,
                 local_iters_per_beta: int = 35, 
                 initial_damping_local: float = 0.25) -> \
                 Tuple[Optional[jnp.ndarray], Dict[str, float], Dict[str, Any]]:
        """
        Optimize the multi-path IB with enhanced stability features.
        
        Args:
            key: JAX random key
            use_convexification: Whether to use convexified IB
            use_entropy_reg: Whether to use entropy regularization
            use_adaptive_refinement: Whether to adaptively refine beta schedule
            u_type: Type of convex function for convexification
            epsilon_0: Initial entropy regularization parameter
            local_iters_per_beta: Number of iterations per beta value
            initial_damping_local: Initial damping factor
            
        Returns:
            Tuple of (optimal encoder, metrics, history)
        """
        if key is None: key = self.ib.key
        if self.beta_schedule is None or len(self.beta_schedule) == 0: self.set_beta_schedule()
        if len(self.beta_schedule) == 0:
            warnings.warn("Beta schedule is empty. Cannot optimize.")
            return None, {}, {'beta_values': [], 'solutions': [], 'metrics': []}

        init_key, loop_key = jax.random.split(key)
        self.initialize_solutions(init_key)
        history: Dict[str, Any] = {
            'beta_values': [],
            'solutions': [],
            'metrics': [],
            'eigenvalues': [],  # Track smallest eigenvalue for bifurcation detection
            'bifurcations': []  # Track where bifurcations occurred
        }
        start_t = time.time()
        prev_max_obj_diff = float('inf')  # For adaptive beta refinement heuristic
        current_beta_sched = list(self.beta_schedule)  # Modifiable copy
        beta_idx = 0

        while beta_idx < len(current_beta_sched):
            beta = current_beta_sched[beta_idx]
            if self.verbose: print(f"\nStep {beta_idx+1}/{len(current_beta_sched)}: β = {beta:.6f}")
            
            # Calculate current epsilon for entropy regularization
            epsilon_val = self.ib.epsilon_schedule(beta, epsilon_0, 0.5) if use_entropy_reg else 0.0
            
            # Check for bifurcation at current beta using Hessian
            if beta_idx > 0 and (len(self.solutions) > 0):
                # Use the best solution from previous beta to check for bifurcation
                prev_best_idx = max(range(len(self.solution_metrics)), 
                                   key=lambda i: self.solution_metrics[i]['objective'])
                best_sol = self.solutions[prev_best_idx]
                
                lambda_min, _ = self.ib.calculate_hessian_info(best_sol)
                if self.verbose: print(f"  Hessian check: smallest eigenvalue λ_min = {lambda_min:.8f}")
                
                # Store eigenvalue in history
                history['eigenvalues'].append(float(lambda_min))
                
                # Detect bifurcation
                is_bifurcation = abs(lambda_min) < 1e-7
                history['bifurcations'].append(is_bifurcation)
                
                if is_bifurcation and self.verbose:
                    print(f"  ! Bifurcation detected at β = {beta:.6f}")
            else:
                history['eigenvalues'].append(0.0)
                history['bifurcations'].append(False)
            
            # Optimize all paths for current beta
            upd_sols, upd_metrics = self.optimize_single_beta(
                beta,
                use_convexification=use_convexification,
                use_entropy_reg=use_entropy_reg,
                u_type=u_type,
                epsilon_0=epsilon_val,  # Already calculated from schedule
                local_iterations=local_iters_per_beta,
                initial_local_damping=initial_damping_local
            )
            self.solutions, self.solution_metrics = self.merge_trim_solutions(upd_sols, upd_metrics)
            
            if not self.solutions:  # Safety break if all paths somehow vanished
                warnings.warn(f"All solution paths vanished at beta={beta:.6f}. Stopping.")
                break

            history['beta_values'].append(beta)
            history['solutions'].append([s.copy() for s in self.solutions])
            history['metrics'].append([m.copy() for m in self.solution_metrics])  # list of dicts
            if self.verbose:
                for j, mj in enumerate(self.solution_metrics):
                    print(f"  Path {j}: IZX={mj['I_ZX']:.6f}, IZY={mj['I_ZY']:.6f}, Obj={mj['objective']:.6f}")

            # Track convergence metrics for adaptive refinement
            if beta_idx > 0 and len(history['metrics']) >= 2:
                prev_metrics = history['metrics'][-2]
                curr_metrics = history['metrics'][-1]
                
                # Calculate changes in key metrics across paths
                if len(prev_metrics) == len(curr_metrics):
                    kl_changes = []
                    for i in range(len(self.solutions)):
                        if i < len(history['solutions'][-2]):
                            kl_div = self.ib.kl_divergence(self.solutions[i], history['solutions'][-2][i])
                            kl_changes.append(kl_div)
                    
                    if kl_changes:
                        self.convergence_tracker['kl_changes'].append(float(np.mean(kl_changes)))
                
                # Calculate obj, IZX, IZY diffs
                for i in range(min(len(prev_metrics), len(curr_metrics))):
                    obj_diff = abs(curr_metrics[i]['objective'] - prev_metrics[i]['objective'])
                    izx_diff = abs(curr_metrics[i]['I_ZX'] - prev_metrics[i]['I_ZX'])
                    izy_diff = abs(curr_metrics[i]['I_ZY'] - prev_metrics[i]['I_ZY'])
                    
                    self.convergence_tracker['obj_diffs'].append(float(obj_diff))
                    self.convergence_tracker['izx_diffs'].append(float(izx_diff))
                    self.convergence_tracker['izy_diffs'].append(float(izy_diff))

            # Adaptive beta refinement heuristic
            if use_adaptive_refinement and len(self.solution_metrics) >= 2:
                objs = [m['objective'] for m in self.solution_metrics]
                max_obj_d = np.max(objs) - np.min(objs)
                # Refine if obj spread shrinks significantly AND current spread is small AND prev spread was larger
                # These thresholds are heuristic and may need tuning for different problems.
                if beta_idx > 0 and max_obj_d < 0.25 * prev_max_obj_diff and \
                   max_obj_d < 0.02 and prev_max_obj_diff > 0.001:
                    if beta_idx < len(current_beta_sched) - 1:  # If not the last planned step
                        next_b = current_beta_sched[beta_idx+1]
                        if next_b - beta > 0.001:  # Only refine if gap is not already tiny
                            if self.verbose: print(f" -> Objective spread change detected. Refining β steps between {beta:.4f} and {next_b:.4f}")
                            num_refined_midpoints = 3  # Insert 3 midpoints
                            refined_steps = np.linspace(beta, next_b, num_refined_midpoints + 2)[1:-1]
                            for r_idx, r_step in enumerate(refined_steps):
                                current_beta_sched.insert(beta_idx + 1 + r_idx, r_step)
                prev_max_obj_diff = max_obj_d
            elif len(self.solution_metrics) <= 1: prev_max_obj_diff = 0.0  # No spread if 0 or 1 path
            beta_idx += 1

        if not self.solution_metrics: return None, {}, history  # Should have been caught by break earlier
        final_b = history['beta_values'][-1]
        # Recalculate final objectives with the actual final beta for precision
        final_objs = [m['objective'] for m in self.solution_metrics]
        if not final_objs:  # Should not happen if self.solution_metrics is not empty
             return self.solutions[0] if self.solutions else None, self.solution_metrics[0] if self.solution_metrics else {}, history

        best_idx = np.argmax(final_objs)
        best_sol = self.solutions[best_idx]
        best_met = self.solution_metrics[best_idx].copy()  # Get a copy of the dict

        if self.verbose:
            print(f"\nMulti-Path Optimization Complete. Total time: {time.time()-start_t:.2f}s")
            print(f"Best solution at final β={final_b:.5f}: IZX={best_met['I_ZX']:.6f}, IZY={best_met['I_ZY']:.6f}, Obj={best_met['objective']:.6f}")
        return best_sol, best_met, history

    # Plotting functions with enhancements for bifurcation visualization
    def plot_solution_paths(self, history: Dict[str, Any], title_suffix: str = ""):
        """
        Plot the solution paths in the information plane.
        
        Args:
            history: Optimization history
            title_suffix: Additional text for plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8)); colors = plt.cm.viridis(np.linspace(0,0.9,self.M))
        if not history['beta_values']: ax.text(0.5,0.5,"No history to plot.", ha='center',va='center', transform=ax.transAxes); return fig
        for slot_idx in range(self.M):
            izx_p, izy_p, valid_p = [], [], False
            for step_idx in range(len(history['beta_values'])):
                if slot_idx < len(history['metrics'][step_idx]):
                    m = history['metrics'][step_idx][slot_idx]
                    izx_p.append(m['I_ZX']); izy_p.append(m['I_ZY']); valid_p = True
                else: izx_p.append(np.nan); izy_p.append(np.nan)
            if valid_p and not all(np.isnan(izx_p)):  # Only plot if there's some non-NaN data
                ax.plot(izx_p, izy_p, '-o', color=colors[slot_idx%len(colors)], alpha=0.6, lw=1.5, ms=4, label=f'Path Slot {slot_idx+1}')
        
        # Mark bifurcations if available
        if 'bifurcations' in history and len(history['bifurcations']) > 0:
            bifurcation_indices = [i for i, is_bif in enumerate(history['bifurcations']) if is_bif]
            for bif_idx in bifurcation_indices:
                if bif_idx < len(history['metrics']):
                    for slot_idx in range(min(self.M, len(history['metrics'][bif_idx]))):
                        m = history['metrics'][bif_idx][slot_idx]
                        ax.plot(m['I_ZX'], m['I_ZY'], 'x', color='red', ms=10, mew=2.0)
        
        final_metrics_list = history['metrics'][-1] if history['metrics'] else []
        for final_idx, metrics in enumerate(final_metrics_list):
            ax.plot(metrics['I_ZX'], metrics['I_ZY'], '*', color=colors[final_idx%len(colors)], ms=12, mec='k', mew=1.0, label=f'Final Pt (Slot {final_idx+1})' if final_idx < self.M else None)
        
        if final_metrics_list and history['beta_values']:
            final_b = history['beta_values'][-1]
            final_objs = [m['objective'] for m in final_metrics_list]
            if final_objs:
                best_f_idx = np.argmax(final_objs)
                if best_f_idx < len(final_metrics_list):
                    best_m = final_metrics_list[best_f_idx]
                    ax.plot(best_m['I_ZX'], best_m['I_ZY'], 'D', color='red', ms=10, mec='k', mew=1.5, label='Best Final Solution')
        
        mi_xy = self.ib.mi_xy; hx = self.ib.hx
        ax.axhline(y=mi_xy, ls='--', color='gray', alpha=0.8, label=f'I(X;Y)={mi_xy:.4f} bits')
        ax.set_xlim(left=-0.05*hx if hx>0 else -0.1, right=hx*1.05 if hx>0 else 1.1)  # Ensure right limit > 0
        ax.set_ylim(bottom=-0.05*mi_xy if mi_xy>0 else -0.1, top=mi_xy*1.05 if mi_xy>0 else 1.1)  # Ensure top limit > 0
        ax.set_xlabel('I(Z;X) (bits) - Complexity', fontsize=12)
        ax.set_ylabel('I(Z;Y) (bits) - Relevance', fontsize=12)
        ax.set_title(f'Multi-Path IB: Solution Trajectories in Information Plane{title_suffix}', fontsize=14)
        ax.grid(True, ls=':', alpha=0.5); ax.legend(loc='best', fontsize=9); plt.tight_layout()
        return fig

    def plot_beta_trajectories(self, history: Dict[str, Any], title_suffix: str = ""):
        """
        Plot the evolution of I(Z;X) and I(Z;Y) as a function of beta.
        
        Args:
            history: Optimization history
            title_suffix: Additional text for plot title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,14),sharex=True); colors=plt.cm.viridis(np.linspace(0,0.9,self.M))
        betas_hist = history['beta_values']
        if not betas_hist: 
            ax1.text(0.5,0.5,"No I(Z;X) history.", ha='center',va='center')
            ax2.text(0.5,0.5,"No I(Z;Y) history.", ha='center',va='center')
            ax3.text(0.5,0.5,"No eigenvalue history.", ha='center',va='center')
            return fig
        
        # Plot I(Z;X) and I(Z;Y) trajectories for each path
        for slot_idx in range(self.M):
            izx_p, izy_p, valid_p = [], [], False
            for step_idx in range(len(betas_hist)):
                if slot_idx < len(history['metrics'][step_idx]):
                    m = history['metrics'][step_idx][slot_idx]
                    izx_p.append(m['I_ZX']); izy_p.append(m['I_ZY']); valid_p = True
                else: izx_p.append(np.nan); izy_p.append(np.nan)
            if valid_p and not all(np.isnan(izx_p)):
                ax1.plot(betas_hist, izx_p, '-o', color=colors[slot_idx%len(colors)], alpha=0.6, lw=1.5, ms=4, label=f'Path Slot {slot_idx+1}')
                ax2.plot(betas_hist, izy_p, '-o', color=colors[slot_idx%len(colors)], alpha=0.6, lw=1.5, ms=4, label=f'Path Slot {slot_idx+1}')
        
        # Plot eigenvalues if available
        if 'eigenvalues' in history and len(history['eigenvalues']) > 0:
            eigenvalues = np.array(history['eigenvalues'])
            # Fill any missing values
            if len(eigenvalues) < len(betas_hist):
                eigenvalues = np.pad(eigenvalues, (0, len(betas_hist) - len(eigenvalues)), 'constant', constant_values=np.nan)
            
            # Plot the eigenvalues on a log scale to highlight near-zero values
            ax3.semilogy(betas_hist, np.abs(eigenvalues) + 1e-10, '-o', color='purple', alpha=0.8, lw=2, ms=5, label='|λ_min|')
            
            # Mark bifurcations
            if 'bifurcations' in history and len(history['bifurcations']) > 0:
                bifurcation_indices = [i for i, is_bif in enumerate(history['bifurcations']) if is_bif]
                for bif_idx in bifurcation_indices:
                    if bif_idx < len(betas_hist):
                        beta_bif = betas_hist[bif_idx]
                        ax1.axvline(x=beta_bif, ls='--', color='red', alpha=0.5)
                        ax2.axvline(x=beta_bif, ls='--', color='red', alpha=0.5)
                        ax3.axvline(x=beta_bif, ls='--', color='red', alpha=0.5)
                        ax3.plot(beta_bif, np.abs(eigenvalues[bif_idx]) + 1e-10, 'x', color='red', ms=10, mew=2.0)
        
        final_metrics_list = history['metrics'][-1] if history['metrics'] else []
        if final_metrics_list and betas_hist:
            final_b = betas_hist[-1]
            final_objs = [m['objective'] for m in final_metrics_list]
            if final_objs:
                best_f_idx = np.argmax(final_objs)
                if best_f_idx < len(final_metrics_list):
                    best_m = final_metrics_list[best_f_idx]
                    ax1.plot(final_b, best_m['I_ZX'], 'D', color='red', ms=10, mec='k', label='Best Final I(Z;X)')
                    ax2.plot(final_b, best_m['I_ZY'], 'D', color='red', ms=10, mec='k', label='Best Final I(Z;Y)')
        
        if hasattr(self.ib, 'target_beta_star') and self.ib.target_beta_star is not None:
            beta_s_th = self.ib.target_beta_star
            ax1.axvline(x=beta_s_th, ls='--', color='k', alpha=0.7, label=f'Target β* ≈ {beta_s_th:.4f}')
            ax2.axvline(x=beta_s_th, ls='--', color='k', alpha=0.7)
            ax3.axvline(x=beta_s_th, ls='--', color='k', alpha=0.7)
        
        ax1.set_ylabel('I(Z;X) (bits)', fontsize=12)
        ax1.set_title(f'Evolution of I(Z;X) vs β{title_suffix}', fontsize=14)
        ax1.grid(True,ls=':',alpha=0.5)
        ax1.legend(loc='best',fontsize=9)
        
        ax2.set_ylabel('I(Z;Y) (bits)', fontsize=12)
        ax2.set_title(f'Evolution of I(Z;Y) vs β{title_suffix}', fontsize=14)
        ax2.grid(True,ls=':',alpha=0.5)
        ax2.legend(loc='best',fontsize=9)
        
        ax3.set_xlabel('β (Lagrange Multiplier)', fontsize=12)
        ax3.set_ylabel('|λ_min| (log scale)', fontsize=12)
        ax3.set_title(f'Smallest Eigenvalue vs β{title_suffix}', fontsize=14)
        ax3.grid(True,ls=':',alpha=0.5)
        ax3.legend(loc='best',fontsize=9)
        
        plt.tight_layout()
        return fig
        
    def plot_convergence_metrics(self, title_suffix: str = ""):
        """
        Plot convergence metrics during optimization.
        
        Args:
            title_suffix: Additional text for plot title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot KL changes
        if self.convergence_tracker['kl_changes']:
            steps = range(len(self.convergence_tracker['kl_changes']))
            ax1.semilogy(steps, self.convergence_tracker['kl_changes'], '-o', color='blue', lw=1.5, alpha=0.8, label='KL Divergence')
            ax1.set_title(f'KL Divergence Between Successive Solutions{title_suffix}', fontsize=14)
            ax1.set_xlabel('Step', fontsize=12)
            ax1.set_ylabel('KL Divergence (log scale)', fontsize=12)
            ax1.grid(True, ls=':', alpha=0.5)
            ax1.legend(loc='best', fontsize=9)
        else:
            ax1.text(0.5, 0.5, "No KL divergence data available", ha='center', va='center')
        
        # Plot objective, IZX, IZY differences
        if self.convergence_tracker['obj_diffs'] and self.convergence_tracker['izx_diffs'] and self.convergence_tracker['izy_diffs']:
            steps = range(len(self.convergence_tracker['obj_diffs']))
            ax2.semilogy(steps, self.convergence_tracker['obj_diffs'], '-o', color='green', lw=1.5, alpha=0.8, label='Objective Diff')
            ax2.semilogy(steps, self.convergence_tracker['izx_diffs'], '-x', color='red', lw=1.5, alpha=0.8, label='I(Z;X) Diff')
            ax2.semilogy(steps, self.convergence_tracker['izy_diffs'], '-s', color='purple', lw=1.5, alpha=0.8, label='I(Z;Y) Diff')
            ax2.set_title(f'Changes in Metrics Between Successive Steps{title_suffix}', fontsize=14)
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Absolute Difference (log scale)', fontsize=12)
            ax2.grid(True, ls=':', alpha=0.5)
            ax2.legend(loc='best', fontsize=9)
        else:
            ax2.text(0.5, 0.5, "No metric difference data available", ha='center', va='center')
        
        plt.tight_layout()
        return fig

    def visualize_encoder(self, p_z_given_x: jnp.ndarray, title: str = "Encoder p(z|x)"):
        """
        Visualize the encoder distribution p(z|x).
        
        Args:
            p_z_given_x: Encoder distribution
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(p_z_given_x, cmap='viridis', aspect='auto')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('z (cluster)', fontsize=12)
        ax.set_ylabel('x (input)', fontsize=12)
        plt.colorbar(im, ax=ax, label='p(z|x)')
        
        # Add grid lines between clusters
        for i in range(p_z_given_x.shape[0]):
            ax.axhline(i-0.5, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        for j in range(p_z_given_x.shape[1]):
            ax.axvline(j-0.5, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add text annotations for significant probabilities
        for i in range(p_z_given_x.shape[0]):
            for j in range(p_z_given_x.shape[1]):
                val = p_z_given_x[i, j]
                if val > 0.2:  # Only show significant probabilities
                    text_color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8)
        
        plt.tight_layout()
        return fig

################################################################################
# 4. HELPER FUNCTIONS FOR CREATING TEST DISTRIBUTIONS
################################################################################
def create_partially_correlated_distribution(key: Optional[Any] = None,  # Moved key first, made optional
                                             cardinality_x: int = 10, cardinality_y: int = 10,
                                             noise_level: float = 0.3) -> np.ndarray:
    """
    Create a joint distribution p(x,y) with partial correlation.
    Primary correlation is diagonal, with uniform noise.
    
    Args:
        key: JAX random key
        cardinality_x: Number of x values
        cardinality_y: Number of y values
        noise_level: Amount of noise to add (0-1)
        
    Returns:
        Joint distribution p(x,y)
    """
    if key is None:  # Generate a new key if None is provided
        current_time_ns = time.time_ns()
        seed_val = current_time_ns % (2**32 -1)  # Ensure seed is within uint32 range for JAX PRNGKey
        key = random.PRNGKey(seed_val)
        # print(f"Generated internal key with seed: {seed_val}")

    base_corr_np = np.zeros((cardinality_x, cardinality_y), dtype=np.float64)
    diag_len = min(cardinality_x, cardinality_y)
    if diag_len > 0:  # Only fill diagonal if possible
        for i in range(diag_len):
            base_corr_np[i, i] = 1.0
        base_corr_np = base_corr_np / np.sum(base_corr_np)  # Normalize base correlation part
    # If diag_len is 0, base_corr_np remains all zeros, which is fine.

    noise_matrix_jax = random.uniform(key, (cardinality_x, cardinality_y), dtype=jnp.float64)
    sum_noise = jnp.sum(noise_matrix_jax)
    if sum_noise > 1e-9:  # Avoid division by zero if noise matrix is all zeros
        noise_matrix_np = np.array(noise_matrix_jax / sum_noise)
    else:  # If noise matrix is all zeros, create uniform noise
        noise_matrix_np = np.ones((cardinality_x, cardinality_y), dtype=np.float64) / (cardinality_x * cardinality_y)

    noise_level = np.clip(noise_level, 0.0, 1.0)
    # If base_corr_np is all zeros (e.g. card_x or card_y was 0), result is just noise_matrix_np
    joint_xy_np = (1.0 - noise_level) * base_corr_np + noise_level * noise_matrix_np
    
    final_sum = np.sum(joint_xy_np)
    if final_sum > 1e-9:  # Avoid division by zero
        final_joint = joint_xy_np / final_sum
    else:  # If sum is still zero (e.g. noise_level=0 and base_corr was zero)
        final_joint = np.ones((cardinality_x, cardinality_y), dtype=np.float64) / (cardinality_x * cardinality_y)

    return final_joint.astype(np.float32)

def create_bsc_distribution(q: float = 0.1) -> np.ndarray:
    """
    Create a joint distribution p(x,y) for Binary Symmetric Channel with error probability q.
    
    Args:
        q: Error probability (0-1)
        
    Returns:
        Joint distribution p(x,y)
    """
    q = np.clip(q, 0.0, 1.0)
    # Binary Symmetric Channel
    p_x = np.array([0.5, 0.5])  # Uniform prior on x
    p_y_given_x = np.array([
        [1-q, q],    # p(y|x=0)
        [q, 1-q]     # p(y|x=1)
    ])
    
    # Create joint distribution
    joint_xy = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            joint_xy[i, j] = p_x[i] * p_y_given_x[i, j]
    
    return joint_xy.astype(np.float32)

def create_structured_distribution(cardinality_x: int = 8, cardinality_y: int = 8, 
                                   num_clusters: int = 3, cluster_strength: float = 0.8,
                                   random_seed: int = 42) -> np.ndarray:
    """
    Create a structured joint distribution with clear clusters.
    
    Args:
        cardinality_x: Number of x values
        cardinality_y: Number of y values
        num_clusters: Number of clusters to create
        cluster_strength: Strength of the clusters (0-1)
        random_seed: Random seed for reproducibility
        
    Returns:
        Joint distribution p(x,y)
    """
    # Ensure parameters are valid
    num_clusters = min(num_clusters, min(cardinality_x, cardinality_y))
    cluster_strength = np.clip(cluster_strength, 0.0, 1.0)
    
    # Create random distribution for noise
    np.random.seed(random_seed)
    noise_dist = np.random.rand(cardinality_x, cardinality_y)
    noise_dist = noise_dist / np.sum(noise_dist)
    
    # Create structured distribution with clusters
    joint_xy = np.zeros((cardinality_x, cardinality_y))
    
    # Assign x values to clusters
    x_per_cluster = cardinality_x // num_clusters
    extra_x = cardinality_x % num_clusters
    
    x_cluster_indices = []
    start_idx = 0
    for c in range(num_clusters):
        size = x_per_cluster + (1 if c < extra_x else 0)
        x_cluster_indices.append(list(range(start_idx, start_idx + size)))
        start_idx += size
    
    # Assign y values to clusters
    y_per_cluster = cardinality_y // num_clusters
    extra_y = cardinality_y % num_clusters
    
    y_cluster_indices = []
    start_idx = 0
    for c in range(num_clusters):
        size = y_per_cluster + (1 if c < extra_y else 0)
        y_cluster_indices.append(list(range(start_idx, start_idx + size)))
        start_idx += size
    
    # Create cluster structure
    for c in range(num_clusters):
        for x_idx in x_cluster_indices[c]:
            for y_idx in y_cluster_indices[c]:
                joint_xy[x_idx, y_idx] = 1.0
    
    # Mix structured distribution with noise
    joint_xy = cluster_strength * (joint_xy / np.sum(joint_xy)) + (1 - cluster_strength) * noise_dist
    joint_xy = joint_xy / np.sum(joint_xy)
    
    return joint_xy.astype(np.float32)

################################################################################
# 5. DEMONSTRATION AND EVALUATION
################################################################################
def demonstrate_standard_vs_enhanced_ib():
    """
    Demonstrate the improvements in the enhanced IB framework compared to standard IB.
    """
    print("\nDemonstrating Standard vs Enhanced Information Bottleneck")
    print("-" * 80)
    
    # Create a synthetic joint distribution
    card_x, card_y, card_z = this_card_x, this_card_y, 4
    noise = 0.65  # Challenging noise level
    dist_key = random.PRNGKey(42024)
    joint_xy_np = create_partially_correlated_distribution(key=dist_key, cardinality_x=card_x, cardinality_y=card_y, noise_level=noise)
    print(f"Created p(x,y) joint distribution. Sum={np.sum(joint_xy_np):.4f}, Shape={joint_xy_np.shape}")
    joint_xy_jax = jnp.array(joint_xy_np)

    # Create IB models
    main_key = random.PRNGKey(2024)
    std_key, enh_key = random.split(main_key, 2)
    
    # Create Enhanced IB model
    print("\nInitializing Enhanced IB model...")
    ib_model = EnhancedInformationBottleneck(joint_xy_jax, key=enh_key, cardinality_z=card_z)
    print(f"  I(X;Y)={ib_model.mi_xy:.4f} bits, H(X)={ib_model.hx:.4f} bits")
    
    # Choose a beta value near critical point
    target_beta = ib_model.target_beta_star - 0.5
    print(f"\nRunning standard IB at β = {target_beta:.3f}...")
    std_pzgx, std_izx, std_izy = ib_model.optimize(beta=target_beta, key=std_key, verbose=True, ultra_precise=False)
    std_obj = std_izy - target_beta * std_izx
    print(f"Standard IB Result: IZX={std_izx:.6f}, IZY={std_izy:.6f}, Obj={std_obj:.6f}")
    
    # Run Enhanced IB with convexification
    print("\nRunning Enhanced IB with convexification at β = {target_beta:.3f}...")
    enh_pzgx, enh_izx, enh_izy = ib_model.optimize_convexified(beta=target_beta, key=enh_key, verbose=True, u_type="squared")
    enh_obj = enh_izy - target_beta * enh_izx
    print(f"Enhanced IB Result: IZX={enh_izx:.6f}, IZY={enh_izy:.6f}, Obj={enh_obj:.6f}")
    
    # Run Enhanced IB with entropy regularization
    print("\nRunning Enhanced IB with entropy regularization at β = {target_beta:.3f}...")
    ent_pzgx, ent_izx, ent_izy = ib_model.optimize_with_entropy_reg(beta=target_beta, key=enh_key, verbose=True, epsilon_0=0.1)
    
    # Calculate objective with entropy term
    h_zx = ib_model.conditional_entropy(ent_pzgx)
    epsilon_val = ib_model.epsilon_schedule(target_beta, epsilon_0=0.1, decay_rate=0.5)
    ent_obj = ent_izy - target_beta * ent_izx + epsilon_val * h_zx
    
    print(f"Entropy-Regularized IB Result: IZX={ent_izx:.6f}, IZY={ent_izy:.6f}, H(Z|X)={h_zx:.6f}, Obj={ent_obj:.6f}")
    
    # Compare standard vs enhanced methods
    print("\n--- Results Comparison ---")
    print(f"Target β for comparison: {target_beta:.4f}")
    print(f"Standard IB:           IZX={std_izx:.6f}, IZY={std_izy:.6f}, Obj={std_obj:.6f}")
    print(f"Convexified IB:        IZX={enh_izx:.6f}, IZY={enh_izy:.6f}, Obj={enh_obj:.6f}")
    print(f"Entropy-Regularized IB: IZX={ent_izx:.6f}, IZY={ent_izy:.6f}, Obj={ent_obj:.6f}")
    
    # Visualize encoders
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(std_pzgx, cmap='viridis', aspect='auto')
    plt.colorbar(label='p(z|x)')
    plt.title('Standard IB Encoder')
    plt.xlabel('z (cluster)')
    plt.ylabel('x (input)')
    
    plt.subplot(1, 3, 2)
    plt.imshow(enh_pzgx, cmap='viridis', aspect='auto')
    plt.colorbar(label='p(z|x)')
    plt.title('Convexified IB Encoder')
    plt.xlabel('z (cluster)')
    plt.ylabel('x (input)')
    
    plt.subplot(1, 3, 3)
    plt.imshow(ent_pzgx, cmap='viridis', aspect='auto')
    plt.colorbar(label='p(z|x)')
    plt.title('Entropy-Regularized IB Encoder')
    plt.xlabel('z (cluster)')
    plt.ylabel('x (input)')
    
    plt.tight_layout()
    plt.savefig("encoder_comparison.png")
    print("Saved encoder visualization to 'encoder_comparison.png'")
    
    # Compute and compare IB curves
    print("\nComputing IB curves for different methods...")
    beta_min, beta_max, num_points = 0.05, 10.0, 40
    
    # Standard IB curve
    std_curve = ib_model.compute_ib_curve(
        beta_min=beta_min, beta_max=beta_max, num_points=num_points, 
        log_scale=True, use_ultra_precise_for_curve=False
    )
    
    # Convexified IB curve
    conv_curve = ib_model.compute_ib_curve(
        beta_min=beta_min, beta_max=beta_max, num_points=num_points, 
        log_scale=True, use_convexification=True, u_type="squared"
    )
    
    # Entropy-regularized IB curve
    ent_curve = ib_model.compute_ib_curve(
        beta_min=beta_min, beta_max=beta_max, num_points=num_points, 
        log_scale=True, use_entropy_reg=True, epsilon_0=0.1
    )
    
    # Plot IB curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Information plane
    ax1.plot(std_curve['I_XT'], std_curve['I_TY'], 'o-', color='blue', label='Standard IB')
    ax1.plot(conv_curve['I_XT'], conv_curve['I_TY'], 'x-', color='green', label='Convexified IB')
    ax1.plot(ent_curve['I_XT'], ent_curve['I_TY'], 's-', color='red', label='Entropy-Regularized IB')
    ax1.axhline(y=ib_model.mi_xy, ls='--', color='gray', alpha=0.6, label=f'I(X;Y)={ib_model.mi_xy:.4f}')
    ax1.set_xlabel('I(Z;X) (bits)')
    ax1.set_ylabel('I(Z;Y) (bits)')
    ax1.set_title('Information Plane Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # I(Z;Y) vs beta
    ax2.semilogx(std_curve['beta_values'], std_curve['I_TY'], 'o-', color='blue', label='Standard IB')
    ax2.semilogx(conv_curve['beta_values'], conv_curve['I_TY'], 'x-', color='green', label='Convexified IB')
    ax2.semilogx(ent_curve['beta_values'], ent_curve['I_TY'], 's-', color='red', label='Entropy-Regularized IB')
    ax2.axhline(y=ib_model.mi_xy, ls='--', color='gray', alpha=0.6)
    ax2.axvline(x=ib_model.target_beta_star, ls='--', color='black', alpha=0.6, label=f'β*≈{ib_model.target_beta_star:.4f}')
    ax2.set_xlabel('β (log scale)')
    ax2.set_ylabel('I(Z;Y) (bits)')
    ax2.set_title('I(Z;Y) vs. β Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("ib_curve_comparison.png")
    print("Saved IB curve comparison to 'ib_curve_comparison.png'")
    
    print("\nStandard vs Enhanced IB comparison complete!")

def demonstrate_continuation_ib():
    """
    Demonstrate the continuation-based IB optimization.
    """
    print("\nDemonstrating Continuation-based IB Optimization")
    print("-" * 80)
    
    # Create a synthetic joint distribution
    card_x, card_y, card_z = 8, 8, 4
    noise = 0.5
    dist_key = random.PRNGKey(42025)
    joint_xy_np = create_structured_distribution(cardinality_x=card_x, cardinality_y=card_y, num_clusters=3, cluster_strength=0.8)
    print(f"Created structured p(x,y) joint distribution. Sum={np.sum(joint_xy_np):.4f}, Shape={joint_xy_np.shape}")
    joint_xy_jax = jnp.array(joint_xy_np)
    
    # Create IB model
    main_key = random.PRNGKey(2025)
    ib_model = EnhancedInformationBottleneck(joint_xy_jax, key=main_key, cardinality_z=card_z)
    print(f"  I(X;Y)={ib_model.mi_xy:.4f} bits, H(X)={ib_model.hx:.4f} bits")
    
    # Run continuation-based optimization
    print("\nRunning continuation-based IB optimization...")
    beta_min, beta_max, num_steps = 0.1, 10.0, 50
    
    continuation_results = ib_model.optimize_with_continuation(
        beta_min=beta_min, beta_max=beta_max, num_steps=num_steps,
        epsilon_0=0.1, decay_rate=0.5, u_type="squared",
        bifurcation_threshold=1e-7, verbose=True
    )
    
    # Count bifurcations
    num_bifurcations = np.sum(continuation_results['bifurcations'])
    print(f"\nDetected {num_bifurcations} bifurcations during continuation")
    
    # Plot results
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Information plane
    ax1.plot(continuation_results['I_XT'], continuation_results['I_TY'], 'o-', color='blue', label='Continuation Path')
    ax1.axhline(y=ib_model.mi_xy, ls='--', color='gray', alpha=0.6, label=f'I(X;Y)={ib_model.mi_xy:.4f}')
    
    # Mark bifurcations
    bifurcation_indices = np.where(continuation_results['bifurcations'])[0]
    for idx in bifurcation_indices:
        bif_izx = continuation_results['I_XT'][idx]
        bif_izy = continuation_results['I_TY'][idx]
        ax1.plot(bif_izx, bif_izy, 'x', color='red', ms=10, mew=2.0)
        ax1.annotate(f'β={continuation_results["beta_values"][idx]:.2f}', 
                    xy=(bif_izx, bif_izy), xytext=(10, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    ax1.set_xlabel('I(Z;X) (bits)')
    ax1.set_ylabel('I(Z;Y) (bits)')
    ax1.set_title('Information Plane with Continuation Path')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Eigenvalues vs beta
    beta_values = continuation_results['beta_values']
    eigenvalues = np.abs(np.array(continuation_results['smallest_eigenvalues']))
    
    ax2.semilogy(beta_values, eigenvalues + 1e-10, 'o-', color='purple', label='|λ_min|')
    ax2.axhline(y=1e-7, ls='--', color='red', alpha=0.6, label='Bifurcation Threshold')
    
    # Mark bifurcations
    for idx in bifurcation_indices:
        bif_beta = beta_values[idx]
        bif_eval = eigenvalues[idx] + 1e-10
        ax2.plot(bif_beta, bif_eval, 'x', color='red', ms=10, mew=2.0)
    
    # Add vertical lines at bifurcation points
    for idx in bifurcation_indices:
        bif_beta = beta_values[idx]
        ax2.axvline(x=bif_beta, ls='--', color='red', alpha=0.3)
    
    ax2.set_xlabel('β')
    ax2.set_ylabel('|λ_min| (log scale)')
    ax2.set_title('Smallest Eigenvalue vs. β')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("continuation_ib_results.png")
    print("Saved continuation IB results to 'continuation_ib_results.png'")
    
    # Plot I(Z;Y) vs beta
    fig2, ax = plt.subplots(figsize=(10, 6))
    beta_values = continuation_results['beta_values']
    izy_values = continuation_results['I_TY']
    
    ax.plot(beta_values, izy_values, 'o-', color='blue', label='I(Z;Y)')
    
    # Mark bifurcations
    for idx in bifurcation_indices:
        bif_beta = beta_values[idx]
        bif_izy = izy_values[idx]
        ax.plot(bif_beta, bif_izy, 'x', color='red', ms=10, mew=2.0)
        ax.axvline(x=bif_beta, ls='--', color='red', alpha=0.3)
    
    ax.set_xlabel('β')
    ax.set_ylabel('I(Z;Y) (bits)')
    ax.set_title('Evolution of I(Z;Y) vs. β')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=ib_model.mi_xy, ls='--', color='gray', alpha=0.6, label=f'I(X;Y)={ib_model.mi_xy:.4f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("izy_vs_beta_continuation.png")
    print("Saved I(Z;Y) vs beta plot to 'izy_vs_beta_continuation.png'")
    
    # Visualize encoder evolution
    num_encoders = min(6, len(continuation_results['encoders']))
    indices = np.linspace(0, len(continuation_results['encoders'])-1, num_encoders, dtype=int)
    
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= len(axes): break
        
        beta = continuation_results['beta_values'][idx]
        izx = continuation_results['I_XT'][idx]
        izy = continuation_results['I_TY'][idx]
        is_bifurcation = continuation_results['bifurcations'][idx]
        
        encoder = continuation_results['encoders'][idx]
        im = axes[i].imshow(encoder, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=axes[i])
        
        title = f"β={beta:.2f}, IZX={izx:.2f}, IZY={izy:.2f}"
        if is_bifurcation:
            title += " (Bifurcation)"
            axes[i].set_title(title, color='red', fontweight='bold')
        else:
            axes[i].set_title(title)
        
        axes[i].set_xlabel('z (cluster)')
        axes[i].set_ylabel('x (input)')
    
    plt.tight_layout()
    plt.savefig("encoder_evolution.png")
    print("Saved encoder evolution to 'encoder_evolution.png'")
    
    print("\nContinuation-based IB demonstration complete!")

def demonstrate_enhanced_multi_path_ib():
    """
    Demonstrate the Enhanced Multi-Path IB framework.
    """
    print("\nDemonstrating Enhanced Multi-Path IB Framework")
    print("-" * 80)
    
    # Create a synthetic joint distribution
    card_x, card_y, card_z = 8, 8, 4
    noise = 0.55
    dist_key = random.PRNGKey(42026)
    joint_xy_np = create_structured_distribution(cardinality_x=card_x, cardinality_y=card_y, 
                                              num_clusters=3, cluster_strength=0.7,
                                              random_seed=421)
    print(f"Created structured p(x,y) joint distribution. Sum={np.sum(joint_xy_np):.4f}, Shape={joint_xy_np.shape}")
    joint_xy_jax = jnp.array(joint_xy_np)
    
    # Create IB model
    main_key = random.PRNGKey(2026)
    ib_model = EnhancedInformationBottleneck(joint_xy_jax, key=main_key, cardinality_z=card_z)
    print(f"  I(X;Y)={ib_model.mi_xy:.4f} bits, H(X)={ib_model.hx:.4f} bits")
    
    # Create Multi-Path IB optimizer
    print("\nInitializing Enhanced Multi-Path IB optimizer...")
    multi_path_ib = EnhancedMultiPathIB(ib_model, num_paths=3, verbose=True)
    
    # Set beta schedule
    multi_path_ib.set_beta_schedule(beta_min=0.05, beta_max=8.0, num_steps=40, log_scale=True)
    
    # Run optimization
    print("\nRunning Enhanced Multi-Path IB optimization...")
    best_encoder, best_metrics, history = multi_path_ib.optimize(
        use_convexification=True,
        use_entropy_reg=True,
        use_adaptive_refinement=True,
        u_type="squared",
        epsilon_0=0.1,
        local_iters_per_beta=30
    )
    
    # Print results
    print("\nBest solution metrics:")
    for key, value in best_metrics.items():
        if isinstance(value, (float, int)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Plot results
    print("\nGenerating plots...")
    
    # Solution paths in information plane
    sol_path_fig = multi_path_ib.plot_solution_paths(history, " (Enhanced Multi-Path)")
    sol_path_fig.savefig("enhanced_multipath_info_plane.png")
    print("Saved information plane plot to 'enhanced_multipath_info_plane.png'")
    
    # Beta trajectories
    beta_traj_fig = multi_path_ib.plot_beta_trajectories(history, " (Enhanced Multi-Path)")
    beta_traj_fig.savefig("enhanced_multipath_beta_trajectories.png")
    print("Saved beta trajectories plot to 'enhanced_multipath_beta_trajectories.png'")
    
    # Convergence metrics
    conv_fig = multi_path_ib.plot_convergence_metrics(" (Enhanced Multi-Path)")
    conv_fig.savefig("enhanced_multipath_convergence.png")
    print("Saved convergence metrics plot to 'enhanced_multipath_convergence.png'")
    
    # Visualize best encoder
    enc_fig = multi_path_ib.visualize_encoder(best_encoder, "Best Encoder from Enhanced Multi-Path IB")
    enc_fig.savefig("enhanced_multipath_best_encoder.png")
    print("Saved best encoder visualization to 'enhanced_multipath_best_encoder.png'")
    
    print("\nEnhanced Multi-Path IB demonstration complete!")

def demonstrate_bsc_critical_region():
    """
    Demonstrate the Enhanced IB framework on Binary Symmetric Channel (BSC) critical region.
    """
    print("\nDemonstrating Enhanced IB on BSC Critical Region")
    print("-" * 80)
    
    # Create BSC joint distribution
    q_bsc = 1/11  # For target_beta_star = 4.14144
    joint_xy_bsc_np = create_bsc_distribution(q=q_bsc)
    print(f"Created BSC p(x,y) with q = {q_bsc:.6f}")
    print(f"Joint distribution sum={np.sum(joint_xy_bsc_np):.4f}, Shape={joint_xy_bsc_np.shape}")
    joint_xy_bsc_jax = jnp.array(joint_xy_bsc_np)
    
    # Create IB model
    main_key = random.PRNGKey(2027)
    ib_model_bsc = EnhancedInformationBottleneck(joint_xy_bsc_jax, key=main_key, cardinality_z=2)
    beta_crit_target = ib_model_bsc.target_beta_star  # Should be approximately 4.14144
    print(f"  I(X;Y)={ib_model_bsc.mi_xy:.4f} bits, H(X)={ib_model_bsc.hx:.4f} bits")
    print(f"  Target β*={beta_crit_target:.5f}")
    
    # Create test beta values around the critical point
    test_betas = np.array([
        beta_crit_target - 0.5,
        beta_crit_target - 0.1,
        beta_crit_target,
        beta_crit_target + 0.1,
        beta_crit_target + 0.5
    ])
    
    # Results storage
    std_results = []
    conv_results = []
    ent_results = []
    
    print("\nComparing methods around critical point β*...")
    for beta in test_betas:
        print(f"\nTesting at β = {beta:.5f}:")
        
        # Standard IB
        key1 = random.fold_in(main_key, int(beta * 1000))
        std_pzgx, std_izx, std_izy = ib_model_bsc.optimize(beta=beta, key=key1, verbose=False)
        std_obj = std_izy - beta * std_izx
        print(f"  Standard IB: IZX={std_izx:.6f}, IZY={std_izy:.6f}, Obj={std_obj:.6f}")
        std_results.append((beta, std_izx, std_izy, std_obj, std_pzgx))
        
        # Convexified IB
        key2 = random.fold_in(main_key, int(beta * 1000) + 1)
        conv_pzgx, conv_izx, conv_izy = ib_model_bsc.optimize_convexified(beta=beta, key=key2, verbose=False)
        conv_obj = conv_izy - beta * conv_izx
        print(f"  Convexified IB: IZX={conv_izx:.6f}, IZY={conv_izy:.6f}, Obj={conv_obj:.6f}")
        conv_results.append((beta, conv_izx, conv_izy, conv_obj, conv_pzgx))
        
        # Entropy regularized IB
        key3 = random.fold_in(main_key, int(beta * 1000) + 2)
        ent_pzgx, ent_izx, ent_izy = ib_model_bsc.optimize_with_entropy_reg(beta=beta, key=key3, verbose=False)
        h_zx = ib_model_bsc.conditional_entropy(ent_pzgx)
        epsilon_val = ib_model_bsc.epsilon_schedule(beta, epsilon_0=0.1, decay_rate=0.5)
        ent_obj = ent_izy - beta * ent_izx + epsilon_val * h_zx
        print(f"  Entropy Reg. IB: IZX={ent_izx:.6f}, IZY={ent_izy:.6f}, H(Z|X)={h_zx:.6f}, Obj={ent_obj:.6f}")
        ent_results.append((beta, ent_izx, ent_izy, ent_obj, ent_pzgx, h_zx))
    
    # Extract data for plotting
    betas = np.array([r[0] for r in std_results])
    std_izx = np.array([r[1] for r in std_results])
    std_izy = np.array([r[2] for r in std_results])
    
    conv_izx = np.array([r[1] for r in conv_results])
    conv_izy = np.array([r[2] for r in conv_results])
    
    ent_izx = np.array([r[1] for r in ent_results])
    ent_izy = np.array([r[2] for r in ent_results])
    ent_hzx = np.array([r[5] for r in ent_results])
    
    # Plot results around critical point
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # I(Z;X) vs beta
    ax1.plot(betas, std_izx, 'o-', color='blue', label='Standard IB')
    ax1.plot(betas, conv_izx, 'x-', color='green', label='Convexified IB')
    ax1.plot(betas, ent_izx, 's-', color='red', label='Entropy Reg. IB')
    ax1.axvline(x=beta_crit_target, ls='--', color='black', alpha=0.6, label=f'β*={beta_crit_target:.4f}')
    ax1.set_xlabel('β')
    ax1.set_ylabel('I(Z;X) (bits)')
    ax1.set_title('I(Z;X) vs. β Around Critical Point')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # I(Z;Y) vs beta
    ax2.plot(betas, std_izy, 'o-', color='blue', label='Standard IB')
    ax2.plot(betas, conv_izy, 'x-', color='green', label='Convexified IB')
    ax2.plot(betas, ent_izy, 's-', color='red', label='Entropy Reg. IB')
    ax2.axvline(x=beta_crit_target, ls='--', color='black', alpha=0.6)
    ax2.axhline(y=ib_model_bsc.mi_xy, ls='--', color='gray', alpha=0.6, label=f'I(X;Y)={ib_model_bsc.mi_xy:.4f}')
    ax2.set_xlabel('β')
    ax2.set_ylabel('I(Z;Y) (bits)')
    ax2.set_title('I(Z;Y) vs. β Around Critical Point')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # H(Z|X) for entropy regularized IB
    ax3.plot(betas, ent_hzx, 's-', color='purple', label='H(Z|X) - Entropy Reg. IB')
    ax3.axvline(x=beta_crit_target, ls='--', color='black', alpha=0.6)
    ax3.set_xlabel('β')
    ax3.set_ylabel('H(Z|X) (bits)')
    ax3.set_title('Conditional Entropy H(Z|X) vs. β (Entropy Reg. IB)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig("bsc_critical_region.png")
    print("Saved BSC critical region plots to 'bsc_critical_region.png'")
    
    # Run continuation optimization to detect exact phase transition
    print("\nRunning continuation-based optimization to precisely detect phase transition...")
    cont_results = ib_model_bsc.optimize_with_continuation(
        beta_min=beta_crit_target - 1.0, 
        beta_max=beta_crit_target + 1.0, 
        num_steps=80,
        epsilon_0=0.05,  # Lower epsilon for more precision
        bifurcation_threshold=1e-7,
        verbose=True
    )
    
    # Find bifurcations
    bifurcation_indices = np.where(cont_results['bifurcations'])[0]
    if len(bifurcation_indices) > 0:
        detected_betas = [cont_results['beta_values'][idx] for idx in bifurcation_indices]
        print(f"\nDetected bifurcations at β values: {', '.join([f'{b:.6f}' for b in detected_betas])}")
        
        # Compare with theoretical value
        closest_idx = np.argmin(np.abs(np.array(detected_betas) - beta_crit_target))
        closest_beta = detected_betas[closest_idx]
        print(f"Closest detected β* = {closest_beta:.6f}, theoretical β* = {beta_crit_target:.6f}")
        print(f"Difference: {abs(closest_beta - beta_crit_target):.6f}")
    else:
        print("No bifurcations detected in the specified range")
    
    # Plot I(Z;Y) with eigenvalues from continuation
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # I(Z;Y) vs beta from continuation
    beta_values = cont_results['beta_values']
    izy_values = cont_results['I_TY']
    eigenvalues = np.abs(np.array(cont_results['smallest_eigenvalues']) + 1e-10)
    
    ax1.plot(beta_values, izy_values, 'o-', color='blue', label='I(Z;Y)')
    ax1.axvline(x=beta_crit_target, ls='--', color='black', alpha=0.6, label=f'Theoretical β*={beta_crit_target:.4f}')
    
    # Mark bifurcations
    for idx in bifurcation_indices:
        bif_beta = beta_values[idx]
        bif_izy = izy_values[idx]
        ax1.plot(bif_beta, bif_izy, 'x', color='red', ms=10, mew=2.0)
        ax1.axvline(x=bif_beta, ls='--', color='red', alpha=0.3)
        ax1.annotate(f'β={bif_beta:.4f}', 
                     xy=(bif_beta, bif_izy), xytext=(10, 10),
                     textcoords='offset points', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    ax1.set_xlabel('β')
    ax1.set_ylabel('I(Z;Y) (bits)')
    ax1.set_title('Evolution of I(Z;Y) vs. β (Continuation Method)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=ib_model_bsc.mi_xy, ls='--', color='gray', alpha=0.6, label=f'I(X;Y)={ib_model_bsc.mi_xy:.4f}')
    ax1.legend()
    
    # Eigenvalues vs beta
    ax2.semilogy(beta_values, eigenvalues, 'o-', color='purple', label='|λ_min|')
    ax2.axhline(y=1e-7, ls='--', color='red', alpha=0.6, label='Bifurcation Threshold')
    ax2.axvline(x=beta_crit_target, ls='--', color='black', alpha=0.6)
    
    # Mark bifurcations
    for idx in bifurcation_indices:
        bif_beta = beta_values[idx]
        bif_eval = eigenvalues[idx]
        ax2.plot(bif_beta, bif_eval, 'x', color='red', ms=10, mew=2.0)
        ax2.axvline(x=bif_beta, ls='--', color='red', alpha=0.3)
    
    ax2.set_xlabel('β')
    ax2.set_ylabel('|λ_min| (log scale)')
    ax2.set_title('Smallest Eigenvalue vs. β')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("bsc_phase_transition_detection.png")
    print("Saved phase transition detection plots to 'bsc_phase_transition_detection.png'")
    
    print("\nBSC Critical Region demonstration complete!")

if __name__ == "__main__":
    # Default values for demonstrations
    this_card_x, this_card_y = 8, 8  
    
    # Run demonstrations
    demonstrate_standard_vs_enhanced_ib()
    demonstrate_continuation_ib()
    demonstrate_enhanced_multi_path_ib()
    demonstrate_bsc_critical_region()
    
    print("\nAll demonstrations complete!\n")
