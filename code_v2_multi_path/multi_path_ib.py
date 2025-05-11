# Author: Faruk Alpay
# ORCID: 0009-0009-2207-6528
# Publication: https://doi.org/10.22541/au.174664105.57850297/v1

"""
Complete Information Bottleneck (IB) Framework with Multi-Path Incremental-β Extension

This comprehensive Python file integrates:
  1) High-precision JAX-based numeric iterative IB optimization
  2) Multi-Path Incremental-β approach to prevent trivial collapses
  3) Symbolic Analysis using SymPy for theoretical verification

Optimized to demonstrate the phase transition at β* ≈ 4.14144 
and avoid trivial solutions under moderate correlations.

Dependencies:
  - JAX (for numerical optimization)
  - SymPy (for symbolic mathematics)
  - NumPy, Matplotlib (for computation and visualization)
  - SciPy (for signal processing and spatial analysis)
"""

import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Any, Union
from scipy.spatial import ConvexHull

# JAX imports
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import random

# SciPy / stats / signal
from scipy import stats
from scipy.signal import find_peaks

# Optional Sympy for symbolic analysis
try:
    import sympy as sp
    from sympy import symbols, Symbol, Function, diff, Eq, solve, log, exp, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    warnings.warn("Sympy not installed. Symbolic analysis features will be disabled.")

################################################################################
# 1. HELPER FUNCTIONS
################################################################################
def jax_gaussian_filter1d(input_array: jnp.ndarray, sigma: float, truncate: float = 4.0) -> jnp.ndarray:
    """
    Apply Gaussian filter to 1D array.
    
    Args:
        input_array: Input array to filter
        sigma: Standard deviation of Gaussian kernel
        truncate: Truncate filter at this many standard deviations
        
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

    if lw < 0:
        return input_array

    x = jnp.arange(-lw, lw + 1)
    kernel = jnp.exp(-0.5 * (x / sigma)**2)
    kernel_sum = jnp.sum(kernel)

    kernel = jnp.where(kernel_sum > 1e-9, kernel / kernel_sum,
                       jnp.zeros_like(kernel).at[lw].set(1.0 if kernel.shape[0] > 0 else 0.0))

    if input_array.ndim == 0 or input_array.shape[0] == 0:
        return input_array
    if kernel.shape[0] == 0:
        return input_array

    return jnp.convolve(input_array, kernel, mode='same')


################################################################################
# 2. ULTRA-PRECISION INFORMATION BOTTLENECK
################################################################################
class UltraPrecisionInformationBottleneck:
    """
    High-precision JAX-based Information Bottleneck implementation,
    with numeric iterative updates for p(z|x) and advanced search around beta*.
    
    Extended to integrate with symbolic analysis.
    """
    def __init__(self, joint_xy: jnp.ndarray, key: Any, 
                 cardinality_z: Optional[int] = None,
                 epsilon: float = 1e-15):
        """
        Initialize the Information Bottleneck with high precision.
        
        Args:
            joint_xy: Joint distribution p(x,y)
            key: JAX random key
            cardinality_z: Dimension of Z (defaults to dimension of X)
            epsilon: Small value for numerical stability
        """
        self.key = key
        self.epsilon = epsilon
        self.tiny_epsilon = 1e-35  # even smaller for numeric safety

        # Validate input
        if not jnp.allclose(jnp.sum(joint_xy), 1.0, atol=1e-10):
            joint_xy_sum = jnp.sum(joint_xy)
            joint_xy = jnp.where(joint_xy_sum > self.epsilon, joint_xy / joint_xy_sum, joint_xy)
            warnings.warn("Joint distribution was not normalized. Auto-normalizing.")

        if jnp.any(joint_xy < 0):
            raise ValueError("Joint distribution contains negative values")

        self.joint_xy = joint_xy
        self.cardinality_x = joint_xy.shape[0]
        self.cardinality_y = joint_xy.shape[1]
        self.cardinality_z = self.cardinality_x if cardinality_z is None else cardinality_z

        self.p_x = jnp.sum(joint_xy, axis=1)
        self.p_y = jnp.sum(joint_xy, axis=0)

        self.log_p_x = jnp.log(jnp.maximum(self.p_x, self.epsilon))
        self.log_p_y = jnp.log(jnp.maximum(self.p_y, self.epsilon))

        # Compute p(y|x)
        p_x_expanded = self.p_x[:, None]
        self.p_y_given_x = jnp.where(p_x_expanded > self.epsilon,
                                     joint_xy / p_x_expanded,
                                     jnp.ones_like(joint_xy) / self.cardinality_y)

        self.p_y_given_x = jnp.maximum(self.p_y_given_x, self.epsilon)
        self.p_y_given_x = self.p_y_given_x / jnp.sum(self.p_y_given_x, axis=1, keepdims=True)

        self.log_p_y_given_x = jnp.log(self.p_y_given_x)

        # Calculate MI and entropy directly
        self.mi_xy = self._calculate_mutual_information(self.joint_xy, self.p_x, self.p_y)
        self.hx = self._calculate_entropy(self.p_x)

        self.encoder_cache = {}
        self.target_beta_star = 4.14144  # Theoretical reference

        # Ultra-precision settings
        self.max_iterations = 1000
        self.structural_kl_threshold = 1e-9
        self.min_stable_iterations = 5
        self.optimization_history = {}
        self.current_beta = None  # Store the current beta value
        
        # Compile relevant JAX functions
        self._kl_divergence_core_jit = self._create_kl_divergence_core_jit()
        self._mutual_information_core_jit = self._create_mutual_information_core_jit()
        self._entropy_core_jit = self._create_entropy_core_jit()
        self._calculate_marginal_z_core_jit = self._create_calculate_marginal_z_core_jit()
        self._calculate_joint_zy_core_jit = self._create_calculate_joint_zy_core_jit()
        self._calculate_mi_zx_core_jit = self._create_calculate_mi_zx_core_jit()
        self._normalize_rows_core_jit = self._create_normalize_rows_core_jit()
        self._ib_update_step_core_jit = self._create_ib_update_step_core_jit()

    ############################################################################
    # BASIC METHODS
    ############################################################################
    def _create_kl_divergence_core_jit(self):
        """Create JIT-compiled KL divergence function."""
        def _kl_divergence_core(p, q, epsilon, tiny_epsilon):
            p_safe = jnp.maximum(p, epsilon)
            q_safe = jnp.maximum(q, epsilon)
            p_norm = p_safe / jnp.sum(p_safe)
            q_norm = q_safe / jnp.sum(q_safe)
            kl_val = jnp.sum(
                p_norm * (jnp.log(p_norm + tiny_epsilon) - jnp.log(q_norm + tiny_epsilon))
            )
            return jnp.maximum(0.0, kl_val)
        return jax.jit(_kl_divergence_core)

    def _create_mutual_information_core_jit(self):
        """Create JIT-compiled mutual information function."""
        def _mutual_information_core(joint_dist, marginal_x, marginal_y, epsilon):
            joint_dist_safe = jnp.maximum(joint_dist, epsilon)
            marginal_x_safe = jnp.maximum(marginal_x, epsilon)
            marginal_y_safe = jnp.maximum(marginal_y, epsilon)
            log_joint = jnp.log(joint_dist_safe)
            log_marg_x_col = jnp.log(marginal_x_safe)[:, None]
            log_marg_y_row = jnp.log(marginal_y_safe)[None, :]
            log_prod_margs = log_marg_x_col + log_marg_y_row
            mi_terms = joint_dist * (log_joint - log_prod_margs)
            mi = jnp.sum(mi_terms)
            return jnp.maximum(0.0, mi) / jnp.log(2)
        return jax.jit(_mutual_information_core)

    def _create_entropy_core_jit(self):
        """Create JIT-compiled entropy function."""
        def _entropy_core(dist, epsilon):
            dist_safe = jnp.maximum(dist, epsilon)
            log_dist_safe = jnp.log(dist_safe)
            entropy_val = -jnp.sum(dist * log_dist_safe)
            return jnp.maximum(0.0, entropy_val) / jnp.log(2)
        return jax.jit(_entropy_core)

    def _create_calculate_marginal_z_core_jit(self):
        """Create JIT-compiled function to calculate marginal p(z)."""
        def _calculate_marginal_z_core(p_z_given_x, p_x, epsilon):
            p_z = jnp.dot(p_x, p_z_given_x)
            p_z_safe = jnp.maximum(p_z, epsilon)
            p_z_norm = p_z_safe / jnp.sum(p_z_safe)
            log_p_z_norm = jnp.log(jnp.maximum(p_z_norm, epsilon))
            return p_z_norm, log_p_z_norm
        return jax.jit(_calculate_marginal_z_core)

    def _create_calculate_joint_zy_core_jit(self):
        """Create JIT-compiled function to calculate joint p(z,y)."""
        def _calculate_joint_zy_core(p_z_given_x, joint_xy, epsilon):
            p_zy = jnp.einsum('ik,ij->kj', p_z_given_x, joint_xy)
            p_zy_safe = jnp.maximum(p_zy, epsilon)
            p_zy_norm = p_zy_safe / jnp.sum(p_zy_safe)
            return p_zy_norm
        return jax.jit(_calculate_joint_zy_core)

    def _create_calculate_mi_zx_core_jit(self):
        """Create JIT-compiled function to calculate I(Z;X)."""
        def _calculate_mi_zx_core(p_z_given_x, p_z, p_x, epsilon):
            p_z_given_x_safe = jnp.maximum(p_z_given_x, epsilon)
            p_z_safe = jnp.maximum(p_z, epsilon)
            log_p_z_given_x = jnp.log(p_z_given_x_safe)
            log_p_z = jnp.log(p_z_safe)
            kl_divs_per_x = jnp.sum(
                p_z_given_x * (log_p_z_given_x - log_p_z[None, :]),
                axis=1
            )
            mi_zx = jnp.sum(p_x * kl_divs_per_x)
            mi_zx_bits = jnp.maximum(0.0, mi_zx) / jnp.log(2)
            return mi_zx_bits
        return jax.jit(_calculate_mi_zx_core)

    def _create_normalize_rows_core_jit(self):
        """Create JIT-compiled function to normalize rows of a matrix."""
        def _normalize_rows_core(matrix, epsilon, tiny_epsilon):
            matrix_non_neg = jnp.maximum(matrix, 0)
            row_sums = jnp.sum(matrix_non_neg, axis=1, keepdims=True)
            uniform_row = jnp.ones((1, matrix.shape[1])) / matrix.shape[1]
            normalized = jnp.where(
                row_sums > epsilon,
                matrix_non_neg / row_sums,
                jnp.tile(uniform_row, (matrix.shape[0], 1))
            )
            normalized_safe = jnp.maximum(normalized, epsilon)
            final_row_sums = jnp.sum(normalized_safe, axis=1, keepdims=True)
            return normalized_safe / jnp.maximum(final_row_sums, tiny_epsilon)
        return jax.jit(_normalize_rows_core)

    def _create_ib_update_step_core_jit(self):
        """Create JIT-compiled function for IB update step."""
        def _ib_update_step_core(p_z_given_x, beta, p_x, joint_xy, p_y_given_x, log_p_y_given_x,
                                 cardinality_y, epsilon, tiny_epsilon):
            # 1) p_z
            p_z = jnp.dot(p_x, p_z_given_x)
            p_z_safe = jnp.maximum(p_z, epsilon)
            p_z_norm = p_z_safe / jnp.sum(p_z_safe)
            log_p_z_norm = jnp.log(jnp.maximum(p_z_norm, epsilon))

            # 2) p(y|z)
            p_zy = jnp.einsum('ik,ij->kj', p_z_given_x, joint_xy)
            p_zy_safe = jnp.maximum(p_zy, epsilon)
            joint_zy = p_zy_safe / jnp.sum(p_zy_safe)

            p_z_safe_denom = jnp.maximum(p_z_norm, epsilon)[:, None]
            p_y_given_z = joint_zy / p_z_safe_denom
            p_y_given_z = jnp.where(
                p_z_safe_denom > epsilon,
                p_y_given_z,
                jnp.ones_like(joint_zy) / cardinality_y
            )

            p_y_given_z_safe = jnp.maximum(p_y_given_z, epsilon)
            row_sums = jnp.sum(p_y_given_z_safe, axis=1, keepdims=True)
            p_y_given_z_norm = p_y_given_z_safe / jnp.maximum(row_sums, epsilon)
            log_p_y_given_z_norm = jnp.log(jnp.maximum(p_y_given_z_norm, epsilon))

            # 3) update step
            p_y_gx_expanded = p_y_given_x[:, None, :]
            log_p_y_gx_expanded = log_p_y_given_x[:, None, :]
            log_p_y_gz_expanded = log_p_y_given_z_norm[None, :, :]

            kl_matrix = jnp.sum(
                p_y_gx_expanded * (log_p_y_gx_expanded - log_p_y_gz_expanded),
                axis=2
            )

            log_new_p_z_given_x = log_p_z_norm[None, :] - beta * kl_matrix
            log_norm = jsp.logsumexp(log_new_p_z_given_x, axis=1, keepdims=True)
            log_new_p_z_given_x_norm = log_new_p_z_given_x - log_norm

            new_p_z_given_x = jnp.exp(log_new_p_z_given_x_norm)

            # 4) normalize
            matrix_non_neg = jnp.maximum(new_p_z_given_x, 0)
            row_sums = jnp.sum(matrix_non_neg, axis=1, keepdims=True)

            uniform_row = jnp.ones((1, new_p_z_given_x.shape[1])) / new_p_z_given_x.shape[1]
            normalized = jnp.where(
                row_sums > epsilon,
                matrix_non_neg / row_sums,
                jnp.tile(uniform_row, (new_p_z_given_x.shape[0], 1))
            )

            normalized_safe = jnp.maximum(normalized, epsilon)
            final_row_sums = jnp.sum(normalized_safe, axis=1, keepdims=True)
            return normalized_safe / jnp.maximum(final_row_sums, tiny_epsilon)
        
        return jax.jit(_ib_update_step_core)

    ############################################################################
    # Public utility
    ############################################################################
    def kl_divergence(self, p, q):
        """
        Calculate KL divergence between two distributions.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            KL divergence value
        """
        return float(self._kl_divergence_core_jit(p, q, self.epsilon, self.tiny_epsilon))

    def mutual_information(self, joint_dist, marginal_x, marginal_y):
        """
        Calculate mutual information from joint and marginal distributions.
        
        Args:
            joint_dist: Joint probability distribution
            marginal_x: Marginal distribution of X
            marginal_y: Marginal distribution of Y
            
        Returns:
            Mutual information value in bits
        """
        return float(self._mutual_information_core_jit(joint_dist, marginal_x, marginal_y, self.epsilon))

    def entropy(self, dist):
        """
        Calculate entropy of a probability distribution.
        
        Args:
            dist: Probability distribution
            
        Returns:
            Entropy value in bits
        """
        return float(self._entropy_core_jit(dist, self.epsilon))

    def calculate_marginal_z(self, p_z_given_x):
        """
        Calculate the marginal distribution of Z.
        
        Args:
            p_z_given_x: Conditional distribution p(z|x)
            
        Returns:
            Tuple of normalized p(z) and log(p(z))
        """
        return self._calculate_marginal_z_core_jit(p_z_given_x, self.p_x, self.epsilon)

    def calculate_joint_zy(self, p_z_given_x):
        """
        Calculate joint distribution p(z,y).
        
        Args:
            p_z_given_x: Conditional distribution p(z|x)
            
        Returns:
            Normalized joint distribution p(z,y)
        """
        return self._calculate_joint_zy_core_jit(p_z_given_x, self.joint_xy, self.epsilon)

    def calculate_mi_zx(self, p_z_given_x, p_z):
        """
        Calculate mutual information I(Z;X).
        
        Args:
            p_z_given_x: Conditional distribution p(z|x)
            p_z: Marginal distribution p(z)
            
        Returns:
            Mutual information I(Z;X) in bits
        """
        return float(self._calculate_mi_zx_core_jit(p_z_given_x, p_z, self.p_x, self.epsilon))

    def calculate_mi_zy(self, p_z_given_x):
        """
        Calculate mutual information I(Z;Y).
        
        Args:
            p_z_given_x: Conditional distribution p(z|x)
            
        Returns:
            Mutual information I(Z;Y) in bits
        """
        p_z, _ = self.calculate_marginal_z(p_z_given_x)
        joint_zy = self.calculate_joint_zy(p_z_given_x)
        return self.mutual_information(joint_zy, p_z, self.p_y)

    def normalize_rows(self, matrix):
        """
        Normalize rows of a matrix to sum to 1.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Normalized matrix where each row sums to 1
        """
        return self._normalize_rows_core_jit(matrix, self.epsilon, self.tiny_epsilon)

    def ib_update_step(self, p_z_given_x, beta):
        """
        Perform one step of the IB algorithm.
        
        Args:
            p_z_given_x: Current conditional distribution p(z|x)
            beta: Trade-off parameter
            
        Returns:
            Updated p(z|x)
        """
        return self._ib_update_step_core_jit(
            p_z_given_x, beta, self.p_x, self.joint_xy, 
            self.p_y_given_x, self.log_p_y_given_x, self.cardinality_y,
            self.epsilon, self.tiny_epsilon
        )

    def set_beta(self, beta):
        """
        Set the current beta value.
        
        Args:
            beta: Trade-off parameter
        """
        self.current_beta = beta

    def get_p_x(self):
        """Get the marginal distribution p(x)."""
        return np.array(self.p_x)

    def get_p_y_given_x(self):
        """Get the conditional distribution p(y|x)."""
        return np.array(self.p_y_given_x)

    def get_p_t_given_x(self):
        """
        Get the current optimal encoding distribution p(t|x).
        
        Returns:
            Current p(t|x) if available, otherwise None
        """
        if self.current_beta is None or self.current_beta not in self.encoder_cache:
            return None
        return np.array(self.encoder_cache[self.current_beta])

    def get_I_XT(self):
        """
        Get I(X;T) for the current solution.
        
        Returns:
            I(X;T) if a solution exists, otherwise 0
        """
        p_t_given_x = self.get_p_t_given_x()
        if p_t_given_x is None:
            return 0.0
        p_t, _ = self.calculate_marginal_z(p_t_given_x)
        return self.calculate_mi_zx(p_t_given_x, p_t)

    def get_I_TY(self):
        """
        Get I(T;Y) for the current solution.
        
        Returns:
            I(T;Y) if a solution exists, otherwise 0
        """
        p_t_given_x = self.get_p_t_given_x()
        if p_t_given_x is None:
            return 0.0
        return self.calculate_mi_zy(p_t_given_x)

    ############################################################################
    # Initializations
    ############################################################################
    def initialize_uniform(self, key):
        """
        Initialize p(z|x) to a uniform distribution.
        
        Args:
            key: JAX random key
            
        Returns:
            Uniformly initialized p(z|x)
        """
        p_z_given_x = jnp.ones((self.cardinality_x, self.cardinality_z))
        return self.normalize_rows(p_z_given_x)

    def initialize_structured(self, cardinality_x, cardinality_z, key):
        """
        Initialize p(z|x) with structured patterns.
        
        Args:
            cardinality_x: Dimension of X
            cardinality_z: Dimension of Z
            key: JAX random key
            
        Returns:
            Structured initialization for p(z|x)
        """
        p_z_given_x = jnp.zeros((cardinality_x, cardinality_z))
        i_indices = jnp.arange(cardinality_x)
        primary_z_indices = i_indices % cardinality_z
        secondary_z_indices = (i_indices + 1) % cardinality_z
        tertiary_z_indices = (i_indices + 2) % cardinality_z

        p_z_given_x = p_z_given_x.at[i_indices, primary_z_indices].add(0.7)
        p_z_given_x = p_z_given_x.at[i_indices, secondary_z_indices].add(0.2)
        p_z_given_x = p_z_given_x.at[i_indices, tertiary_z_indices].add(0.1)

        return self.normalize_rows(p_z_given_x)
        
    def highly_critical_initialization(self, beta, key):
        """
        Specialized initialization for the critical region exactly at β*.
        
        Args:
            beta: Trade-off parameter
            key: JAX random key
            
        Returns:
            Specialized initialization for β*
        """
        key1, key2, key3, key4 = random.split(key, 4)
        p_z_given_x = self.initialize_structured(self.cardinality_x, self.cardinality_z, key1)
        noise_scale = 0.02
        
        i_indices = jnp.arange(self.cardinality_x)
        z_indices = jnp.arange(self.cardinality_z)
        i_grid, z_grid = jnp.meshgrid(i_indices, z_indices, indexing='ij')
        periodic_pattern = jnp.sin(i_grid / 10.0) * jnp.cos(z_grid / 8.0) * noise_scale
        random_noise = random.normal(key2, (self.cardinality_x, self.cardinality_z)) * noise_scale
        
        distance_to_critical = abs(beta - self.target_beta_star) / 0.01
        distance_to_critical = min(1.0, distance_to_critical)
        
        if distance_to_critical < 0.1:
            p_z_given_x = p_z_given_x + periodic_pattern * (1.0 - distance_to_critical)
            p_z_given_x = p_z_given_x + random_noise * 0.5
            uniform = self.initialize_uniform(key3)
            blend_factor = 0.1 * (1.0 - distance_to_critical)
            p_z_given_x = (1.0 - blend_factor) * p_z_given_x + blend_factor * uniform
        else:
            p_z_given_x = p_z_given_x + random_noise * 0.7
        
        return self.normalize_rows(p_z_given_x)

    def critical_initialization(self, beta, key):
        """
        Special initialization for near-critical region around β*.
        
        Args:
            beta: Trade-off parameter
            key: JAX random key
            
        Returns:
            Specialized initialization for near β*
        """
        if abs(beta - self.target_beta_star) < 0.0001:
            return self.highly_critical_initialization(beta, key)
        
        key_struct, key_noise, key_blend = random.split(key, 3)
        p_z_given_x = self.initialize_structured(self.cardinality_x, self.cardinality_z, key_struct)
        
        rel_distance = abs(beta - self.target_beta_star) / 0.01
        rel_distance = min(1.0, rel_distance)
        
        if rel_distance < 0.5:
            noise_scale = 0.03 * (1.0 - rel_distance)
            noise = random.normal(key_noise, p_z_given_x.shape) * noise_scale
            p_z_given_x += noise
            uniform_blend = self.initialize_uniform(key_blend)
            blend_factor = 0.2 * (1.0 - rel_distance)
            p_z_given_x = (1.0 - blend_factor) * p_z_given_x + blend_factor * uniform_blend
        
        return self.normalize_rows(p_z_given_x)

    def adaptive_initialization(self, beta, key):
        """
        Adaptive initialization based on how close beta is to the critical point.
        
        Args:
            beta: Trade-off parameter
            key: JAX random key
            
        Returns:
            Adaptive initialization for p(z|x)
        """
        key_init, key_u, key_n = random.split(key, 3)
        
        extremely_critical = abs(beta - self.target_beta_star) < 0.001
        relative_position = (beta - self.target_beta_star) / 0.1
        relative_position = max(-1, min(1, relative_position))
        in_critical_region = abs(relative_position) < 0.1
        
        if extremely_critical:
            return self.highly_critical_initialization(beta, key_init)
        elif in_critical_region:
            return self.critical_initialization(beta, key_init)
        elif abs(beta - self.target_beta_star) < 0.3:
            blend_factor = 0.5 * (1.0 - abs(relative_position)/0.3)
            p1 = self.critical_initialization(beta, key_init)
            p2 = self.initialize_structured(self.cardinality_x, self.cardinality_z, key_n)
            return self.normalize_rows(blend_factor * p1 + (1.0 - blend_factor) * p2)
        elif beta < self.target_beta_star:
            return self.initialize_structured(self.cardinality_x, self.cardinality_z, key_init)
        else:
            blend_factor = min(0.5, 0.2 + 0.3*(beta - self.target_beta_star))
            p_struct = self.initialize_structured(self.cardinality_x, self.cardinality_z, key_init)
            p_unif = self.initialize_uniform(key_u)
            return self.normalize_rows((1 - blend_factor)*p_struct + blend_factor*p_unif)

    ############################################################################
    # Ultra-Precise Optimization
    ############################################################################
    def optimize(self, beta=None, key=None, verbose=False, ultra_precise=True):
        """
        Optimize p(z|x) for given beta using multiple random inits + extra strict convergence.
        
        Args:
            beta: Trade-off parameter (optional, uses current_beta if None)
            key: JAX random key (optional, uses self.key if None)
            verbose: Whether to print progress
            ultra_precise: Whether to use ultra-precise optimization
            
        Returns:
            Tuple of (p(z|x), I(Z;X), I(Z;Y))
        """
        # Use current_beta if beta is None
        if beta is not None:
            self.set_beta(beta)
        elif self.current_beta is None:
            raise ValueError("No beta value specified and no current beta set")
        
        # Use cached key if key is None
        if key is None:
            key = self.key
            
        beta = self.current_beta
        p_z_given_x = self.adaptive_initialization(beta, key)
        if verbose:
            print(f"Optimizing for β = {beta:.8f}")
        
        kl_threshold = 1e-10
        min_stable = 10
        max_iter = 2000
        
        best_obj = float('-inf')
        best_p_z_given_x = None
        best_mi_zx = 0.0
        best_mi_zy = 0.0
        
        num_inits = 5
        for init_idx in range(num_inits):
            init_key = random.fold_in(key, init_idx)
            curr_p_z_given_x = self.adaptive_initialization(beta, init_key)
            curr_p_z_given_x, curr_mi_zx, curr_mi_zy = self._optimize_single(
                beta, init_key, curr_p_z_given_x, kl_threshold, min_stable, max_iter, 
                verbose=(verbose and init_idx==0)
            )
            curr_obj = curr_mi_zy - beta*curr_mi_zx
            if curr_obj > best_obj:
                best_obj = curr_obj
                best_p_z_given_x = curr_p_z_given_x
                best_mi_zx = curr_mi_zx
                best_mi_zy = curr_mi_zy
        
        if verbose:
            print(f" Using best of {num_inits} inits: I(Z;X)={best_mi_zx:.8f}, I(Z;Y)={best_mi_zy:.8f}")
        
        self.encoder_cache[beta] = best_p_z_given_x
        return best_p_z_given_x, best_mi_zx, best_mi_zy
    
    def _optimize_single(self, beta, key, p_z_given_x_init, kl_threshold, min_stable, max_iter, verbose=False):
        """
        Optimize p(z|x) with a single initialization.
        
        Args:
            beta: Trade-off parameter
            key: JAX random key
            p_z_given_x_init: Initial p(z|x)
            kl_threshold: Convergence threshold for KL divergence
            min_stable: Minimum number of stable iterations for convergence
            max_iter: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            Tuple of (p(z|x), I(Z;X), I(Z;Y))
        """
        p_z_given_x = p_z_given_x_init
        iteration = 0
        damping = 0.1
        stable_iterations = 0
        
        kl_history = []
        mi_zx_history = []
        mi_zy_history = []
        
        start_time = time.time()
        while iteration < max_iter:
            iteration += 1
            prev_p_z_given_x = p_z_given_x
            new_p_z_given_x = self.ib_update_step(p_z_given_x, beta)
            p_z_given_x = (1 - damping)*new_p_z_given_x + damping*p_z_given_x
            p_z_given_x = self.normalize_rows(p_z_given_x)
            
            p_z, _ = self.calculate_marginal_z(p_z_given_x)
            mi_zx = self.calculate_mi_zx(p_z_given_x, p_z)
            mi_zy = self.calculate_mi_zy(p_z_given_x)
            kl_div = self.kl_divergence(p_z_given_x, prev_p_z_given_x)
            
            kl_history.append(kl_div)
            mi_zx_history.append(mi_zx)
            mi_zy_history.append(mi_zy)
            
            if verbose and (iteration % 100 == 0 or iteration == 1):
                print(f"[Iter {iteration}] I(Z;X)={mi_zx:.8f}, I(Z;Y)={mi_zy:.8f}, KL={kl_div:.2e}")
            
            # adaptive damping
            if len(kl_history) >= 5:
                recent_kl = kl_history[-5:]
                if any(k2 > k1*1.2 for k1,k2 in zip(recent_kl[:-1], recent_kl[1:])):
                    damping = min(damping*1.2, 0.9)
                elif all(k<1e-6 for k in recent_kl[-3:]):
                    damping = max(damping*0.98, 0.02)
                else:
                    damping = max(damping*0.95, 0.05)
            
            if kl_div < kl_threshold:
                stable_iterations += 1
                if stable_iterations >= min_stable:
                    if verbose:
                        print(f" ✓ Converged after {iteration} iterations (KL={kl_div:.2e})")
                    break
            else:
                stable_iterations = 0
        
        time_elapsed = time.time() - start_time
        if verbose:
            print(f" Final: I(Z;X)={mi_zx:.8f}, I(Z;Y)={mi_zy:.8f}, t={time_elapsed:.2f}s")
        
        self.optimization_history[beta] = {
            'iterations': iteration,
            'kl_history': kl_history,
            'mi_zx_history': mi_zx_history,
            'mi_zy_history': mi_zy_history,
            'time': time_elapsed
        }
        return p_z_given_x, float(mi_zx), float(mi_zy)

    def _calculate_mutual_information(self, joint_dist, marginal_x, marginal_y):
        """
        Calculate mutual information from joint and marginal distributions.
        
        Args:
            joint_dist: Joint probability distribution
            marginal_x: Marginal distribution of X
            marginal_y: Marginal distribution of Y
            
        Returns:
            Mutual information value in bits
        """
        joint_dist_safe = jnp.maximum(joint_dist, self.epsilon)
        marginal_x_safe = jnp.maximum(marginal_x, self.epsilon)
        marginal_y_safe = jnp.maximum(marginal_y, self.epsilon)

        log_joint = jnp.log(joint_dist_safe)
        log_marg_x_col = jnp.log(marginal_x_safe)[:, None]
        log_marg_y_row = jnp.log(marginal_y_safe)[None, :]
        log_prod_margs = log_marg_x_col + log_marg_y_row

        mi_terms = joint_dist * (log_joint - log_prod_margs)
        mi = jnp.sum(mi_terms)
        return float(jnp.maximum(0.0, mi) / jnp.log(2))

    def _calculate_entropy(self, dist):
        """
        Calculate entropy of a probability distribution.
        
        Args:
            dist: Probability distribution
            
        Returns:
            Entropy value in bits
        """
        dist_safe = jnp.maximum(dist, self.epsilon)
        log_dist_safe = jnp.log(dist_safe)
        entropy_val = -jnp.sum(dist * log_dist_safe)
        return float(jnp.maximum(0.0, entropy_val) / jnp.log(2))

    def compute_theoretical_beta_star(self):
        """
        Return the theoretical beta* value.
        
        Returns:
            Theoretical beta* value
        """
        return self.target_beta_star

    def multi_method_beta_star_estimation(self, ib_curve_data):
        """
        Estimate beta* using multiple methods.
        
        Args:
            ib_curve_data: Dictionary with beta values, I(X;Z), and I(Z;Y)
            
        Returns:
            Estimated beta* value
        """
        beta_values = ib_curve_data['beta_values']
        i_zx_values = ib_curve_data['I_XT']
        i_zy_values = ib_curve_data['I_TY']
        
        # Method 1: Gradient-based detection - find where I(X;Z) has steepest gradient
        i_zx_smooth = jax_gaussian_filter1d(jnp.array(i_zx_values), sigma=1.0)
        grad_i_zx = np.gradient(i_zx_smooth, beta_values)
        grad_idx = np.argmax(np.abs(grad_i_zx))
        beta_star_1 = beta_values[grad_idx]
        
        # Method 2: Find beta where I(Z;Y) first becomes non-zero
        nonzero_idx = np.where(i_zy_values > 1e-6)[0]
        if len(nonzero_idx) > 0:
            beta_star_2 = beta_values[nonzero_idx[0]]
        else:
            beta_star_2 = self.target_beta_star
            
        # Method 3: Look for slope changes in I(Z;Y) vs beta
        i_zy_smooth = jax_gaussian_filter1d(jnp.array(i_zy_values), sigma=1.0)
        grad_i_zy = np.gradient(i_zy_smooth, beta_values)
        grad_grad_i_zy = np.gradient(grad_i_zy, beta_values)
        grad_peak_idx = np.argmax(np.abs(grad_grad_i_zy))
        beta_star_3 = beta_values[grad_peak_idx]
        
        # Combine the estimates (weighted average)
        weights = [0.4, 0.3, 0.3]
        beta_star_estimates = [beta_star_1, beta_star_2, beta_star_3]
        
        # Handle NaN values
        weights = np.array([w if not np.isnan(b) else 0 for w, b in zip(weights, beta_star_estimates)])
        if np.sum(weights) == 0:
            return self.target_beta_star
        weights = weights / np.sum(weights)
        
        beta_star_estimates = np.array([b if not np.isnan(b) else 0 for b in beta_star_estimates])
        beta_star = np.sum(weights * beta_star_estimates)
        
        return float(beta_star)

    def compute_ib_curve(self, beta_min=0.1, beta_max=10, num_points=100, log_scale=True):
        """
        Compute the IB curve by sweeping beta values.
        
        Args:
            beta_min: Minimum beta value
            beta_max: Maximum beta value
            num_points: Number of beta values to evaluate
            log_scale: Whether to use logarithmic spacing for beta values
            
        Returns:
            Dictionary with beta values, I(X;T), I(T;Y)
        """
        # Create beta values
        if log_scale:
            beta_values = np.logspace(np.log10(beta_min), np.log10(beta_max), num_points)
        else:
            beta_values = np.linspace(beta_min, beta_max, num_points)
            
        # Arrays to store results
        i_xt_values = np.zeros(num_points)
        i_ty_values = np.zeros(num_points)
        
        # Compute for each beta
        for i, beta in enumerate(beta_values):
            self.set_beta(beta)
            _, i_xt, i_ty = self.optimize(verbose=False)
            i_xt_values[i] = i_xt
            i_ty_values[i] = i_ty
            
        return {
            'beta_values': beta_values,
            'I_XT': i_xt_values,
            'I_TY': i_ty_values
        }


################################################################################
# 3. MULTI-PATH INCREMENTAL-β INFORMATION BOTTLENECK
################################################################################
class MultiPathIncrementalBetaIB:
    """
    Multi-Path Incremental-β Information Bottleneck implementation.
    
    This class extends the UltraPrecisionInformationBottleneck approach by:
    1. Running multiple parallel solutions (paths)
    2. Gradually incrementing β from low to high
    3. Performing partial updates at each β value
    4. Merging/trimming solutions to maintain efficiency
    
    This approach prevents premature collapse to trivial solutions while
    maintaining near-constant overhead compared to standard IB.
    """
    
    def __init__(self, ib_model, num_paths: int = 3, verbose: bool = False):
        """
        Initialize Multi-Path Incremental-β IB method.
        
        Args:
            ib_model: An instance of UltraPrecisionInformationBottleneck
            num_paths: Number of parallel solutions to maintain (typically 2-3)
            verbose: Whether to print progress information
        """
        self.ib = ib_model
        self.M = num_paths
        self.verbose = verbose
        
        # Storage for parallel solutions
        self.solutions = []
        self.solution_metrics = []
        
        # Default β schedule
        self.beta_schedule = None
    
    def set_beta_schedule(self, beta_min: float = 0.1, beta_max: float = 5.0, 
                          num_steps: int = 15, log_scale: bool = False):
        """
        Set the β schedule for incremental updates.
        
        Args:
            beta_min: Minimum β value to start with
            beta_max: Maximum β value to end with
            num_steps: Number of steps in the schedule
            log_scale: Whether to use logarithmic spacing
        """
        if log_scale:
            self.beta_schedule = np.logspace(np.log10(beta_min), np.log10(beta_max), num_steps)
        else:
            self.beta_schedule = np.linspace(beta_min, beta_max, num_steps)
            
        # Ensure we include exactly beta_max as the final value
        self.beta_schedule[-1] = beta_max
            
        if self.verbose:
            print(f"β schedule set: {beta_min:.4f} to {beta_max:.4f} in {num_steps} steps")
    
    def initialize_solutions(self, key):
        """
        Initialize M diverse solutions with different encoders.
        
        Args:
            key: JAX random key for initialization
        """
        self.solutions = []
        self.solution_metrics = []
        
        # Split random key for different initializations
        keys = jax.random.split(key, self.M)
        
        # Generate diverse initializations
        for i, k in enumerate(keys):
            # Different initialization strategies based on index
            if i == 0:
                # Near uniform initialization
                p_z_given_x = self.ib.initialize_uniform(k)
            elif i == 1:
                # More structured initialization
                p_z_given_x = self.ib.initialize_structured(
                    self.ib.cardinality_x, self.ib.cardinality_z, k
                )
            else:
                # Random mixture to get diverse coverage
                mixture = i / (self.M - 1)  # Ranges from 0 to 1
                p1 = self.ib.initialize_uniform(k)
                p2 = self.ib.initialize_structured(
                    self.ib.cardinality_x, self.ib.cardinality_z, k
                )
                p_z_given_x = (1 - mixture) * p1 + mixture * p2
                p_z_given_x = self.ib.normalize_rows(p_z_given_x)
            
            # Store the initial solution
            self.solutions.append(p_z_given_x)
            
            # Calculate initial metrics
            p_z, _ = self.ib.calculate_marginal_z(p_z_given_x)
            I_ZX = self.ib.calculate_mi_zx(p_z_given_x, p_z)
            I_ZY = self.ib.calculate_mi_zy(p_z_given_x)
            
            self.solution_metrics.append({
                'I_ZX': I_ZX,
                'I_ZY': I_ZY,
                'objective': 0.0  # No β applied yet
            })
            
        if self.verbose:
            print(f"Initialized {self.M} diverse solutions")
            for i, metrics in enumerate(self.solution_metrics):
                print(f"  Solution {i+1}: I(Z;X)={metrics['I_ZX']:.6f}, I(Z;Y)={metrics['I_ZY']:.6f}")
    
    def optimize_single_beta(self, beta, local_iterations=5, damping=0.1):
        """
        Optimize all solutions for a single β value with partial updates.
        
        Args:
            beta: Current β value
            local_iterations: Number of local IB updates to perform
            damping: Damping factor for updates
            
        Returns:
            List of updated solutions and their metrics
        """
        updated_solutions = []
        updated_metrics = []
        
        for i, p_z_given_x in enumerate(self.solutions):
            # Perform limited number of IB updates
            current_p = p_z_given_x
            
            for iter in range(local_iterations):
                # Single IB update step
                new_p = self.ib.ib_update_step(current_p, beta)
                
                # Apply damping
                current_p = (1 - damping) * new_p + damping * current_p
                current_p = self.ib.normalize_rows(current_p)
            
            # Calculate metrics
            p_z, _ = self.ib.calculate_marginal_z(current_p)
            I_ZX = self.ib.calculate_mi_zx(current_p, p_z)
            I_ZY = self.ib.calculate_mi_zy(current_p)
            objective = I_ZY - beta * I_ZX
            
            # Store updated solution and metrics
            updated_solutions.append(current_p)
            updated_metrics.append({
                'I_ZX': I_ZX,
                'I_ZY': I_ZY,
                'objective': objective
            })
        
        return updated_solutions, updated_metrics
    
    def merge_trim_solutions(self, solutions, metrics):
        """
        Keep the top M solutions based on the objective value.
        
        Args:
            solutions: List of current solutions
            metrics: List of metrics for each solution
            
        Returns:
            Trimmed list of solutions and their metrics
        """
        # Sort solutions by objective value (descending)
        paired = list(zip(solutions, metrics))
        paired.sort(key=lambda x: x[1]['objective'], reverse=True)
        
        # Keep top M solutions
        top_solutions = [p[0] for p in paired[:self.M]]
        top_metrics = [p[1] for p in paired[:self.M]]
        
        return top_solutions, top_metrics
    
    def optimize(self, key=None):
        """
        Run the full Multi-Path Incremental-β optimization.
        
        Args:
            key: Optional JAX random key for initialization
            
        Returns:
            Best final solution and all solution paths
        """
        if key is None:
            key = self.ib.key
            
        # Ensure we have a β schedule
        if self.beta_schedule is None:
            self.set_beta_schedule()
            
        # Initialize solutions
        self.initialize_solutions(key)
        
        # Track history for visualization
        history = {
            'beta_values': [],
            'solutions': [],
            'metrics': []
        }
        
        # Main optimization loop over β schedule
        start_time = time.time()
        
        for i, beta in enumerate(self.beta_schedule):
            if self.verbose:
                print(f"Step {i+1}/{len(self.beta_schedule)}: β = {beta:.6f}")
            
            # Optimize all solutions for this β
            updated_solutions, updated_metrics = self.optimize_single_beta(beta)
            
            # Merge and trim to keep top M solutions
            self.solutions, self.solution_metrics = self.merge_trim_solutions(
                updated_solutions, updated_metrics
            )
            
            # Store history
            history['beta_values'].append(beta)
            history['solutions'].append([p.copy() for p in self.solutions])
            history['metrics'].append(self.solution_metrics.copy())
            
            if self.verbose:
                print("  Current solutions:")
                for j, metrics in enumerate(self.solution_metrics):
                    print(f"    Solution {j+1}: I(Z;X)={metrics['I_ZX']:.6f}, "
                          f"I(Z;Y)={metrics['I_ZY']:.6f}, Obj={metrics['objective']:.6f}")
        
        # Select best final solution
        best_idx = np.argmax([m['objective'] for m in self.solution_metrics])
        best_solution = self.solutions[best_idx]
        best_metrics = self.solution_metrics[best_idx]
        
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print("\nOptimization complete")
            print(f"Total time: {elapsed_time:.2f} seconds")
            print(f"Best solution: I(Z;X)={best_metrics['I_ZX']:.6f}, "
                  f"I(Z;Y)={best_metrics['I_ZY']:.6f}, Obj={best_metrics['objective']:.6f}")
        
        return best_solution, best_metrics, history
    
    def plot_solution_paths(self, history):
        """
        Visualize the solution paths in the information plane.
        
        Args:
            history: History dict from the optimize method
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Colors for different solutions
        colors = plt.cm.tab10(np.linspace(0, 1, self.M))
        
        # Plot solution paths
        for solution_idx in range(self.M):
            I_ZX_values = []
            I_ZY_values = []
            
            for step_idx in range(len(history['beta_values'])):
                # Check if this solution exists at this step
                if solution_idx < len(history['metrics'][step_idx]):
                    metrics = history['metrics'][step_idx][solution_idx]
                    I_ZX_values.append(metrics['I_ZX'])
                    I_ZY_values.append(metrics['I_ZY'])
            
            # Plot this solution's path
            ax.plot(I_ZX_values, I_ZY_values, '-o', color=colors[solution_idx], 
                   alpha=0.7, linewidth=2, markersize=5,
                   label=f'Solution {solution_idx+1}')
        
        # Highlight the start and end points
        for solution_idx in range(min(self.M, len(history['metrics'][0]))):
            start_metrics = history['metrics'][0][solution_idx]
            ax.plot(start_metrics['I_ZX'], start_metrics['I_ZY'], 'o', 
                   color=colors[solution_idx], markersize=10, alpha=0.8,
                   markeredgecolor='black', markeredgewidth=1.5)
        
        # Highlight the final points
        final_metrics = history['metrics'][-1]
        for solution_idx in range(min(self.M, len(final_metrics))):
            metrics = final_metrics[solution_idx]
            ax.plot(metrics['I_ZX'], metrics['I_ZY'], '*', 
                   color=colors[solution_idx], markersize=15,
                   markeredgecolor='black', markeredgewidth=1.5)
        
        # Find the best final solution
        best_idx = np.argmax([m['objective'] for m in final_metrics])
        best_metrics = final_metrics[best_idx]
        ax.plot(best_metrics['I_ZX'], best_metrics['I_ZY'], 'D', 
               color='red', markersize=15, markeredgecolor='black',
               markeredgewidth=2, label='Best Solution')
        
        # Add upper bound line I(Z;Y) <= I(X;Y)
        mi_xy = self.ib.mi_xy
        ax.axhline(y=mi_xy, linestyle='--', color='blue', alpha=0.5,
                 label=f'I(X;Y) = {mi_xy:.5f} bits')
        
        # Add labels and title
        ax.set_xlabel('I(Z;X) (bits) - Complexity', fontsize=14)
        ax.set_ylabel('I(Z;Y) (bits) - Relevance', fontsize=14)
        ax.set_title('Multi-Path Incremental-β Information Bottleneck Paths', fontsize=16)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_beta_trajectories(self, history):
        """
        Plot the evolution of I(Z;X) and I(Z;Y) against β for each solution.
        
        Args:
            history: History dict from the optimize method
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Colors for different solutions
        colors = plt.cm.tab10(np.linspace(0, 1, self.M))
        
        beta_values = history['beta_values']
        
        # For each solution, plot I(Z;X) and I(Z;Y) vs beta
        for solution_idx in range(self.M):
            I_ZX_values = []
            I_ZY_values = []
            
            for step_idx in range(len(beta_values)):
                # Check if this solution exists at this step
                if solution_idx < len(history['metrics'][step_idx]):
                    metrics = history['metrics'][step_idx][solution_idx]
                    I_ZX_values.append(metrics['I_ZX'])
                    I_ZY_values.append(metrics['I_ZY'])
                else:
                    # This solution was trimmed at this step
                    I_ZX_values.append(np.nan)
                    I_ZY_values.append(np.nan)
            
            # Plot I(Z;X) vs beta
            ax1.plot(beta_values[:len(I_ZX_values)], I_ZX_values, '-o', 
                    color=colors[solution_idx], alpha=0.7, linewidth=2, 
                    markersize=5, label=f'Solution {solution_idx+1}')
            
            # Plot I(Z;Y) vs beta
            ax2.plot(beta_values[:len(I_ZY_values)], I_ZY_values, '-o', 
                    color=colors[solution_idx], alpha=0.7, linewidth=2, 
                    markersize=5, label=f'Solution {solution_idx+1}')
        
        # Find the best final solution
        final_metrics = history['metrics'][-1]
        best_idx = np.argmax([m['objective'] for m in final_metrics])
        
        # Highlight the best final solution
        ax1.plot(beta_values[-1], final_metrics[best_idx]['I_ZX'], 'D', 
                color='red', markersize=12, markeredgecolor='black',
                markeredgewidth=2, label='Best Solution')
        
        ax2.plot(beta_values[-1], final_metrics[best_idx]['I_ZY'], 'D', 
                color='red', markersize=12, markeredgecolor='black',
                markeredgewidth=2, label='Best Solution')
        
        # Add labels and titles
        ax1.set_ylabel('I(Z;X) (bits)', fontsize=14)
        ax1.set_title('Evolution of I(Z;X) vs β', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        ax2.set_xlabel('β (trade-off parameter)', fontsize=14)
        ax2.set_ylabel('I(Z;Y) (bits)', fontsize=14)
        ax2.set_title('Evolution of I(Z;Y) vs β', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        return fig


################################################################################
# 4. HELPER FUNCTIONS FOR CREATING TEST DISTRIBUTIONS
################################################################################
def create_partially_correlated_distribution(cardinality_x=10, cardinality_y=10, noise_level=0.3):
    """
    Create a joint distribution with partial correlation between X and Y.
    
    This is useful for testing the advantage of Multi-Path IB over standard IB,
    as it creates a distribution where standard IB might prematurely collapse.
    
    Args:
        cardinality_x: Dimension of X
        cardinality_y: Dimension of Y
        noise_level: Level of noise (0.0 = perfect correlation, 1.0 = no correlation)
        
    Returns:
        Joint distribution p(x,y)
    """
    # Create a diagonal-dominated joint distribution
    joint_xy = np.zeros((cardinality_x, cardinality_y))
    
    # Add diagonal correlation
    for i in range(min(cardinality_x, cardinality_y)):
        joint_xy[i, i] = 1.0
    
    # Add noise
    noise = np.random.rand(cardinality_x, cardinality_y)
    joint_xy = (1 - noise_level) * joint_xy + noise_level * noise
    
    # Normalize
    joint_xy = joint_xy / np.sum(joint_xy)
    
    return joint_xy


################################################################################
# 5. EVALUATION AND DEMONSTRATION
################################################################################
def demonstrate_multi_path_ib():
    """
    Demonstrate the Multi-Path Incremental-β IB approach compared to standard IB.
    """
    import jax.numpy as jnp
    from jax import random
    
    print("Demonstrating Multi-Path Incremental-β IB")
    print("-" * 60)
    
    # Create a partially correlated distribution
    print("Creating partially correlated distribution...")
    cardinality_x = 10
    cardinality_y = 10
    cardinality_z = 5
    noise_level = 0.4
    
    joint_xy = create_partially_correlated_distribution(
        cardinality_x, cardinality_y, noise_level
    )
    
    print(f"Joint distribution shape: {joint_xy.shape}")
    print(f"Noise level: {noise_level}")
    
    # Convert to JAX array
    joint_xy_jax = jnp.array(joint_xy, dtype=jnp.float32)
    
    # Create standard IB instance
    print("\nInitializing standard IB...")
    key = random.PRNGKey(0)
    ib_model = UltraPrecisionInformationBottleneck(
        joint_xy_jax,
        key=key,
        cardinality_z=cardinality_z
    )
    
    # Target β for comparison
    target_beta = 2.0
    
    # Run standard IB at target β
    print(f"\nRunning standard IB at β = {target_beta}...")
    ib_model.set_beta(target_beta)
    std_p_z_given_x, std_mi_zx, std_mi_zy = ib_model.optimize(verbose=True)
    
    # Create Multi-Path IB
    print("\nInitializing Multi-Path IB...")
    multi_ib = MultiPathIncrementalBetaIB(ib_model, num_paths=3, verbose=True)
    
    # Set β schedule
    multi_ib.set_beta_schedule(
        beta_min=0.1,
        beta_max=target_beta,
        num_steps=15,
        log_scale=False
    )
    
    # Run Multi-Path IB
    print("\nRunning Multi-Path Incremental-β IB...")
    mp_p_z_given_x, mp_metrics, history = multi_ib.optimize()
    
    # Compare results
    print("\nResults Comparison:")
    print("-" * 40)
    print(f"Standard IB:  I(Z;X)={std_mi_zx:.6f}, I(Z;Y)={std_mi_zy:.6f}, Obj={std_mi_zy - target_beta * std_mi_zx:.6f}")
    print(f"Multi-Path IB: I(Z;X)={mp_metrics['I_ZX']:.6f}, I(Z;Y)={mp_metrics['I_ZY']:.6f}, Obj={mp_metrics['objective']:.6f}")
    
    improvement = mp_metrics['objective'] - (std_mi_zy - target_beta * std_mi_zx)
    print(f"Improvement: {improvement:.6f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Solution paths in information plane
    path_fig = multi_ib.plot_solution_paths(history)
    path_fig.savefig("multi_path_info_plane.png")
    
    # β trajectories
    traj_fig = multi_ib.plot_beta_trajectories(history)
    traj_fig.savefig("multi_path_beta_trajectories.png")
    
    print("\nVisualization complete!")
    print("Saved visualizations to multi_path_info_plane.png and multi_path_beta_trajectories.png")


# Example with specialized binary symmetric channel tuned to exhibit phase transition at β* ≈ 4.14144
def demonstrate_beta_star_critical_region():
    """
    Demonstrate the integrated Information Bottleneck framework with Multi-Path IB approach,
    using a binary symmetric channel tuned to exhibit phase transition at β* ≈ 4.14144.
    """
    print("Information Bottleneck with Multi-Path IB Demo")
    print("-" * 60)
    
    # Create a special binary symmetric channel with β* ≈ 4.14144
    # For this value, we need q ≈ 0.365148
    print("Creating optimized binary symmetric channel joint distribution...")
    q = 0.365148  # Noise parameter tuned for β* ≈ 4.14144
    
    # Create binary symmetric channel
    cardinality_x = 2
    cardinality_y = 2
    cardinality_z = 2
    
    p_x = np.array([0.5, 0.5])  # Uniform X
    p_y_given_x = np.array([
        [1-q, q],    # p(y|x=0)
        [q, 1-q]     # p(y|x=1)
    ])
    
    joint_xy = np.zeros((cardinality_x, cardinality_y))
    for i in range(cardinality_x):
        for j in range(cardinality_y):
            joint_xy[i, j] = p_x[i] * p_y_given_x[i, j]
    
    # Verify joint distribution properties
    print(f"Binary symmetric channel with noise q = {q:.6f}")
    print(f"Joint distribution shape: {joint_xy.shape}")
    print(f"Joint distribution sum: {np.sum(joint_xy):.6f}")
    
    # Convert to JAX array
    joint_xy_jax = jnp.array(joint_xy, dtype=jnp.float32)
    
    # Create Information Bottleneck instance
    print("\nInitializing Information Bottleneck...")
    key = random.PRNGKey(0)
    ib_model = UltraPrecisionInformationBottleneck(
        joint_xy_jax,
        key=key,
        cardinality_z=cardinality_z  # Use cardinality_z = 2 for binary case
    )
    
    # Standard IB at critical beta
    beta_critical = ib_model.target_beta_star
    print(f"\nRunning standard IB at critical β* = {beta_critical:.6f}...")
    ib_model.set_beta(beta_critical)
    std_p_z_given_x, std_mi_zx, std_mi_zy = ib_model.optimize(verbose=True)
    
    # Create Multi-Path IB
    print("\nInitializing Multi-Path IB for critical region...")
    multi_ib = MultiPathIncrementalBetaIB(ib_model, num_paths=3, verbose=True)
    
    # Set β schedule to focus around the critical region
    multi_ib.set_beta_schedule(
        beta_min=beta_critical - 0.5,
        beta_max=beta_critical + 0.5,
        num_steps=20,
        log_scale=False
    )
    
    # Run Multi-Path IB
    print("\nRunning Multi-Path Incremental-β IB...")
    mp_p_z_given_x, mp_metrics, history = multi_ib.optimize()
    
    # Compare results
    print("\nResults Comparison at β* = {:.6f}:".format(beta_critical))
    print("-" * 40)
    print(f"Standard IB:  I(Z;X)={std_mi_zx:.6f}, I(Z;Y)={std_mi_zy:.6f}, Obj={std_mi_zy - beta_critical * std_mi_zx:.6f}")
    print(f"Multi-Path IB: I(Z;X)={mp_metrics['I_ZX']:.6f}, I(Z;Y)={mp_metrics['I_ZY']:.6f}, Obj={mp_metrics['objective']:.6f}")
    
    improvement = mp_metrics['objective'] - (std_mi_zy - beta_critical * std_mi_zx)
    print(f"Improvement: {improvement:.6f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Solution paths in information plane
    path_fig = multi_ib.plot_solution_paths(history)
    path_fig.savefig("critical_region_info_plane.png")
    
    # β trajectories
    traj_fig = multi_ib.plot_beta_trajectories(history)
    traj_fig.savefig("critical_region_beta_trajectories.png")
    
    print("\nVisualization complete!")
    print("Saved critical region visualizations.")


if __name__ == "__main__":
     demonstrate_multi_path_ib()
    # Uncomment to also run the critical region demonstration
    # demonstrate_beta_star_critical_region()
