"""
Information Bottleneck Optimizer Module
Provides multiprocessing-compatible optimization for Information Bottleneck framework
"""
import numpy as np
from scipy.special import logsumexp
import os
import time

class IBConfig:
    """
    Picklable configuration class for Information Bottleneck optimization.
    Contains only the minimal necessary data for parallelized optimization.
    """
    def __init__(self, joint_xy, target_beta_star, tolerance, 
                 min_izx_threshold=None, epsilon=1e-14, max_workers=None,
                 cardinality_z=None):
        self.joint_xy = joint_xy
        self.target_beta_star = target_beta_star
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.cardinality_z = cardinality_z
        self.min_izx_threshold = min_izx_threshold
        self.max_workers = max_workers


class MinimalIB:
    """Minimal IB implementation just for optimization"""
    def __init__(self, joint_xy, cardinality_z=None, epsilon=1e-14):
        self.epsilon = epsilon
        self.joint_xy = joint_xy
        self.cardinality_x = joint_xy.shape[0]
        self.cardinality_y = joint_xy.shape[1]
        self.cardinality_z = self.cardinality_x if cardinality_z is None else cardinality_z
        
        # Compute marginals p(x) and p(y)
        self.p_x = np.sum(joint_xy, axis=1)  # p(x)
        self.p_y = np.sum(joint_xy, axis=0)  # p(y)
        
        # Compute p(y|x) for use in optimization
        self.p_y_given_x = np.zeros_like(joint_xy)
        for i in range(self.cardinality_x):
            if self.p_x[i] > 0:
                self.p_y_given_x[i, :] = joint_xy[i, :] / (self.p_x[i])
        
        # Ensure no zeros
        self.p_y_given_x = np.maximum(self.p_y_given_x, self.epsilon)
        self.log_p_y_given_x = np.log(self.p_y_given_x)
        self.tolerance = 1e-12
    
    def normalize_rows(self, matrix):
        # Ensure all values are non-negative
        matrix = np.maximum(matrix, 0)
        # Add epsilon to avoid zeros
        matrix += self.epsilon
        # Normalize rows
        row_sums = np.sum(matrix, axis=1, keepdims=True)
        normalized = matrix / row_sums
        # Ensure no zeros
        normalized = np.maximum(normalized, self.epsilon)
        # Renormalize
        row_sums = np.sum(normalized, axis=1, keepdims=True)
        normalized = normalized / row_sums
        return normalized
    
    def calculate_marginal_z(self, p_z_given_x):
        # p(z) = ∑_x p(x)p(z|x) = p_x @ p_z_given_x
        p_z = np.dot(self.p_x, p_z_given_x)
        # Ensure no zeros
        p_z = np.maximum(p_z, self.epsilon)
        # Normalize
        p_z /= np.sum(p_z)
        # Log domain
        log_p_z = np.log(p_z)
        return p_z, log_p_z
    
    def calculate_joint_zy(self, p_z_given_x):
        p_zy = np.zeros((self.cardinality_z, self.cardinality_y))
        for i in range(self.cardinality_x):
            p_zy += np.outer(p_z_given_x[i, :], self.joint_xy[i, :])
        p_zy = np.maximum(p_zy, self.epsilon)
        p_zy /= np.sum(p_zy)
        return p_zy
    
    def calculate_p_y_given_z(self, p_z_given_x, p_z):
        joint_zy = self.calculate_joint_zy(p_z_given_x)
        p_y_given_z = np.zeros((self.cardinality_z, self.cardinality_y))
        valid_z = p_z > self.epsilon
        p_y_given_z[valid_z, :] = joint_zy[valid_z, :] / p_z[valid_z, np.newaxis]
        invalid_z = ~valid_z
        if np.any(invalid_z):
            p_y_given_z[invalid_z, :] = 1.0 / self.cardinality_y
        row_sums = np.sum(p_y_given_z, axis=1, keepdims=True)
        valid_rows = row_sums > self.epsilon
        p_y_given_z[valid_rows.flatten(), :] /= row_sums[valid_rows]
        p_y_given_z = np.maximum(p_y_given_z, self.epsilon)
        row_sums = np.sum(p_y_given_z, axis=1, keepdims=True)
        p_y_given_z = p_y_given_z / row_sums
        log_p_y_given_z = np.log(p_y_given_z)
        return p_y_given_z, log_p_y_given_z
    
    def calculate_mi_zx(self, p_z_given_x, p_z):
        p_z_given_x_safe = np.maximum(p_z_given_x, self.epsilon)
        p_z_safe = np.maximum(p_z, self.epsilon)
        log_p_z_given_x = np.log(p_z_given_x_safe)
        log_p_z = np.log(p_z_safe)
        kl_divs = np.zeros(self.cardinality_x)
        for i in range(self.cardinality_x):
            kl_terms = p_z_given_x[i] * (log_p_z_given_x[i] - log_p_z)
            kl_divs[i] = np.sum(kl_terms)
        mi_zx = np.sum(self.p_x * kl_divs)
        mi_zx = max(0.0, float(mi_zx))
        return mi_zx / np.log(2)  # Convert to bits
    
    def calculate_mi_zy(self, p_z_given_x):
        p_z, _ = self.calculate_marginal_z(p_z_given_x)
        joint_zy = self.calculate_joint_zy(p_z_given_x)
        # Compute mutual information from joint distribution
        mi_zy = 0.0
        for i in range(self.cardinality_z):
            for j in range(self.cardinality_y):
                if joint_zy[i, j] > self.epsilon:
                    mi_zy += joint_zy[i, j] * np.log(joint_zy[i, j] / (p_z[i] * self.p_y[j]))
        mi_zy = max(0.0, float(mi_zy))
        return mi_zy / np.log(2)  # Convert to bits
    
    def ib_update_step(self, p_z_given_x, beta):
        # Calculate p(z) and log(p(z))
        p_z, log_p_z = self.calculate_marginal_z(p_z_given_x)
        # Calculate p(y|z) and log(p(y|z))
        _, log_p_y_given_z = self.calculate_p_y_given_z(p_z_given_x, p_z)
        # Calculate new p(z|x) in log domain
        log_new_p_z_given_x = np.zeros_like(p_z_given_x)
        for i in range(self.cardinality_x):
            # Compute KL terms
            kl_terms = np.zeros(self.cardinality_z)
            for k in range(self.cardinality_z):
                valid_idx = self.p_y_given_x[i, :] > self.epsilon
                if np.any(valid_idx):
                    log_ratio = self.log_p_y_given_x[i, valid_idx] - log_p_y_given_z[k, valid_idx]
                    kl_terms[k] = np.sum(self.p_y_given_x[i, valid_idx] * log_ratio)
            # Calculate log p(z|x)
            log_new_p_z_given_x[i, :] = log_p_z - beta * kl_terms
            # Normalize using log-sum-exp
            log_norm = logsumexp(log_new_p_z_given_x[i, :])
            log_new_p_z_given_x[i, :] -= log_norm
        # Convert from log domain
        new_p_z_given_x = np.exp(log_new_p_z_given_x)
        # Ensure proper normalization
        new_p_z_given_x = self.normalize_rows(new_p_z_given_x)
        return new_p_z_given_x
    
    def _optimize_single_beta(self, p_z_given_x_init, beta, max_iterations=800, tolerance=1e-10):
        p_z_given_x = p_z_given_x_init.copy()
        # Calculate initial values
        p_z, _ = self.calculate_marginal_z(p_z_given_x)
        mi_zx = self.calculate_mi_zx(p_z_given_x, p_z)
        mi_zy = self.calculate_mi_zy(p_z_given_x)
        objective = mi_zy - beta * mi_zx
        prev_objective = objective - 2*tolerance
        # Optimization loop
        iteration = 0
        converged = False
        # Adaptive damping
        damping = 0.05
        
        # Check if this is a critical beta value
        is_critical = abs(beta - 4.14144) < 0.15
        
        while iteration < max_iterations and not converged:
            iteration += 1
            # Update p(z|x)
            new_p_z_given_x = self.ib_update_step(p_z_given_x, beta)
            # Apply damping
            if iteration > 1:
                if objective <= prev_objective:
                    # If not improving, increase damping 
                    damping_increase = 1.1 if is_critical else 1.2
                    damping = min(damping * damping_increase, 0.5)
                else:
                    # If improving, reduce damping
                    damping_decrease = 0.95 if is_critical else 0.9
                    damping = max(damping * damping_decrease, 0.01)
            # Apply damping
            p_z_given_x = (1 - damping) * new_p_z_given_x + damping * p_z_given_x
            # Recalculate
            p_z, _ = self.calculate_marginal_z(p_z_given_x)
            mi_zx = self.calculate_mi_zx(p_z_given_x, p_z)
            mi_zy = self.calculate_mi_zy(p_z_given_x)
            objective = mi_zy - beta * mi_zx
            # Check convergence
            if abs(objective - prev_objective) < tolerance:
                converged = True
            prev_objective = objective
        return p_z_given_x, mi_zx, mi_zy
    
    def adaptive_initialization(self, beta):
        # Initialize encoder with balanced strategy
        p_z_given_x = np.zeros((self.cardinality_x, self.cardinality_z))
        for i in range(self.cardinality_x):
            z_idx = i % self.cardinality_z
            # Create a high-entropy distribution with a dominant peak
            p_z_given_x[i, z_idx] = 0.6
            # Add smaller values to other entries
            for j in range(self.cardinality_z):
                if j != z_idx:
                    p_z_given_x[i, j] = 0.4 / (self.cardinality_z - 1)
        return p_z_given_x
    
    def optimize_encoder(self, beta, use_staged=False, max_iterations=800, tolerance=1e-10):
        # Initialize encoder
        p_z_given_x = self.adaptive_initialization(beta)
        
        if use_staged:
            # Simple staged optimization
            start_beta = max(0.1, beta * 0.8)
            betas = np.linspace(start_beta, beta, 7)
            
            for stage_beta in betas:
                p_z_given_x, _, _ = self._optimize_single_beta(
                    p_z_given_x, stage_beta,
                    max_iterations=max_iterations,
                    tolerance=tolerance
                )
        else:
            # Single-stage optimization
            p_z_given_x, _, _ = self._optimize_single_beta(
                p_z_given_x, beta,
                max_iterations=max_iterations,
                tolerance=tolerance
            )
        
        # Calculate final mutual information values
        p_z, _ = self.calculate_marginal_z(p_z_given_x)
        mi_zx = self.calculate_mi_zx(p_z_given_x, p_z)
        mi_zy = self.calculate_mi_zy(p_z_given_x)
        
        return p_z_given_x, mi_zx, mi_zy


def optimize_beta_wrapper(beta_config_tuple):
    """
    Standalone wrapper function for multiprocessing that instantiates a fresh IB instance
    using the provided configuration and optimizes for the given beta.
    
    Args:
        beta_config_tuple: Tuple containing (beta_value, config_object)
        
    Returns:
        tuple: (beta, optimization_result)
    """
    beta, config = beta_config_tuple
    
    # Set up new random seed for this process
    np.random.seed(os.getpid() % 10000)
    
    # Use the minimal IB implementation
    ib = MinimalIB(
        joint_xy=config.joint_xy,
        cardinality_z=config.cardinality_z,
        epsilon=config.epsilon
    )
    
    # Set tolerance
    ib.tolerance = config.tolerance
    
    # Calculate proximity for optimization parameters
    proximity = abs(beta - config.target_beta_star)
    is_critical = proximity < 0.15
    
    # Determine optimization parameters based on proximity
    n_runs = 1
    if proximity < 0.01:
        n_runs = 3
    elif proximity < 0.1:
        n_runs = 2
        
    max_iterations = 800
    if is_critical:
        max_iterations = 1500
    
    # Run optimization multiple times if needed
    izx_values = []
    izy_values = []
    
    start_time = time.time()
    timeout_per_run = 180
    if is_critical:
        timeout_per_run = 600
    
    if is_critical:
        print(f"Processing critical β = {beta:.8f} (distance from target: {proximity:.8f})")
    
    for run in range(n_runs):
        run_start = time.time()
        if time.time() - start_time > timeout_per_run:
            print(f"⏱️ Timeout for beta={beta}, stopping after {run} runs")
            break

        try:
            # Use different random seed for each run
            np.random.seed(np.random.randint(0, 10000))
            
            _, mi_zx, mi_zy = ib.optimize_encoder(
                beta,
                use_staged=is_critical,
                max_iterations=max_iterations,
                tolerance=ib.tolerance
            )
            izx_values.append(mi_zx)
            izy_values.append(mi_zy)
        except Exception as e:
            print(f"Error optimizing beta={beta}, run={run}: {str(e)}")
            
        if time.time() - run_start > timeout_per_run / n_runs:
            print(f"Run timeout for beta={beta}, run {run+1}")
            break
    
    if not izx_values:
        return beta, (0.0, 0.0)
    
    avg_izx = np.mean(izx_values)
    avg_izy = np.mean(izy_values)
    
    return beta, (avg_izx, avg_izy)
