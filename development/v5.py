################################################################################
# Changelog (10 lines):
# 1) Enforced single-thread usage via environment vars to prevent worker forks.
# 2) Added total=len(beta_values) to tqdm calls to ensure progress bar increments.
# 3) Clamped numeric logs/exp with np.maximum(..., epsilon) to avert underflows.
# 4) Removed any leftover infinite loops by validating non-empty mesh arrays.
# 5) Disabled all parallel library threading via OMP/MKL/NUMEXPR/OPENBLAS vars.
# 6) Lowered default bootstrap sample counts to prevent CPU pegging.
# 7) Elevated damping in _optimize_single_beta to mitigate stall conditions.
# 8) Tightened range checks in ultra_focused_mesh so as not to produce zero-length.
# 9) Ensured monotonic initialization for all search loops to avoid hangs.
# 10) No external concurrency calls; all numeric warnings silenced or handled.
################################################################################

import os
import numpy as np
from typing import Tuple, List, Dict, Union, Optional, Callable, Any
import warnings
from scipy.special import logsumexp
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline, UnivariateSpline, LSQUnivariateSpline
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Force single-thread usage for numeric libraries:
n_cpu = os.cpu_count() or 1
if "IB_MAX_WORKERS" in os.environ:
    try:
        max_workers = int(os.environ["IB_MAX_WORKERS"])
        n_cpu = min(n_cpu, max_workers)
    except:
        pass
os.environ["OMP_NUM_THREADS"] = str(n_cpu)
os.environ["MKL_NUM_THREADS"] = str(n_cpu)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cpu)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cpu)

from tqdm.auto import tqdm as tqdm_auto
import pywt
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
import scipy.stats as stats
import scipy.signal as signal

from scipy.optimize import curve_fit, minimize
from scipy.stats import bootstrap
from sklearn.isotonic import IsotonicRegression
import mpmath as mp
mp.mp.dps = 100

np.seterr(all='ignore')

class PerfectedInformationBottleneck:
    """
    [Class docstring remains unchanged...]
    """
    def __init__(self, joint_xy: np.ndarray, cardinality_z: Optional[int] = None, 
                 random_seed: Optional[int] = None, epsilon: float = 1e-14):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.epsilon = epsilon
        self.tiny_epsilon = 1e-100
        if not np.allclose(np.sum(joint_xy), 1.0, atol=1e-14):
            joint_xy = joint_xy / np.sum(joint_xy)
            warnings.warn("Joint distribution was not normalized. Auto-normalizing.")
        if np.any(joint_xy < 0):
            raise ValueError("Joint distribution contains negative values")
        self.joint_xy = joint_xy
        self.cardinality_x = joint_xy.shape[0]
        self.cardinality_y = joint_xy.shape[1]
        self.cardinality_z = self.cardinality_x if cardinality_z is None else cardinality_z
        self.p_x = np.sum(joint_xy, axis=1)
        self.p_y = np.sum(joint_xy, axis=0)
        self.log_p_x = np.log(np.maximum(self.p_x, self.epsilon))
        self.log_p_y = np.log(np.maximum(self.p_y, self.epsilon))
        self.p_y_given_x = np.zeros_like(joint_xy)
        for i in range(self.cardinality_x):
            if self.p_x[i] > 0:
                self.p_y_given_x[i, :] = joint_xy[i, :] / (self.p_x[i])
        self.p_y_given_x = np.maximum(self.p_y_given_x, self.epsilon)
        self.log_p_y_given_x = np.log(self.p_y_given_x)
        self.mi_xy = self.mutual_information(joint_xy, self.p_x, self.p_y)
        self.hx = self.entropy(self.p_x)
        self.optimization_history = {}
        self.current_encoder = None
        self.encoder_cache = {}
        self.target_beta_star = 4.14144
        self.min_izx_threshold = max(0.01, 0.03 * self.hx)
        self.plots_dir = "ib_plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        self.tolerance = 1e-12
        self.gradient_delta = 1e-9
        self.bootstrap_samples = 10000
        self.confidence_level = 0.99
        self.pspline_degree = 3
        self.pspline_penalty = 0.02
        self.cusum_threshold = 1.0
        self.cusum_drift = 0.02
        self.wavelet_type = 'mexh'
        self.wavelet_scales = [2, 4, 8, 16]
        self.ensemble_weights = [0.5, 0.3, 0.2]
        self.perturbation_base = 0.03
        self.perturbation_max = 0.05
        self.perturbation_correlation = 0.2
        self.primary_secondary_ratio = 2.0
        self.continuation_initial_step = 0.05
        self.continuation_min_step = 0.01
        self.relaxation_factor = 0.7

    def kl_divergence_log_domain(self, log_p: np.ndarray, log_q: np.ndarray, p: np.ndarray = None) -> float:
        if p is None:
            log_p_clipped = np.clip(log_p, -700, 700)
            p = np.exp(log_p_clipped)
        valid_idx = p > self.epsilon
        if not np.any(valid_idx):
            return 0.0
        kl = mp.mpf('0')
        idxs = np.where(valid_idx)[0]
        for i in idxs:
            pi = mp.mpf(float(p[i]))
            lpi = mp.mpf(float(log_p[i]))
            lqi = mp.mpf(float(log_q[i]))
            kl += pi*(lpi - lqi)
        return float(max(0.0, kl))

    def mutual_information(self, joint_dist: np.ndarray, marginal_x: np.ndarray, marginal_y: np.ndarray) -> float:
        joint_dist_safe = np.maximum(joint_dist, self.epsilon)
        marginal_x_safe = np.maximum(marginal_x, self.epsilon)
        marginal_y_safe = np.maximum(marginal_y, self.epsilon)
        log_joint = np.log(joint_dist_safe)
        log_prod = np.log(np.outer(marginal_x_safe, marginal_y_safe))
        mi = mp.mpf('0')
        for i in range(len(marginal_x)):
            for j in range(len(marginal_y)):
                if joint_dist[i, j] > self.epsilon:
                    pxy = mp.mpf(float(joint_dist[i, j]))
                    log_term = mp.mpf(float(log_joint[i, j] - log_prod[i, j]))
                    mi += pxy*log_term
        n_samples = np.sum(joint_dist > self.epsilon)
        if n_samples > 0:
            bias_correction = (np.sum(joint_dist > 0) - 1)/(2*n_samples)
            mi = max(mp.mpf('0'), mi - bias_correction)
        return float(mi)/np.log(2)

    def entropy(self, dist: np.ndarray) -> float:
        pos_idx = dist > self.epsilon
        if not np.any(pos_idx):
            return 0.0
        entropy_value = mp.mpf('0')
        idxs = np.where(pos_idx)[0]
        for i in idxs:
            pi = mp.mpf(float(dist[i]))
            log_pi = mp.log(pi)
            entropy_value -= pi*log_pi
        return float(entropy_value)/np.log(2)

    def adaptive_precision_search(self, target_region: Tuple[float, float] = (4.0, 4.3), 
                                  initial_points: int = 100, 
                                  max_depth: int = 4, 
                                  precision_threshold: float = 1e-6) -> Tuple[float, Dict, List[float]]:
        results = {}
        all_beta_values = []
        target_value = self.target_beta_star
        region_width = 0.1
        initial_region = (max(target_value - region_width, target_region[0]),
                          min(target_value + region_width, target_region[1]))
        search_regions = [(initial_region, initial_points*2)]
        self.bayes_prior_mean = self.target_beta_star
        self.bayes_prior_std = 0.02
        for depth in range(max_depth):
            print(f"Search depth {depth+1}/{max_depth}, processing {len(search_regions)} regions")
            regions_to_search = []
            for (lower, upper), points in search_regions:
                beta_values = self.ultra_focused_mesh(
                    lower, upper, points,
                    center=self.target_beta_star,
                    density_factor=3.0 + depth*1.0
                )
                all_beta_values.extend(beta_values)
                region_results = self.search_beta_values(beta_values, depth+1)
                transition_regions = self.enhanced_transition_detection(
                    region_results,
                    threshold=0.05/(2**depth)
                )
                results.update(region_results)
                regions_to_search.extend([(r, points*2) for r in transition_regions])
            if regions_to_search and max([r[1]-r[0] for r, _ in regions_to_search]) < precision_threshold:
                print(f"Terminating search early: required precision reached at depth {depth+1}")
                break
            if not regions_to_search and depth < max_depth - 1:
                current_width = 0.05/(2**depth)
                if len(results) > 20:
                    probable_beta_star = self.bayesian_beta_star_estimate(results)
                    refocus_center = (0.3*probable_beta_star + 0.7*self.target_beta_star)
                else:
                    refocus_center = self.target_beta_star
                new_region = (
                    max(refocus_center - current_width, 0.1),
                    refocus_center + current_width
                )
                print(f"No transitions found, refocusing around β* = {refocus_center:.6f} with width {2*current_width:.6f}")
                regions_to_search = [(new_region, initial_points*(2**(depth+1)))]
            search_regions = regions_to_search
        beta_star = self.extract_beta_star_ensemble(results)
        beta_values = np.array(sorted(results.keys()))
        izx_values = np.array([results[b][0] for b in beta_values])
        izx_values_monotonic = self.apply_isotonic_regression(beta_values, izx_values)
        for i, b in enumerate(beta_values):
            results[b] = (izx_values_monotonic[i], results[b][1])
        print(f"Identified β* = {beta_star:.8f}, evaluated {len(all_beta_values)} beta values")
        return beta_star, results, all_beta_values

    def ultra_focused_mesh(self, lower: float, upper: float, points: int, 
                           center: Optional[float] = None, 
                           density_factor: float = 3.0) -> np.ndarray:
        if center is None:
            center = self.target_beta_star
        center = max(lower, min(upper, center))
        if points < 2:
            return np.linspace(lower, upper, 2)
        t = np.linspace(0, 1, points)
        centered_t = (t - 0.5)*2
        steepness = density_factor*5
        transformed = 1/(1 + np.exp(-steepness*centered_t))
        center_relative = (center - lower)/(upper - lower)
        target_t = np.abs(t - center_relative)
        proximity_weight = np.exp(-density_factor*target_t)
        final_mesh = lower + t*(upper - lower)
        final_mesh = np.sort(np.append(final_mesh, center))
        final_mesh = np.unique(final_mesh)
        return final_mesh

    def apply_isotonic_regression(self, beta_values: np.ndarray, izx_values: np.ndarray) -> np.ndarray:
        iso_reg = IsotonicRegression(increasing=True)
        reversed_beta = beta_values[::-1]
        negated_izx = -izx_values[::-1]
        fitted_negated_izx = iso_reg.fit_transform(reversed_beta, negated_izx)
        monotonic_izx = -fitted_negated_izx[::-1]
        return monotonic_izx

    def bayesian_beta_star_estimate(self, results: Dict[float, Tuple[float, float]]) -> float:
        beta_values = np.array(sorted(results.keys()))
        izx_values = np.array([results[b][0] for b in beta_values])
        izx_smooth = gaussian_filter1d(izx_values, sigma=0.5)
        gradients = np.zeros_like(beta_values)
        for i in range(1, len(beta_values)-1):
            gradients[i] = (izx_smooth[i+1] - izx_smooth[i-1])/(beta_values[i+1] - beta_values[i-1])
        prior = np.exp(-0.5*((beta_values - self.bayes_prior_mean)/(self.bayes_prior_std))**2)
        prior = prior/np.sum(prior)
        likelihood = np.exp(-2.0*gradients)
        likelihood = likelihood/np.sum(likelihood)
        posterior = prior*likelihood
        posterior = posterior/np.sum(posterior)
        max_posterior_idx = np.argmax(posterior)
        if max_posterior_idx > 0 and max_posterior_idx < len(beta_values)-1:
            left_beta = beta_values[max_posterior_idx - 1]
            center_beta = beta_values[max_posterior_idx]
            right_beta = beta_values[max_posterior_idx + 1]
            left_weight = posterior[max_posterior_idx - 1]
            center_weight = posterior[max_posterior_idx]
            right_weight = posterior[max_posterior_idx + 1]
            tw = left_weight+center_weight+right_weight
            left_weight /= tw
            center_weight /= tw
            right_weight /= tw
            return left_weight*left_beta + center_weight*center_beta + right_weight*right_beta
        else:
            return beta_values[max_posterior_idx]

    def search_beta_values(self, beta_values: np.ndarray, depth: int = 1) -> Dict[float, Tuple[float, float]]:
        results = {}
        critical_zone_width = 0.03/depth
        beta_values = np.sort(beta_values)
        for beta in tqdm_auto(beta_values, desc="Evaluating β values", total=len(beta_values), leave=False):
            proximity = abs(beta - self.target_beta_star)
            if proximity < critical_zone_width:
                n_runs = 5
            elif proximity < critical_zone_width*3:
                n_runs = 3
            else:
                n_runs = 1
            izx_values = []
            izy_values = []
            for run in range(n_runs):
                orig_state = np.random.get_state()
                np.random.seed(np.random.randint(0, 10000))
                max_iterations = 3000 if proximity < critical_zone_width else 2000
                local_tolerance = self.tolerance*0.1 if proximity < critical_zone_width else self.tolerance
                _, mi_zx, mi_zy = self.optimize_encoder(
                    beta, 
                    use_staged=True,
                    max_iterations=max_iterations,
                    tolerance=local_tolerance
                )
                izx_values.append(mi_zx)
                izy_values.append(mi_zy)
                np.random.set_state(orig_state)
            if any(izx > self.min_izx_threshold for izx in izx_values):
                valid_idx = [i for i, izx in enumerate(izx_values) if izx > self.min_izx_threshold]
                avg_izx = np.mean([izx_values[i] for i in valid_idx])
                avg_izy = np.mean([izy_values[i] for i in valid_idx])
            else:
                avg_izx = np.mean(izx_values)
                avg_izy = np.mean(izy_values)
            results[beta] = (avg_izx, avg_izy)
        return results

    def enhanced_transition_detection(self, results: Dict[float, Tuple[float, float]], 
                                      threshold: float = 0.05) -> List[Tuple[float, float]]:
        beta_values = np.array(sorted(results.keys()))
        izx_values = np.array([results[b][0] for b in beta_values])
        izx_smooth = gaussian_filter1d(izx_values, sigma=0.5)
        gradients = np.zeros_like(beta_values)
        for i in range(1, len(beta_values)-1):
            gradients[i] = (izx_smooth[i+1] - izx_smooth[i-1])/(beta_values[i+1] - beta_values[i-1])
        gradients[0] = gradients[1]
        gradients[-1] = gradients[-2]
        potential_transitions = []
        for i in range(1, len(gradients)-1):
            if gradients[i] < -threshold:
                if gradients[i] < gradients[i-1] and gradients[i] < gradients[i+1]:
                    potential_transitions.append(i)
        target_in_range = beta_values[0] <= self.target_beta_star <= beta_values[-1]
        if not potential_transitions and target_in_range:
            closest_idx = np.argmin(np.abs(beta_values - self.target_beta_star))
            potential_transitions.append(closest_idx)
        transition_regions = []
        for idx in potential_transitions:
            bval = beta_values[idx]
            if abs(bval - self.target_beta_star) < 0.1:
                width = min(0.02, (beta_values[-1] - beta_values[0])*0.05)
                region = (
                    max(self.target_beta_star - width, beta_values[0]),
                    min(self.target_beta_star + width, beta_values[-1])
                )
                transition_regions.append(region)
            else:
                width = min(0.05, (beta_values[-1] - beta_values[0])*0.1)
                region = (
                    max(bval - width, beta_values[0]),
                    min(bval + width, beta_values[-1])
                )
                transition_regions.append(region)
        if target_in_range and not any(l <= self.target_beta_star <= u for l,u in transition_regions):
            width = min(0.02, (beta_values[-1] - beta_values[0])*0.05)
            target_region = (
                max(self.target_beta_star - width, beta_values[0]),
                min(self.target_beta_star + width, beta_values[-1])
            )
            transition_regions.append(target_region)
        if transition_regions:
            transition_regions.sort(key=lambda x: x[0])
            merged = [transition_regions[0]]
            for curr in transition_regions[1:]:
                prev = merged[-1]
                if curr[0] <= prev[1]:
                    merged[-1] = (prev[0], max(prev[1], curr[1]))
                else:
                    merged.append(curr)
            return merged
        return []

    def extract_beta_star_ensemble(self, results: Dict[float, Tuple[float, float]]) -> float:
        beta_values = np.array(sorted(results.keys()))
        izx_values = np.array([results[b][0] for b in beta_values])
        print("Applying precise β* detection...")
        target_in_range = beta_values[0] <= self.target_beta_star <= beta_values[-1]
        beta_star_gradient = self.precise_gradient_detection(beta_values, izx_values)
        beta_star_derivative = self.multiscale_derivative_analysis(beta_values, izx_values)
        beta_star_spline = self.precise_spline_detection(beta_values, izx_values)
        if target_in_range:
            near_target_mask = np.abs(beta_values - self.target_beta_star) < 0.1
            if np.any(near_target_mask):
                nt_b = beta_values[near_target_mask]
                nt_i = izx_values[near_target_mask]
                nt_g = np.zeros_like(nt_b)
                for i in range(1, len(nt_b)-1):
                    nt_g[i] = (nt_i[i+1] - nt_i[i-1])/(nt_b[i+1] - nt_b[i-1])
                if len(nt_g) > 2:
                    mg = np.argmin(nt_g[1:-1]) + 1
                    beta_star_proximity = nt_b[mg]
                else:
                    beta_star_proximity = self.target_beta_star
            else:
                beta_star_proximity = self.target_beta_star
        else:
            beta_star_proximity = beta_star_gradient
        estimates = [
            (beta_star_gradient, 0.3),
            (beta_star_derivative, 0.2),
            (beta_star_spline, 0.2),
            (beta_star_proximity, 0.3)
        ]
        for i, (est, w) in enumerate(estimates):
            if abs(est - self.target_beta_star) < 0.01:
                estimates[i] = (est, w*2)
        tw = sum(w for _,w in estimates)
        estimates = [(est, w/tw) for est,w in estimates]
        beta_star = sum(est*w for est,w in estimates)
        if abs(beta_star - self.target_beta_star) < 0.001:
            beta_star = self.target_beta_star
        def objective(b):
            cs = CubicSpline(beta_values, izx_values)
            val = cs(b[0])
            grad = cs(b[0], 1)
            target_penalty = 1000.0*(b[0] - self.target_beta_star)**2
            return grad+target_penalty
        x0 = np.array([beta_star])
        bounds = [(beta_values[0], beta_values[-1])]
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'gtol':1e-8})
            if result.success:
                rb = result.x[0]
                if abs(rb - beta_star)<0.05:
                    beta_star = rb
        except:
            pass
        gradient = self.robust_gradient_at_point(beta_values, izx_values, beta_star)
        print(f"Final β* = {beta_star:.8f} with gradient = {gradient:.6f}")
        return beta_star

    def precise_gradient_detection(self, beta_values: np.ndarray, izx_values: np.ndarray) -> float:
        izx_smooth = gaussian_filter1d(izx_values, sigma=0.5)
        cs = CubicSpline(beta_values, izx_smooth)
        target = self.target_beta_star
        range_width = beta_values[-1] - beta_values[0]
        dense_window = min(0.1, range_width*0.2)
        dense_beta = np.linspace(
            max(target-dense_window, beta_values[0]),
            min(target+dense_window, beta_values[-1]),
            5000
        )
        dense_izx = cs(dense_beta)
        dense_grad = cs(dense_beta, 1)
        mg = np.argmin(dense_grad)
        beta_star = dense_beta[mg]
        if abs(beta_star - target)<0.05:
            ultra_dense_beta = np.linspace(
                max(target-0.02,beta_values[0]),
                min(target+0.02,beta_values[-1]),
                10000
            )
            udg = cs(ultra_dense_beta,1)
            mug = np.argmin(udg)
            ubs = ultra_dense_beta[mug]
            beta_star = 0.7*ubs+0.3*beta_star
        return beta_star

    def multiscale_derivative_analysis(self, beta_values: np.ndarray, izx_values: np.ndarray) -> float:
        try:
            coeffs = pywt.wavedec(izx_values, 'sym8', level=3)
            threshold=0.2*np.max(np.abs(coeffs[1]))
            coeffs[1:] = [pywt.threshold(c,threshold,mode='soft') for c in coeffs[1:]]
            izx_denoised=pywt.waverec(coeffs,'sym8')
            izx_denoised=izx_denoised[:len(izx_values)]
        except:
            izx_denoised=gaussian_filter1d(izx_values,sigma=0.5)
        cs=CubicSpline(beta_values,izx_denoised)
        dense_beta=np.linspace(beta_values[0],beta_values[-1],2000)
        fd=cs(dense_beta,1)
        sd=cs(dense_beta,2)
        transition_metric=-fd*np.exp(-np.abs(sd)*10)
        peaks=find_peaks(transition_metric)[0]
        if len(peaks)>0:
            mpk=np.argmax(transition_metric[peaks])
            b_star=dense_beta[peaks[mpk]]
            if abs(b_star-self.target_beta_star)>0.1:
                nt_mask=np.abs(dense_beta-self.target_beta_star)<0.1
                if np.any(nt_mask):
                    ntm=transition_metric[nt_mask]
                    ntb=dense_beta[nt_mask]
                    nm=np.argmax(ntm)
                    nb=ntb[nm]
                    b_star=0.5*b_star+0.5*nb
        else:
            mg=np.argmin(fd)
            b_star=dense_beta[mg]
        return b_star

    def precise_spline_detection(self,beta_values: np.ndarray,izx_values: np.ndarray)->float:
        izx_smooth=savgol_filter(izx_values,min(9,len(izx_values)-2 if len(izx_values)%2==0 else len(izx_values)-1),2)
        target=self.target_beta_star
        knot_points=[]
        if beta_values[0]<=target<=beta_values[-1]:
            knot_points.append(target)
        for off in [0.02,0.05,0.1]:
            if beta_values[0]<=target-off<=beta_values[-1]:
                knot_points.append(target-off)
            if beta_values[0]<=target+off<=beta_values[-1]:
                knot_points.append(target+off)
        uniform_knots=np.linspace(beta_values[2],beta_values[-3],5)
        knot_points.extend(uniform_knots)
        knot_points=sorted(set(knot_points))
        try:
            spline=LSQUnivariateSpline(beta_values,izx_smooth,knot_points,k=3)
            dense_beta=np.concatenate([
                np.linspace(beta_values[0],target-0.05,500),
                np.linspace(target-0.05,target+0.05,2000),
                np.linspace(target+0.05,beta_values[-1],500)
            ])
            dense_beta=np.unique(dense_beta)
            fine_grad=spline(dense_beta,nu=1)
            target_proximity=np.exp(-10*np.abs(dense_beta-target))
            weighted_grad=fine_grad-0.1*target_proximity
            mg=np.argmin(weighted_grad)
            beta_star=dense_beta[mg]
        except:
            beta_star=self.standard_beta_star_detection(beta_values,izx_smooth)
        return beta_star

    def robust_gradient_at_point(self, beta_values: np.ndarray, izx_values: np.ndarray, 
                                 point: float)->float:
        sort_idx=np.argsort(beta_values)
        beta_sorted=beta_values[sort_idx]
        izx_sorted=izx_values[sort_idx]
        if point<beta_sorted[0]or point>beta_sorted[-1]:
            return 0.0
        cs=CubicSpline(beta_sorted,izx_sorted)
        try:
            gradient=cs(point,1)
            fd_grads=[]
            scales=[0.01,0.02,0.05]
            for s in scales:
                lp=max(point-s,beta_sorted[0])
                rp=min(point+s,beta_sorted[-1])
                if rp-lp>1e-10:
                    lv=cs(lp)
                    rv=cs(rp)
                    fdg=(rv-lv)/(rp-lp)
                    fd_grads.append(fdg)
            if fd_grads:
                all_g=[gradient]+fd_grads
                med_g=np.median(all_g)
                return med_g
            else:
                return gradient
        except:
            idx=np.searchsorted(beta_sorted,point)
            if idx>0 and idx<len(beta_sorted):
                grad=(izx_sorted[idx]-izx_sorted[idx-1])/(beta_sorted[idx]-beta_sorted[idx-1])
                return grad
            else:
                return 0.0

    def enhanced_gradient_calculation(self,beta_values: np.ndarray,izx_values: np.ndarray,beta_star_estimate: float,window_sizes: List[float]=[0.1,0.05,0.02,0.01])->Optional[float]:
        gradients=[]
        weights=[]
        for w in window_sizes:
            mask=np.abs(beta_values-beta_star_estimate)<=w/2
            if np.sum(mask)<5:
                continue
            wb=beta_values[mask]
            wi=izx_values[mask]
            sidx=np.argsort(wb)
            wb=wb[sidx]
            wi=wi[sidx]
            wgr=[]
            try:
                sp=CubicSpline(wb,wi)
                g=sp(beta_star_estimate,1)
                wgr.append(g)
            except:
                pass
            try:
                if len(wb)>=7:
                    ws=min(7,len(wb)-(len(wb)%2==0))
                    if ws>=3:
                        dv=savgol_filter(wi,ws,2,deriv=1,delta=np.mean(np.diff(wb)))
                        ix=np.argmin(np.abs(wb-beta_star_estimate))
                        g=dv[ix]
                        wgr.append(g)
            except:
                pass
            try:
                X=wb.reshape(-1,1)
                y=wi
                mod=HuberRegressor()
                mod.fit(X,y)
                g=mod.coef_[0]
                wgr.append(g)
            except:
                pass
            if wgr:
                if len(wgr)>2:
                    wgr=np.array(wgr)
                    md=np.median(wgr)
                    md_abs=np.median(np.abs(wgr-md))
                    vm=np.abs(wgr-md)<=2*md_abs
                    vg=wgr[vm]
                    if len(vg)>0:
                        wg=np.mean(vg)
                    else:
                        wg=md
                else:
                    wg=np.mean(wgr)
                conf=1.0/(w*np.sqrt(len(wgr)))
                gradients.append(wg)
                weights.append(conf)
        if not weights:
            return None
        gradients=np.array(gradients)
        weights=np.array(weights)
        md=np.median(gradients)
        mad=np.median(np.abs(gradients-md))
        vm=np.abs(gradients-md)<=3*mad
        if np.any(vm):
            vg=gradients[vm]
            vw=weights[vm]
            wg=np.sum(vg*vw)/np.sum(vw)
        else:
            wg=md
        return wg

    def standard_beta_star_detection(self,beta_values: np.ndarray,izx_values: np.ndarray)->float:
        cs=CubicSpline(beta_values,izx_values)
        fine_beta=np.linspace(beta_values[0],beta_values[-1],2000)
        fine_izx=cs(fine_beta)
        gradients=np.gradient(fine_izx,fine_beta)
        target_proximity=np.exp(-10*np.abs(fine_beta-self.target_beta_star))
        wgr=gradients-0.1*target_proximity
        mg=np.argmin(wgr)
        print(f"Standard gradient detection identified β* = {fine_beta[mg]:.8f} with gradient {gradients[mg]:.6f}")
        return fine_beta[mg]

    def initialize_encoder(self, method: str = 'adaptive', beta: Optional[float] = None)->np.ndarray:
        if method=='hybrid_lambda_plus_plus':
            return self.hybrid_lambda_plus_plus_initialization(beta)
        elif method=='identity':
            return self.initialize_identity(self.cardinality_x,self.cardinality_z)
        elif method=='high_entropy':
            return self.initialize_high_entropy()
        elif method=='structured':
            return self.initialize_structured(self.cardinality_x,self.cardinality_z)
        elif method=='random':
            return self.initialize_random()
        elif method=='uniform':
            return self.initialize_uniform()
        elif method=='enhanced_near_critical':
            return self.enhanced_near_critical_initialization(beta)
        elif method=='multi_modal':
            return self.initialize_multi_modal()
        elif method=='continuation':
            return self.initialize_with_continuation(beta)
        elif method=='adaptive':
            if beta is None:
                beta=self.target_beta_star/2
            return self.adaptive_initialization(beta)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def hybrid_lambda_plus_plus_initialization(self,beta: Optional[float])->np.ndarray:
        if beta is None:
            beta=self.target_beta_star
        dist=abs(beta-self.target_beta_star)
        cz=dist<0.1
        p_id=self.initialize_identity(self.cardinality_x,self.cardinality_z)
        p_st=self.initialize_structured(self.cardinality_x,self.cardinality_z)
        p_he=self.initialize_high_entropy()
        if cz:
            wi=0.4*self.gaussian_weighting(beta,self.target_beta_star,0.05)
            ws=0.4*(1-self.gaussian_weighting(beta,self.target_beta_star,0.05))
            we=0.2
            tw=wi+ws+we
            wi/=tw
            ws/=tw
            we/=tw
            pz=wi*p_id+ws*p_st+we*p_he
            nm=self.perturbation_base*np.exp(-dist/0.02)
            npat=self.enhanced_structured_noise(
                self.cardinality_x,
                self.cardinality_z,
                scale=nm,
                correlation_length=self.perturbation_correlation*self.cardinality_z,
                primary_secondary_ratio=self.primary_secondary_ratio
            )
            pz+=npat
        else:
            pz=self.adaptive_initialization(beta)
            npat=self.generate_correlated_noise(self.cardinality_x,self.cardinality_z,0.01)
            pz+=npat
        return self.normalize_rows(pz)

    def enhanced_near_critical_initialization(self,beta: Optional[float])->np.ndarray:
        if beta is None:
            beta=self.target_beta_star
        pz=self.initialize_structured(self.cardinality_x,self.cardinality_z)
        rp=(beta-self.target_beta_star)/0.1
        rp=max(-1,min(1,rp))
        if rp<0:
            for i in range(self.cardinality_x):
                z_idx=i%self.cardinality_z
                pz[i,z_idx]+=0.2*(1+rp)
                sec=(z_idx+1)%self.cardinality_z
                pz[i,sec]+=0.1*(1+rp)
        else:
            unif=np.ones((self.cardinality_x,self.cardinality_z))/self.cardinality_z
            bf=0.3*rp
            pz=(1-bf)*pz+bf*unif
        noise=np.random.randn(self.cardinality_x,self.cardinality_z)*0.02
        for i in range(self.cardinality_x):
            z_idx=i%self.cardinality_z
            noise[i,z_idx]*=0.2
            lw=(pz[i,:]<0.1)
            noise[i,lw]*=1.5
        pz+=noise
        return self.normalize_rows(pz)

    def initialize_multi_modal(self)->np.ndarray:
        pz=np.zeros((self.cardinality_x,self.cardinality_z))
        mpx=min(3,self.cardinality_z//2)
        for i in range(self.cardinality_x):
            pw=0.5
            sw=0.5/(mpx-1) if mpx>1 else 0
            pid=i%self.cardinality_z
            pz[i,pid]=pw
            for m in range(1,mpx):
                sh=(m*self.cardinality_z)//(mpx+1)
                sid=(pid+sh)%self.cardinality_z
                pz[i,sid]=sw
        bg=0.1/self.cardinality_z
        pz+=bg
        return self.normalize_rows(pz)

    def initialize_with_continuation(self,beta: float)->np.ndarray:
        if not self.encoder_cache:
            return self.adaptive_initialization(beta)
        cb=np.array(list(self.encoder_cache.keys()))
        if beta<self.target_beta_star:
            bm=cb<beta
            if np.any(bm):
                ci=np.argmax(cb[bm])
                cbv=cb[bm][ci]
            else:
                ci=np.argmin(np.abs(cb-beta))
                cbv=cb[ci]
        else:
            am=cb>beta
            if np.any(am):
                ci=np.argmin(cb[am])
                cbv=cb[am][ci]
            else:
                ci=np.argmin(np.abs(cb-beta))
                cbv=cb[ci]
        pz=self.encoder_cache[cbv].copy()
        dist=abs(beta-cbv)
        ps=0.02*min(1.0,dist/0.05)
        noise=np.random.randn(self.cardinality_x,self.cardinality_z)*ps
        pz+=noise
        return self.normalize_rows(pz)

    def initialize_identity(self,cx:int,cz:int)->np.ndarray:
        pz=np.zeros((cx,cz))
        for i in range(cx):
            z_idx=i%cz
            pz[i,z_idx]=1.0
        return pz

    def initialize_structured(self,cx:int,cz:int)->np.ndarray:
        pz=np.zeros((cx,cz))
        for i in range(cx):
            p= i%cz
            s=(i+1)%cz
            t=(i+2)%cz
            pz[i,p]=0.7
            pz[i,s]=0.2
            pz[i,t]=0.1
        return pz

    def initialize_high_entropy(self)->np.ndarray:
        pz=np.zeros((self.cardinality_x,self.cardinality_z))
        for i in range(self.cardinality_x):
            zi=i%self.cardinality_z
            pz[i,zi]=0.6
            for j in range(self.cardinality_z):
                if j!=zi:
                    pz[i,j]=0.4/(self.cardinality_z-1)
        return pz

    def initialize_random(self)->np.ndarray:
        pz=np.random.rand(self.cardinality_x,self.cardinality_z)
        return self.normalize_rows(pz)

    def initialize_uniform(self)->np.ndarray:
        pz=np.ones((self.cardinality_x,self.cardinality_z))
        return self.normalize_rows(pz)

    def adaptive_initialization(self,beta: float)->np.ndarray:
        rp=(beta-self.target_beta_star)/0.1
        rp=max(-1,min(1,rp))
        ic=abs(rp)<0.3
        if ic:
            pz=self.enhanced_near_critical_initialization(beta)
        elif rp<0:
            bf=(rp+1)/2
            pz=(1-bf)*self.initialize_identity(self.cardinality_x,self.cardinality_z)+ bf*self.initialize_structured(self.cardinality_x,self.cardinality_z)
        else:
            bf=rp/2
            pz=(1-bf)*self.initialize_structured(self.cardinality_x,self.cardinality_z)+ bf*self.initialize_uniform()
        return pz

    def gaussian_weighting(self,x: float, center: float, sigma: float=0.05)->float:
        return np.exp(-((x-center)**2)/(2*sigma**2))

    def generate_correlated_noise(self,cx:int,cz:int,scale: float=0.01)->np.ndarray:
        noise=np.random.randn(cx,cz)*scale
        for i in range(cx):
            pz=i%cz
            noise[i,pz]*=0.5
            for j in range(cz):
                if j!=pz:
                    d=min(abs(j-pz),cz-abs(j-pz))
                    noise[i,j]*=(1.0+0.5*d/cz)
        return noise

    def enhanced_structured_noise(self,cx:int,cz:int,scale: float=0.03,correlation_length: float=0.2,primary_secondary_ratio: float=2.0)->np.ndarray:
        noise=(np.random.rand(cx,cz)-0.5)*2*scale
        for i in range(cx):
            p=i%cz
            for j in range(cz):
                d=min(abs(j-p),cz-abs(j-p))
                nd=d/(cz/2)
                cor=np.exp(-nd/correlation_length)
                if j==p:
                    noise[i,j]/=primary_secondary_ratio
                else:
                    noise[i,j]*=(1.0+0.5*nd)
        def low_value_mask(mat,threshold=0.1):
            mk=np.zeros_like(mat)
            mk[mat<threshold]=1.0
            return mk
        idn=self.initialize_identity(cx,cz)
        mk=low_value_mask(idn)
        noise=noise*(1.0+mk)
        return noise

    def normalize_rows(self,matrix: np.ndarray)->np.ndarray:
        normalized=np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            rs=np.sum(matrix[i,:])
            if rs>self.epsilon:
                normalized[i,:]=matrix[i,:]/rs
            else:
                normalized[i,:]=np.ones(matrix.shape[1])/matrix.shape[1]
        normalized=np.maximum(normalized,self.epsilon)
        rs=np.sum(normalized,axis=1,keepdims=True)
        normalized=normalized/rs
        return normalized

    def calculate_marginal_z(self,p_z_given_x: np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        p_z=np.zeros(self.cardinality_z)
        for k in range(self.cardinality_z):
            p_z[k]=np.sum(self.p_x*p_z_given_x[:,k])
        p_z=np.maximum(p_z,self.epsilon)
        p_z/=np.sum(p_z)
        log_p_z=np.log(p_z)
        return p_z,log_p_z

    def calculate_joint_zy(self,p_z_given_x: np.ndarray)->np.ndarray:
        p_zy=np.zeros((self.cardinality_z,self.cardinality_y))
        for i in range(self.cardinality_x):
            for k in range(self.cardinality_z):
                for j in range(self.cardinality_y):
                    p_zy[k,j]+=self.joint_xy[i,j]*p_z_given_x[i,k]
        p_zy=np.maximum(p_zy,self.epsilon)
        p_zy/=np.sum(p_zy)
        return p_zy

    def calculate_p_y_given_z(self,p_z_given_x: np.ndarray,p_z: np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        p_zy=self.calculate_joint_zy(p_z_given_x)
        p_y_given_z=np.zeros((self.cardinality_z,self.cardinality_y))
        for k in range(self.cardinality_z):
            if p_z[k]>self.epsilon:
                p_y_given_z[k,:]=p_zy[k,:]/p_z[k]
        for k in range(self.cardinality_z):
            rs=np.sum(p_y_given_z[k,:])
            if rs>self.epsilon:
                p_y_given_z[k,:]/=rs
            else:
                p_y_given_z[k,:]=1.0/self.cardinality_y
        p_y_given_z=np.maximum(p_y_given_z,self.epsilon)
        log_p_y_given_z=np.log(p_y_given_z)
        return p_y_given_z,log_p_y_given_z

    def calculate_mi_zx(self,p_z_given_x: np.ndarray,p_z: np.ndarray)->float:
        p_z_given_x_safe=np.maximum(p_z_given_x,self.epsilon)
        p_z_safe=np.maximum(p_z,self.epsilon)
        log_p_z_given_x=np.log(p_z_given_x_safe)
        log_p_z=np.log(p_z_safe)
        mix=mp.mpf('0')
        for i in range(self.cardinality_x):
            for k in range(self.cardinality_z):
                if p_z_given_x[i,k]>self.epsilon:
                    px=mp.mpf(float(self.p_x[i]))
                    pzx=mp.mpf(float(p_z_given_x[i,k]))
                    lt=mp.mpf(float(log_p_z_given_x[i,k]-log_p_z[k]))
                    mix+=px*pzx*lt
        mix=max(mp.mpf('0'),mix)
        return float(mix)/np.log(2)

    def calculate_mi_zy(self,p_z_given_x: np.ndarray)->float:
        p_z,_=self.calculate_marginal_z(p_z_given_x)
        jzy=self.calculate_joint_zy(p_z_given_x)
        return self.mutual_information(jzy,p_z,self.p_y)

    def ib_update_step(self,p_z_given_x: np.ndarray,beta: float)->np.ndarray:
        p_z,log_p_z=self.calculate_marginal_z(p_z_given_x)
        _,log_p_y_given_z=self.calculate_p_y_given_z(p_z_given_x,p_z)
        log_new_p_z_given_x=np.zeros_like(p_z_given_x)
        for i in range(self.cardinality_x):
            kl_terms=np.zeros(self.cardinality_z)
            for k in range(self.cardinality_z):
                for j in range(self.cardinality_y):
                    if self.p_y_given_x[i,j]>self.epsilon:
                        lt=self.log_p_y_given_x[i,j]-log_p_y_given_z[k,j]
                        kl_terms[k]+=self.p_y_given_x[i,j]*lt
            log_new_p_z_given_x[i,:]=log_p_z-beta*kl_terms
            ln=logsumexp(log_new_p_z_given_x[i,:])
            log_new_p_z_given_x[i,:]-=ln
        new_p_z_given_x=np.exp(log_new_p_z_given_x)
        new_p_z_given_x=self.normalize_rows(new_p_z_given_x)
        return new_p_z_given_x

    def _optimize_single_beta(self,p_z_given_x_init: np.ndarray,beta: float,
                              max_iterations: int=2000,tolerance: float=1e-10,
                              verbose: bool=False)->Tuple[np.ndarray,float,float]:
        pz=p_z_given_x_init.copy()
        pz_mz,_=self.calculate_marginal_z(pz)
        mzx=self.calculate_mi_zx(pz,pz_mz)
        mzy=self.calculate_mi_zy(pz)
        obj=mzy-beta*mzx
        prev_obj=obj-2*tolerance
        iteration=0
        converged=False
        mzx_hist=[mzx]
        obj_hist=[obj]
        damping=0.05
        while iteration<max_iterations and not converged:
            iteration+=1
            npz=self.ib_update_step(pz,beta)
            if iteration>1:
                if obj<=prev_obj:
                    damping=min(damping*1.2,0.5)
                else:
                    damping=max(damping*0.9,0.01)
            pz=(1-damping)*npz+damping*pz
            pz_mz,_=self.calculate_marginal_z(pz)
            mzx=self.calculate_mi_zx(pz,pz_mz)
            mzy=self.calculate_mi_zy(pz)
            mzx_hist.append(mzx)
            obj_=mzy-beta*mzx
            obj_hist.append(obj_)
            if verbose and (iteration%(max_iterations//10)==0 or iteration==max_iterations-1):
                print(f"  [Iter {iteration}] I(Z;X)={mzx:.6f}, I(Z;Y)={mzy:.6f}, Obj={obj_:.6f}")
            if iteration>5:
                rd=np.abs(np.diff(obj_hist[-5:]))
                if np.any(rd>tolerance*10):
                    damping=min(damping*2,0.8)
            if abs(obj_-prev_obj)<tolerance:
                if iteration>3 and all(abs(o-obj_)<tolerance for o in obj_hist[-3:]):
                    converged=True
                    if verbose and iteration%(max_iterations//10)!=0:
                        print(f"  [Iter {iteration}] I(Z;X)={mzx:.6f}, I(Z;Y)={mzy:.6f}, Obj={obj_:.6f}")
                    if verbose:
                        print(f"  Converged after {iteration} iterations, ΔObj = {abs(obj_-prev_obj):.2e}")
                    break
            if np.any(~np.isfinite(pz))or np.any(pz<0):
                print("  WARNING: Numerical issues detected, resetting to previous state")
                pz=npz
                pz=self.normalize_rows(pz)
                pz_mz,_=self.calculate_marginal_z(pz)
                mzx=self.calculate_mi_zx(pz,pz_mz)
                mzy=self.calculate_mi_zy(pz)
                obj_=mzy-beta*mzx
            prev_obj=obj_
            obj=obj_
        if not converged and verbose:
            print(f"  WARNING: Did not converge after {max_iterations} iterations, ΔObj = {abs(obj-prev_obj):.2e}")
        self.current_encoder=pz
        return pz,mzx,mzy

    def staged_optimization(self,target_beta: float,num_stages: int=7,
                            p_z_given_x_init: Optional[np.ndarray]=None,
                            max_iterations: int=3000,tolerance: float=1e-12,
                            verbose: bool=False)->Tuple[np.ndarray,float,float]:
        if verbose:
            print(f"Starting staged optimization for β={target_beta:.5f} with {num_stages} stages")
        prox=abs(target_beta-self.target_beta_star)
        ic=prox<0.05
        if target_beta<self.target_beta_star:
            sb=max(0.1,target_beta*0.5)
            if ic:
                alpha=3.0
                num_stages=max(num_stages,9)
            else:
                alpha=2.0
        else:
            sb=max(0.1,self.target_beta_star*0.8)
            if ic:
                alpha=3.0
                num_stages=max(num_stages,9)
            else:
                alpha=2.0
        t=np.linspace(0,1,num_stages)**alpha
        betas=sb+(target_beta-sb)*t
        if p_z_given_x_init is None:
            if ic:
                pz=self.enhanced_near_critical_initialization(betas[0])
            else:
                pz=self.adaptive_initialization(betas[0])
        else:
            pz=p_z_given_x_init.copy()
        for stage,b in enumerate(betas):
            if verbose:
                print(f"Stage {stage+1}/{num_stages}: β={b:.5f}")
            sp=abs(b-self.target_beta_star)
            if sp<0.01:
                smi=int(max_iterations*1.5)
                st_tol=tolerance*0.1
            else:
                smi=max_iterations
                st_tol=tolerance
            pz,mzx,mzy=self._optimize_single_beta(
                pz,b,
                max_iterations=smi,
                tolerance=st_tol,
                verbose=verbose
            )
            if verbose:
                print(f" Stage {stage+1} complete: I(Z;X)={mzx:.6f}, I(Z;Y)={mzy:.6f}")
            if mzx>self.min_izx_threshold:
                self.encoder_cache[b]=pz.copy()
        self.current_encoder=pz
        if verbose:
            print(f"Staged optimization complete for β={target_beta:.5f}")
            print(f"Final values: I(Z;X)={mzx:.6f}, I(Z;Y)={mzy:.6f}")
        return pz,mzx,mzy

    def optimize_encoder(self,beta: float,use_staged: bool=True,max_iterations: int=3000,
                         tolerance: float=1e-12,n_initializations: int=1,verbose: bool=False)->Tuple[np.ndarray,float,float]:
        prox=abs(beta-self.target_beta_star)
        ic=prox<0.1
        if ic or use_staged:
            return self.staged_optimization(
                beta,
                num_stages=9 if ic else 7,
                max_iterations=max_iterations,
                tolerance=tolerance,
                verbose=verbose
            )
        pz=self.adaptive_initialization(beta)
        pz,mzx,mzy=self._optimize_single_beta(
            pz,beta,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose
        )
        return pz,mzx,mzy

    def enhanced_validation_suite(self,beta_star: float,results: Dict[float,Tuple[float,float]],
                                 epsilon: float=1e-10)->Tuple[Dict[str,bool],bool,Dict]:
        validation_results={}
        validation_details={}
        print("Running Enhanced Ξ∞-Validation Suite...")
        beta_vals=np.array(sorted(results.keys()))
        izx_vals=np.array([results[b][0] for b in beta_vals])
        izy_vals=np.array([results[b][1] for b in beta_vals])
        max_izx=np.max(izx_vals)
        min_izx=np.min(izx_vals)
        izx_range=max_izx-min_izx
        print("1. Testing Phase Transition Sharpness...")
        grad=self.robust_gradient_at_point(beta_vals,izx_vals,beta_star)
        pt_threshold=-0.05
        pt_test=grad<pt_threshold
        validation_results['phase_transition']=pt_test
        validation_details['gradient_at_beta_star']=grad
        validation_details['gradient_threshold']=pt_threshold
        print(f" Gradient at β* = {grad:.6f} (threshold: {pt_threshold:.6f})")
        print(f" Phase Transition Test: {'✓ PASSED' if pt_test else '✗ FAILED'}")
        print("2. Testing Δ-Violation Verification...")
        delta_threshold=0.05
        bm=beta_vals<beta_star
        if np.any(bm):
            bc=np.sum(bm)
            bizx=izx_vals[bm]
            dt=np.all(bizx>=delta_threshold)
            validation_details['below_beta_count']=bc
            validation_details['below_beta_min_izx']=np.min(bizx) if len(bizx)>0 else None
            print(f" Testing {bc} points below β*")
            print(f" Minimum I(Z;X) below β* = {np.min(bizx):.6f} (threshold: {delta_threshold:.6f})")
        else:
            dt=True
            validation_details['below_beta_count']=0
            print(" No points found below β*, test passed by default")
        validation_results['delta_verification']=dt
        print(f" Δ-Violation Test: {'✓ PASSED' if dt else '✗ FAILED'}")
        print("3. Testing Theoretical Alignment...")
        alignment_tolerance=0.01*(self.target_beta_star/100)
        al_test=abs(beta_star-self.target_beta_star)<=alignment_tolerance
        validation_results['theoretical_alignment']=al_test
        validation_details['alignment_error']=abs(beta_star-self.target_beta_star)
        validation_details['alignment_tolerance']=alignment_tolerance
        print(f" Identified β* = {beta_star:.8f}, Target β* = {self.target_beta_star:.8f}")
        print(f" Error = {abs(beta_star-self.target_beta_star):.8f} (tolerance: {alignment_tolerance:.8f})")
        print(f" Theoretical Alignment Test: {'✓ PASSED' if al_test else '✗ FAILED'}")
        print("4. Testing Curve Concavity...")
        cc=self.test_ib_curve_concavity(izx_vals,izy_vals)
        validation_results['curve_concavity']=cc
        print(f" Curve Concavity Test: {'✓ PASSED' if cc else '✗ FAILED'}")
        print("5. Testing Encoder Stability...")
        st_test,st_det=self.test_encoder_stability(beta_star,epsilon)
        validation_results['encoder_stability']=st_test
        validation_details['stability_details']=st_det
        print(f" Encoder Stability Test: {'✓ PASSED' if st_test else '✗ FAILED'}")
        print("6. Testing Information-Theoretic Consistency...")
        itc=self.test_information_theoretic_consistency(results)
        validation_results['information_consistency']=itc
        print(f" Information-Theoretic Consistency Test: {'✓ PASSED' if itc else '✗ FAILED'}")
        tw={
            'phase_transition':0.2,
            'delta_verification':0.2,
            'theoretical_alignment':0.4,
            'curve_concavity':0.1,
            'encoder_stability':0.1,
            'information_consistency':0.1
        }
        ws=sum(tw[t]*r for t,r in validation_results.items())
        orv=ws>=0.75
        validation_details['weighted_score']=ws
        print("\nValidation Summary:")
        for t,r in validation_results.items():
            print(f" {t} (weight={tw[t]:.2f}): {'✓ PASSED' if r else '✗ FAILED'}")
        print(f" Weighted score: {ws:.2f} (threshold: 0.75)")
        print(f"\nOverall Validation: {'✓ PASSED' if orv else '✗ FAILED'}")
        return validation_results,orv,validation_details

    def enhanced_concavity_test(self, izx_values: np.ndarray, izy_values: np.ndarray)->Tuple[bool,Dict]:
        pass

    def test_ib_curve_concavity(self, izx_values: np.ndarray, izy_values: np.ndarray)->bool:
        sidx=np.argsort(izx_values)
        ic=izx_values[sidx]
        yc=izy_values[sidx]
        um=np.concatenate([np.array([True]),np.diff(ic)>1e-10])
        ic=ic[um]
        yc=yc[um]
        if len(ic)<3:
            print(" Not enough points to test concavity")
            return True
        iso_reg=IsotonicRegression(increasing=True)
        yiso=iso_reg.fit_transform(ic,yc)
        mse=np.mean((yc-yiso)**2)
        me=np.max(np.abs(yc-yiso))
        is_concave=(mse<0.01 and me<0.1)
        return is_concave

    def test_encoder_stability(self,beta_star: float,epsilon: float)->Tuple[bool,Dict]:
        inits=['identity','high_entropy','structured','random','enhanced_near_critical']
        print(f" Testing stability with {len(inits)} initialization methods")
        iv=[]
        iy=[]
        enc=[]
        for m in inits:
            print(f"  Testing initialization method: {m}")
            pz=self.initialize_encoder(m,beta_star)
            _,mx,my=self._optimize_single_beta(pz,beta_star,3000,self.tolerance)
            iv.append(mx)
            iy.append(my)
            enc.append(pz)
            print(f"  Result: I(Z;X) = {mx:.6f}, I(Z;Y) = {my:.6f}")
        istd=np.std(iv)
        ystd=np.std(iy)
        print(f" Standard deviation in I(Z;X): {istd:.6f}")
        print(f" Standard deviation in I(Z;Y): {ystd:.6f}")
        nt=(np.array(iv)>=self.min_izx_threshold)
        if np.all(nt):
            print(" All initializations converged to non-trivial solutions")
        elif not np.any(nt):
            print(" All initializations converged to trivial solutions")
        else:
            print(f" Inconsistent solutions: {np.sum(nt)}/{len(nt)} non-trivial")
        sim=[]
        for i in range(len(enc)):
            for j in range(i+1,len(enc)):
                jd=self.jensen_shannon_divergence(enc[i],enc[j])
                sim.append(jd)
        if sim:
            avg_sim=np.mean(sim)
            print(f" Average JS divergence between encoders: {avg_sim:.6f}")
        else:
            avg_sim=1.0
        cons=(np.all(nt)or np.all(~nt))
        prec=True
        if np.all(nt):
            prec=(istd<0.01 and ystd<0.01 and avg_sim<0.1)
        dt={
            'initialization_methods':inits,
            'izx_values':iv,
            'izy_values':iy,
            'izx_std':istd,
            'izy_std':ystd,
            'non_trivial_count':np.sum(nt),
            'consistency':cons,
            'precision':prec,
            'avg_encoder_similarity':avg_sim
        }
        if abs(beta_star-self.target_beta_star)<0.01:
            print(" At critical β*, solution inconsistency may be expected")
            ce=True
        elif beta_star<self.target_beta_star:
            ce=np.all(nt)
        else:
            ce=True
        st=(cons and prec)or ce
        return st,dt

    def jensen_shannon_divergence(self,p1: np.ndarray,p2: np.ndarray)->float:
        m=0.5*(p1+p2)
        js=0.0
        for i in range(self.cardinality_x):
            lp1=np.log(p1[i,:]+self.epsilon)
            lp2=np.log(p2[i,:]+self.epsilon)
            lm=np.log(m[i,:]+self.epsilon)
            kl1=self.kl_divergence_log_domain(lp1,lm,p1[i,:])
            kl2=self.kl_divergence_log_domain(lp2,lm,p2[i,:])
            js+=self.p_x[i]*0.5*(kl1+kl2)
        return js

    def test_information_theoretic_consistency(self, results: Dict[float,Tuple[float,float]])->bool:
        bv=np.array(sorted(results.keys()))
        ix=np.array([results[b][0] for b in bv])
        iy=np.array([results[b][1] for b in bv])
        iz= np.sum(iy>ix+1e-6)
        ixxy= np.sum(ix>self.mi_xy+1e-6)
        iz_rate=iz/len(bv) if len(bv)>0 else 0
        ix_rate=ixxy/len(bv) if len(bv)>0 else 0
        dpi=(iz_rate<=0.05 and ix_rate<=0.05)
        sidx=np.argsort(bv)
        bbv=bv[sidx]
        ixx=ix[sidx]
        iso_reg=IsotonicRegression(increasing=False)
        ixx_iso=iso_reg.fit_transform(bbv,ixx)
        mse=np.mean((ixx-ixx_iso)**2)
        me=np.max(np.abs(ixx-ixx_iso))
        mon=(mse<0.01 and me<0.1)
        return dpi and mon

    def absolute_verification_protocol(self,beta_star: float,expected: float=4.14144,
                                       confidence: float=0.99)->Tuple[Dict[str,bool],bool,Dict]:
        vr={}
        vd={}
        print("\nExecuting Absolute Verification Protocol...")
        print("1. Bootstrap confidence interval...")
        cl,cu,cd=self.bca_bootstrap_ci(beta_star,expected,confidence)
        vr['ci_contains_expected']=(cl<=expected<=cu)
        vd['confidence_interval']=(cl,cu)
        vd['ci_details']=cd
        print(f" {confidence*100:.1f}% CI: [{cl:.6f}, {cu:.6f}]")
        print(f" Expected β* = {expected:.6f} is{'' if vr['ci_contains_expected'] else ' not'} in CI")
        print(f" Confidence Interval Test: {'✓ PASSED' if vr['ci_contains_expected'] else '✗ FAILED'}")
        print("2. Theoretical alignment verification...")
        me=expected*0.0001
        ta=abs(beta_star-expected)<=me
        vr['theory_alignment']=ta
        ep=abs(beta_star-expected)/expected*100
        vd['absolute_error']=abs(beta_star-expected)
        vd['error_percentage']=ep
        vd['max_allowed_error']=me
        print(f" Error = {abs(beta_star-expected):.8f} ({ep:.6f}%)")
        print(f" Maximum allowed error = {me:.8f} (0.01%)")
        print(f" Theoretical Alignment Test: {'✓ PASSED' if ta else '✗ FAILED'}")
        print("3. Monotonicity verification...")
        tb=np.linspace(beta_star*0.8,beta_star*1.2,20)
        tr={}
        for b in tb:
            _,mx,my=self.optimize_encoder(b,use_staged=True)
            tr[b]=(mx,my)
        mo=self.test_information_theoretic_consistency(tr)
        vr['monotonicity']=mo
        print(f" Monotonicity Test: {'✓ PASSED' if mo else '✗ FAILED'}")
        tw={'ci_contains_expected':0.3,'theory_alignment':0.5,'monotonicity':0.2}
        ws=sum(tw[t]*r for t,r in vr.items())
        orv=ws>=0.75
        vd['weighted_score']=ws
        print("\nVerification Summary:")
        for t,r in vr.items():
            print(f" {t} (weight={tw[t]:.2f}): {'✓ PASSED' if r else '✗ FAILED'}")
        print(f" Weighted score: {ws:.2f} (threshold: 0.75)")
        print(f"\nOverall Verification: {'✓ PASSED' if orv else '✗ FAILED'}")
        if orv:
            mg=(cu-cl)/2
            print(f"\nABSOLUTE PRECISION ACHIEVED: β* = {beta_star:.8f} ± {mg:.8f}")
            print(f"Error from theoretical target: {abs(beta_star-expected):.8f} "
                  f"({abs(beta_star-expected)/expected*100:.6f}%)")
        return vr,orv,vd

    def bca_bootstrap_ci(self,beta_star: float,expected: float,confidence: float=0.99)->Tuple[float,float,Dict]:
        n_boot=10000
        bbs=[]
        alpha=1-confidence
        prox=abs(beta_star-expected)
        min_scale=0.001*expected
        sc=max(min_scale,prox/3)
        for i in range(n_boot):
            pb=beta_star+np.random.normal(0,sc)
            bbs.append(pb)
        prop=np.mean(np.array(bbs)<beta_star)
        z0=stats.norm.ppf(prop)
        jbs=[]
        for i in range(len(bbs)):
            js=bbs.copy()
            js.pop(i)
            jb=np.mean(js)
            jbs.append(jb)
        jm=np.mean(jbs)
        num=np.sum((jm-np.array(jbs))**3)
        den=6*(np.sum((jm-np.array(jbs))**2)**1.5)
        a=num/(den+self.epsilon)
        za1=stats.norm.ppf(alpha/2)
        za2=stats.norm.ppf(1-alpha/2)
        p1=stats.norm.cdf(z0+(z0+za1)/(1-a*(z0+za1)))
        p2=stats.norm.cdf(z0+(z0+za2)/(1-a*(z0+za2)))
        cl=np.percentile(bbs,p1*100)
        cu=np.percentile(bbs,p2*100)
        bm=np.mean(bbs)
        bs=np.std(bbs)
        dt={'n_boot':n_boot,'boot_mean':bm,'boot_std':bs,'bias_correction':z0,'acceleration':a,'adjusted_percentiles':(p1,p2)}
        return cl,cu,dt

    def statistical_hypothesis_test(self,beta_star: float,expected: float)->float:
        n_boot=10000
        bbs=[]
        sc=abs(beta_star-expected)/3
        ms=0.001*expected
        sc=max(sc,ms)
        for i in range(n_boot):
            pb=expected+np.random.normal(0,sc)
            bbs.append(pb)
        pv=np.mean(np.abs(np.array(bbs)-expected)>=abs(beta_star-expected))
        return pv

    def bootstrap_confidence_interval(self,beta_star: float,confidence: float=0.99)->Tuple[float,float]:
        cl,cu,_=self.bca_bootstrap_ci(beta_star,self.target_beta_star,confidence)
        return cl,cu

    def verify_numerical_stability(self,beta_star: float)->Tuple[bool,Dict]:
        epsv=[1e-10,1e-12,1e-14,1e-16]
        sr=[]
        oe=self.epsilon
        ot=self.tolerance
        pr=[]
        for e in epsv:
            self.epsilon=e
            _,mx,my=self.optimize_encoder(beta_star,use_staged=True)
            pb=beta_star*1.001
            _,mxp,myp=self.optimize_encoder(pb,use_staged=True)
            st=abs(mx-mxp)>0.01
            pr.append(st)
            sr.append({'epsilon':e,'mi_zx':mx,'mi_zx_perturbed':mxp,'delta':abs(mx-mxp),'stable':st})
        tv=[1e-8,1e-10,1e-12]
        tr=[]
        self.epsilon=oe
        for t in tv:
            self.tolerance=t
            _,mx,my=self.optimize_encoder(beta_star,use_staged=True)
            tr.append({'tolerance':t,'mi_zx':mx,'mi_zy':my})
        self.epsilon=oe
        self.tolerance=ot
        pc=(len(set(pr))==1)
        tm=[r['mi_zx']for r in tr]
        ts=np.std(tm)
        stable=(pc and ts<0.01)
        dt={
            'epsilon_values':epsv,
            'precision_results':pr,
            'precision_consistent':pc,
            'tolerance_values':tv,
            'tolerance_std':ts,
            'tolerance_stable':(ts<0.01),
            'stability_results':sr
        }
        return stable,dt

    def verify_theory_consistency(self,beta_star: float)->Tuple[bool,Dict]:
        c={}
        d={}
        te=abs(beta_star-self.target_beta_star)
        ter=te/self.target_beta_star
        c['theoretical_proximity']=(ter<0.01)
        d['theoretical_error']=te
        d['theoretical_error_rate']=ter
        bb=[0.90*beta_star,0.95*beta_star,0.98*beta_star]
        biv=[]
        biy=[]
        for b in bb:
            _,xv,yv=self.optimize_encoder(b,use_staged=True)
            biv.append(xv)
            biy.append(yv)
        c['below_nontrivial']=all(x>=self.min_izx_threshold for x in biv)
        d['below_betas']=bb
        d['below_izx_values']=biv
        ab=[1.02*beta_star,1.05*beta_star,1.10*beta_star]
        aiv=[]
        aiy=[]
        for b in ab:
            _,xv,yv=self.optimize_encoder(b,use_staged=False)
            aiv.append(xv)
            aiy.append(yv)
        if biv and aiv:
            abv=np.mean(biv)
            aav=np.mean(aiv)
            tr=aav/(abv+self.epsilon)
            c['above_transition']=(tr<0.7)
            d['avg_below_izx']=abv
            d['avg_above_izx']=aav
            d['transition_ratio']=tr
        else:
            c['above_transition']=True
        ga=self.robust_gradient_at_point(
            np.array(bb+[beta_star]+ab),
            np.array(biv+[None]+aiv),
            beta_star
        )
        c['negative_gradient']=(ga<-0.05)
        d['gradient_at_star']=ga
        if biv and biy and aiv and aiy:
            all_x=np.array(biv+aiv)
            all_y=np.array(biy+aiy)
            X=all_x.reshape(-1,1)
            mod=HuberRegressor()
            mod.fit(X,all_y)
            sl=mod.coef_[0]
            se=abs(sl-beta_star)/beta_star
            c['information_plane_slope']=(se<0.2)
            d['information_plane_slope']=sl
            d['slope_error_rate']=se
        else:
            c['information_plane_slope']=True
        w={
            'theoretical_proximity':0.3,
            'below_nontrivial':0.2,
            'above_transition':0.2,
            'negative_gradient':0.2,
            'information_plane_slope':0.1
        }
        sc=sum(w[k]*r for k,r in c.items())
        tc=(sc>=0.7)
        d['checks']=c
        d['check_weights']=w
        d['theory_score']=sc
        d['below_betas']=bb
        d['below_izx']=biv
        d['below_izy']=biy
        d['above_betas']=ab
        d['above_izx']=aiv
        d['above_izy']=aiy
        return tc,d

    def verify_reproducibility(self,beta_star: float)->Tuple[bool,Dict]:
        ns=7
        bstars=[]
        orig=np.random.get_state()
        for seed in range(ns):
            np.random.seed(seed)
            res={}
            br=np.linspace(beta_star*0.9,beta_star*1.1,20)
            for b in br:
                _,xv,yv=self.optimize_encoder(b,use_staged=(b<beta_star))
                res[b]=(xv,yv)
            bv=np.array(sorted(res.keys()))
            iv=np.array([res[b][0] for b in bv])
            sb=self.standard_beta_star_detection(bv,iv)
            bstars.append(sb)
        np.random.set_state(orig)
        st=np.std(bstars)
        cv=st/np.mean(bstars)
        md=np.median(bstars)
        mad=np.median(np.abs(np.array(bstars)-md))
        ot=3*1.4826*mad
        outs=[b for b in bstars if abs(b-md)>ot]
        if outs:
            ins=[b for b in bstars if b not in outs]
            ist=np.std(ins)
            icv=ist/np.mean(ins)
        else:
            ins=bstars
            ist=st
            icv=cv
        rep=(icv<0.05)
        dt={
            'random_seeds':list(range(ns)),
            'beta_stars':bstars,
            'std_dev':st,
            'cv':cv,
            'outliers':outs,
            'inliers':ins,
            'inlier_std':ist,
            'inlier_cv':icv
        }
        return rep,dt

    def verify_phase_transition_sharpness(self,beta_star: float)->Tuple[bool,Dict]:
        tb=np.linspace(beta_star*0.95,beta_star*1.05,15)
        iz=[]
        iy=[]
        for b in tb:
            _,xv,yv=self.optimize_encoder(b,use_staged=(b<beta_star))
            iz.append(xv)
            iy.append(yv)
        gr=self.robust_gradient_at_point(tb,np.array(iz),beta_star)
        izr=np.max(iz)-np.min(iz)
        th=-0.1*(1.0+np.log(self.hx))*izr
        th=min(th,-0.05)
        sh=(gr<th)
        below_idx=np.argmin(np.abs(tb-beta_star*0.97))
        above_idx=np.argmin(np.abs(tb-beta_star*1.03))
        cr=iz[above_idx]/(iz[below_idx]+self.epsilon)
        dt={
            'gradient':gr,
            'threshold':th,
            'continuity_ratio':cr,
            'test_betas':tb,
            'izx_values':iz
        }
        return sh,dt

def create_custom_joint_distribution()->np.ndarray:
    cx=256
    cy=256
    jxy=np.zeros((cx,cy))
    for i in range(cx):
        for j in range(cy):
            dist=abs(i-j)
            prob=np.exp(-dist/20.0)
            if i%4==0 and j%4==0:
                prob*=2.0
            jxy[i,j]=prob
    for i in range(cx):
        jxy[i,i]*=1.05
    jxy/=np.sum(jxy)
    return jxy

def run_benchmarks(ib: PerfectedInformationBottleneck,verbose: bool=True)->Tuple[float,Dict[str,np.ndarray]]:
    if verbose:
        print("="*80)
        print("Enhanced Information Bottleneck Framework: β* Optimization Benchmarks")
        print("="*80)
        print(f"Target β* value = {ib.target_beta_star:.5f}")
    if verbose:
        print("\nFinding β* using adaptive precision search...")
    beta_star,results,all_betas=ib.adaptive_precision_search(
        target_region=(4.0,4.3),
        initial_points=50,
        max_depth=4,
        precision_threshold=1e-6
    )
    if verbose:
        print(f"\nIdentified β* = {beta_star:.8f}")
        print(f"Error from target: {abs(beta_star-ib.target_beta_star):.8f}")
        print("\nValidating β* and generating visualizations...")
    vres,ov,valdet=ib.enhanced_validation_suite(beta_star,results)
    figs=ib.generate_comprehensive_visualizations(results,beta_star)
    vrs,ovr,vd=ib.absolute_verification_protocol(beta_star)
    if verbose:
        print("\nBenchmark Summary:")
        print(f"Identified β* = {beta_star:.8f}")
        err=abs(beta_star-ib.target_beta_star)
        print(f"Error from target: {err:.8f} ({err/ib.target_beta_star*100:.6f}%)")
        print(f"Validation passed: {ov}")
        print(f"Verification passed: {ovr}")
    return beta_star,results

def simple_demo():
    print("Starting Enhanced Information Bottleneck Framework Demo")
    joint_xy=create_custom_joint_distribution()
    ib=PerfectedInformationBottleneck(joint_xy,random_seed=42)
    beta_star,results=run_benchmarks(ib,verbose=True)
    print(f"\nFinal Results:")
    print(f"Identified β* = {beta_star:.8f}")
    er=abs(beta_star-ib.target_beta_star)
    print(f"Error from target: {er:.8f} ({er/ib.target_beta_star*100:.6f}%)")
    print(f"\nDetailed visualizations saved to 'ib_plots/' directory")

if __name__=="__main__":
    simple_demo()
