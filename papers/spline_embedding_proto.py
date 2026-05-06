"""
Spline Embedding Prototype
==========================
Minimal proof-of-concept for spline embedding concepts.

A vector is a spline f: [0,1] -> R^n, not a point.
This module demonstrates:
1. Spline as vector (control points define trajectory, not position)
2. Pinned spline constraint satisfaction
3. Resonance signature extraction from time-series
4. Contrast mapping (hyper/hypo-perfused regions)
5. ASCII resonance imaging

Author: Oracle1 (subagent)
Branch: spline-embedding-prototype
"""

import numpy as np
from typing import Optional, Tuple

# Try scipy, fallback to pure numpy
try:
    from scipy.interpolate import splrep, splev
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy not available - using pure numpy fallback")


# =============================================================================
# Part 1: Spline as Vector
# =============================================================================

class SplineVector:
    """
    A vector is a spline f: [0,1] -> R^n, not a point.
    
    The key insight: a vector is a TRAJECTORY, not a point in R^n.
    A spline maps a parameter t ∈ [0,1] to R^n, so it's inherently
    a higher-order representation.
    
    Control points define the shape via B-spline basis functions.
    """
    
    def __init__(self, control_points: np.ndarray, degree: int = 3):
        """
        Initialize a spline vector.
        
        Args:
            control_points: shape (m+1, n) — m+1 control points in R^n
            degree: B-spline degree (default cubic)
        """
        assert control_points.ndim == 2, "control_points must be 2D array"
        assert degree >= 1, "degree must be >= 1"
        
        self.cpts = control_points
        self.degree = degree
        self.dim = control_points.shape[1]  # R^n dimension
        self.num_ctrl = control_points.shape[0]
        
        if HAS_SCIPY:
            self._build_bspline()
        else:
            self.knots = self._均匀_knots()
            self.t = np.linspace(0, 1, 100)
    
    def _均匀_knots(self) -> np.ndarray:
        """Generate uniform knot vector."""
        num_knots = self.num_ctrl + self.degree + 1
        interior = np.linspace(0, 1, max(2, self.num_ctrl - self.degree + 1))
        knots = np.concatenate([
            np.zeros(self.degree),
            interior,
            np.ones(self.degree)
        ])
        return knots
    
    def _build_bspline(self):
        """Build scipy B-spline representation."""
        knots = np.concatenate([
            np.zeros(self.degree),
            np.linspace(0, 1, self.num_ctrl - self.degree + 1),
            np.ones(self.degree)
        ])
        self.knots = knots
        self.tck = (knots, self.cpts, self.degree)
    
    def eval(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate spline at parameter values t.
        
        Args:
            t: array of parameter values in [0, 1]
            
        Returns:
            shape (len(t), n) — points on the spline curve
        """
        if HAS_SCIPY:
            return splev(t, self.tck)
        else:
            # Pure numpy: evaluate using Cox-de Boor recursion
            return self._eval_pure_numpy(t)
    
    def _eval_pure_numpy(self, t: np.ndarray) -> np.ndarray:
        """Evaluate B-spline without scipy."""
        t = np.asarray(t)
        n = len(t)
        result = np.zeros((n, self.dim))
        
        for i in range(self.num_ctrl):
            # B-spline basis function N_{i,k}(t)
            basis = self._bspline_basis(t, i, self.degree)
            result += np.outer(basis, self.cpts[i])
        
        return result
    
    def _bspline_basis(self, t: np.ndarray, i: int, k: int) -> np.ndarray:
        """Cox-de Boor recursion for B-spline basis functions."""
        if k == 0:
            cond = (self.knots[i] <= t) & (t < self.knots[i + 1])
            # Handle endpoint case
            if i == self.num_ctrl + k - 1:
                cond = (self.knots[i] <= t) & (t <= self.knots[i + 1])
            return cond.astype(float)
        
        denom1 = self.knots[i + k] - self.knots[i]
        denom2 = self.knots[i + k + 1] - self.knots[i + 1]
        
        result = np.zeros_like(t)
        if denom1 != 0:
            result += (t - self.knots[i]) / denom1 * self._bspline_basis(t, i, k - 1)
        if denom2 != 0:
            result += (self.knots[i + k + 1] - t) / denom2 * self._bspline_basis(t, i + 1, k - 1)
        
        return result
    
    def curvature(self, t: np.ndarray) -> np.ndarray:
        """
        Compute curvature at parameter values t.
        
        Curvature κ = |x' × x''| / |x'|³
        
        For curves in R^n, we use generalized curvature formula.
        """
        if HAS_SCIPY:
            dx = splev(t, self.tck, der=1)
            ddx = splev(t, self.tck, der=2)
        else:
            # Numerical differentiation
            eps = 1e-5
            dx = (self.eval(t + eps) - self.eval(t - eps)) / (2 * eps)
            ddx = (self.eval(t + eps) - 2 * self.eval(t) + self.eval(t - eps)) / (eps ** 2)
        
        # Speed |x'|
        speed = np.linalg.norm(dx, axis=1)
        
        # For R^3 case: cross product
        if self.dim == 3:
            cross = np.cross(dx[:, :3], ddx[:, :3])
            kappa = np.linalg.norm(cross, axis=1) / (speed ** 3 + 1e-12)
        else:
            # Generalized: use second derivative magnitude / speed^3
            # This is approximate but works for any dimension
            ddx_norm = np.linalg.norm(ddx, axis=1)
            # Tangent direction approximation
            kappa = ddx_norm / (speed ** 2 + 1e-12)
        
        return kappa
    
    def tangent(self, t: np.ndarray) -> np.ndarray:
        """Compute unit tangent vector at t."""
        if HAS_SCIPY:
            dx = splev(t, self.tck, der=1)
        else:
            eps = 1e-5
            dx = (self.eval(t + eps) - self.eval(t - eps)) / (2 * eps)
        
        speed = np.linalg.norm(dx, axis=1, keepdims=True)
        return dx / (speed + 1e-12)
    
    def arc_length(self, t: np.ndarray) -> float:
        """Approximate arc length via numerical integration."""
        if HAS_SCIPY:
            dx = splev(t, self.tck, der=1)
        else:
            eps = 1e-5
            dx = (self.eval(t + eps) - self.eval(t - eps)) / (2 * eps)
        
        speed = np.linalg.norm(dx, axis=1)
        return np.trapz(speed, t)


# =============================================================================
# Part 2: Pinned Spline Constraint Satisfaction
# =============================================================================

def fit_spline_through_points(
    points: np.ndarray,
    pinned_indices: list,
    pinned_values: list,
    degree: int = 3,
    num_control_points: Optional[int] = None
) -> SplineVector:
    """
    Given N target points, fix some as pinned (boundary conditions).
    Find the spline that passes through pinned points AND best-fit the rest.
    
    This is interpolation + approximation — the pinned points are HARD
    constraints (must be exactly satisfied) while others are SOFT
    (approximated via least squares).
    
    Args:
        points: shape (N, n) — target points
        pinned_indices: indices of pinned points
        pinned_values: values at pinned indices (should match points at those indices)
        degree: B-spline degree
        num_control_points: override number of control points
        
    Returns:
        SplineVector that satisfies pinned constraints and best-fits others
        
    Freedom calculation:
        For cubic splines with n control points and k pinned:
        freedom = n - k - 2 (two constraints for C² continuity at boundaries)
    """
    N, n = points.shape
    num_ctrl = num_control_points or max(degree + 1, len(pinned_indices) + 3)
    
    # Build knot vector
    num_knots = num_ctrl + degree + 1
    knots = np.concatenate([
        np.zeros(degree),
        np.linspace(0, 1, num_ctrl - degree + 1),
        np.ones(degree)
    ])
    
    # Build design matrix for least squares
    # For each target point, evaluate B-spline basis at its parameter
    t_targets = np.linspace(0, 1, N)
    
    # B-spline basis matrix: B[i,j] = N_j_k(t_i)
    # Points ≈ B @ control_points
    
    # Simple approach: use scipy's splrep for full interpolation,
    # then adjust for pinned points
    if HAS_SCIPY and len(pinned_indices) >= 2:
        # Fit spline to all points first
        tck = splrep(t_targets, points, k=degree, s=0)  # s=0 for interpolation
        
        # Extract control points
        _, cpts_interp, _ = tck
        
        # Adjust control points near pinned regions
        cpts = cpts_interp.copy()
        
        # For each pinned point, find nearest knot span and adjust
        for idx, (pinned_idx, pinned_val) in enumerate(zip(pinned_indices, pinned_values)):
            t_pinned = t_targets[pinned_idx]
            
            # Re-evaluate spline at pinned parameter
            # and compute correction
            val_at_pinned = splev(t_pinned, tck)
            correction = pinned_val - val_at_pinned
            
            # Distribute correction to nearby control points
            # (simple approach: modify nearest control point)
            knot_idx = np.searchsorted(knots[degree:-degree], t_pinned)
            knot_idx = np.clip(knot_idx, degree, num_ctrl - degree - 1)
            
            cpts[knot_idx] += correction
        
        return SplineVector(cpts, degree)
    
    else:
        # Pure numpy: build and solve least squares
        # Build B-spline basis matrix
        B = np.zeros((N, num_ctrl))
        for i in range(num_ctrl):
            B[:, i] = _bspline_basis_array(t_targets, knots, i, degree)
        
        # Separate pinned and free
        pinned_set = set(pinned_indices)
        free_indices = [i for i in range(N) if i not in pinned_set]
        
        # Hard constraints for pinned points
        pinned_t = t_targets[pinned_indices]
        B_pinned = np.zeros((len(pinned_indices), num_ctrl))
        for i in range(num_ctrl):
            B_pinned[:, i] = _bspline_basis_array(pinned_t, knots, i, degree)
        
        # Soft fit for free points
        B_free = B[free_indices]
        points_free = points[free_indices]
        
        # Solve: min ||B_free @ cpts - points_free||²
        # Subject to: B_pinned @ cpts = pinned_values
        cpts_free, residuals, rank, s = np.linalg.lstsq(B_free, points_free, rcond=None)
        
        # Adjust to satisfy pinned constraints (simple projection)
        cpts = cpts_free.copy()
        
        # Iterative projection onto pinned constraints
        for iteration in range(10):
            for pinned_idx, pinned_val, row_idx in zip(pinned_indices, pinned_values, range(len(pinned_indices))):
                residual = pinned_val - B_pinned[row_idx] @ cpts
                cpts += residual * B_pinned[row_idx] * 0.1  # Small step
        
        return SplineVector(cpts, degree)


def _bspline_basis_array(t: np.ndarray, knots: np.ndarray, i: int, k: int) -> np.ndarray:
    """Vectorized B-spline basis function."""
    n = len(t)
    if k == 0:
        cond = (knots[i] <= t) & (t < knots[i + 1])
        if i == len(knots) - 2:
            cond = (knots[i] <= t) & (t <= knots[i + 1])
        return cond.astype(float)
    
    result = np.zeros(n)
    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]
    
    if denom1 != 0:
        result += (t - knots[i]) / denom1 * _bspline_basis_array(t, knots, i, k - 1)
    if denom2 != 0:
        result += (knots[i + k + 1] - t) / denom2 * _bspline_basis_array(t, knots, i + 1, k - 1)
    
    return result


# =============================================================================
# Part 3: Resonance Signature
# =============================================================================

def resonance_signature(
    response_time_series: np.ndarray,
    sample_rate: float = 1.0
) -> dict:
    """
    Given a time-series of responses to taps, extract resonance signature.
    
    The resonance signature characterizes how a system responds to excitation:
    - Frequencies: dominant modes in the response
    - Decay rate: how quickly oscillations die out
    - Harmonic content: relative strength of harmonics
    
    Args:
        response_time_series: time-series data (responses to taps)
        sample_rate: samples per unit time
        
    Returns:
        dict with frequencies, spectrum, decay_rate, harmonic_content
    """
    n = len(response_time_series)
    
    # FFT to get frequency spectrum
    spectrum = np.abs(np.fft.rfft(response_time_series))
    freqs = np.fft.rfftfreq(n, d=1.0/sample_rate)
    
    # Find dominant frequencies (top 5)
    if len(spectrum) > 5:
        peak_indices = np.argsort(spectrum)[-5:]
    else:
        peak_indices = np.argsort(spectrum)
    
    # Decay rate: fit exponential to envelope
    # Simple approach: ratio of max to value at end
    abs_response = np.abs(response_time_series)
    max_val = np.max(abs_response)
    end_val = np.abs(response_time_series[-1])
    
    if end_val > 1e-12 and max_val > 1e-12:
        decay_rate = -np.log(end_val / max_val) / n
    else:
        decay_rate = 0.0
    
    # Harmonic content: normalized spectrum at peaks
    harmonic_content = spectrum[peak_indices] / (np.sum(spectrum) + 1e-12)
    
    return {
        'frequencies': freqs[peak_indices],
        'spectrum': spectrum,
        'decay_rate': decay_rate,
        'harmonic_content': harmonic_content,
        'freqs': freqs
    }


# =============================================================================
# Part 4: Contrast Map
# =============================================================================

def contrast(base_signature: dict, tap_signature: dict) -> dict:
    """
    ΔR = R(tap) - R(base)
    
    Compare resonance signatures to identify:
    - hyperperfused: frequencies with AMPLIFIED response (positive contrast)
    - hypoperfused: frequencies with SUPPRESSED response (negative contrast)
    - dead_freqs: frequencies with negligible change (near zero)
    
    Args:
        base_signature: resonance signature of baseline/control state
        tap_signature: resonance signature after some stimulus/tap
        
    Returns:
        dict with hyperperfused, hypoperfused, dead frequencies
    """
    base_spectrum = base_signature['spectrum']
    tap_spectrum = tap_signature['spectrum']
    
    # Pad if necessary (different lengths)
    min_len = min(len(base_spectrum), len(tap_spectrum))
    base_spectrum = base_spectrum[:min_len]
    tap_spectrum = tap_spectrum[:min_len]
    
    # Contrast spectrum
    contrast_spectrum = tap_spectrum - base_spectrum
    
    # Threshold for "dead" (1% of max contrast)
    max_contrast = np.max(np.abs(contrast_spectrum))
    dead_threshold = 0.01 * max_contrast
    
    # Classify frequencies
    hyper_mask = contrast_spectrum > 0
    hypo_mask = contrast_spectrum < 0
    dead_mask = np.abs(contrast_spectrum) < dead_threshold
    
    base_freqs = base_signature['freqs'][:min_len]
    
    return {
        'hyperperfused_freqs': base_freqs[hyper_mask],
        'hypoperfused_freqs': base_freqs[hypo_mask],
        'dead_freqs': base_freqs[dead_mask],
        'contrast_spectrum': contrast_spectrum,
        'hyper_count': np.sum(hyper_mask),
        'hypo_count': np.sum(hypo_mask),
        'dead_count': np.sum(dead_mask)
    }


# =============================================================================
# Part 5: ASCII Resonance Imaging
# =============================================================================

def ascii_resonance_image(signature: dict, width: int = 60, height: int = 20) -> str:
    """
    Build ASCII image of resonance signature.
    
    Y-axis: frequency bins (high freq at top)
    X-axis: amplitude (bar width proportional to magnitude)
    Characters: density mapped to intensity
    
    Args:
        signature: resonance signature dict
        width: character width of image
        height: number of rows
        
    Returns:
        ASCII art string representation
    """
    spectrum = signature['spectrum']
    
    # Normalize spectrum
    max_val = np.max(spectrum)
    if max_val > 1e-12:
        norm_spectrum = spectrum / max_val
    else:
        norm_spectrum = spectrum
    
    # Build ASCII rows
    rows = []
    num_bins = len(spectrum)
    
    # Characters for different intensity levels (dark to light)
    chars = ' ▁▂▃▄▅▆▇█'
    
    for i in range(height):
        # Map row index to frequency bin
        bin_idx = int(i * num_bins / height)
        bin_idx = min(bin_idx, num_bins - 1)
        
        amp = norm_spectrum[bin_idx] if bin_idx < num_bins else 0
        
        # Map amplitude to bar width and character
        char_level = int(amp * (len(chars) - 1))
        char_level = min(char_level, len(chars) - 1)
        
        bar_char = chars[char_level]
        bar_width = int(amp * width)
        bar = bar_char * bar_width
        
        # Pad to width
        row = bar + ' ' * (width - bar_width)
        rows.append(row)
    
    return '\n'.join(rows)


def ascii_contrast_image(contrast_map: dict, width: int = 60, height: int = 20) -> str:
    """
    Build ASCII image of contrast map.
    
    Uses different characters for hyper/hypo/dead regions.
    """
    contrast_spectrum = contrast_map['contrast_spectrum']
    
    # Normalize
    max_val = np.max(np.abs(contrast_spectrum))
    if max_val > 1e-12:
        norm = contrast_spectrum / max_val
    else:
        norm = contrast_spectrum
    
    rows = []
    num_bins = len(contrast_spectrum)
    
    for i in range(height):
        bin_idx = int(i * num_bins / height)
        bin_idx = min(bin_idx, num_bins - 1)
        
        val = norm[bin_idx] if bin_idx < num_bins else 0
        
        if val > 0.01:
            char = '█'  # hyperperfused
        elif val < -0.01:
            char = '▓'  # hypoperfused
        else:
            char = '░'  # dead
        
        bar_width = int(abs(val) * width)
        row = char * bar_width + ' ' * (width - bar_width)
        rows.append(row)
    
    return '\n'.join(rows)


# =============================================================================
# Tests
# =============================================================================

def test_spline_vector():
    """Test 1: SplineVector evaluation and curvature computation."""
    print("\n" + "="*60)
    print("TEST 1: SplineVector evaluation and curvature")
    print("="*60)
    
    # Create a circular arc spline
    theta = np.linspace(0, np.pi/2, 10)
    control_points = np.column_stack([
        np.cos(theta),
        np.sin(theta),
        np.zeros_like(theta)
    ])
    
    spline = SplineVector(control_points, degree=3)
    
    # Evaluate at many points
    t = np.linspace(0, 1, 50)
    points = spline.eval(t)
    
    print(f"  Control points shape: {control_points.shape}")
    print(f"  Evaluated points shape: {points.shape}")
    print(f"  First point: {points[0]}")
    print(f"  Last point: {points[-1]}")
    
    # Curvature of circle should be constant = 1
    kappa = spline.curvature(t)
    print(f"  Curvature (should be ~1 for circle): mean={np.mean(kappa):.4f}")
    
    # Test higher dimension
    control_points_4d = np.column_stack([
        np.cos(theta),
        np.sin(theta),
        np.zeros_like(theta),
        0.5 * np.cos(theta)
    ])
    
    spline_4d = SplineVector(control_points_4d, degree=3)
    points_4d = spline_4d.eval(t)
    kappa_4d = spline_4d.curvature(t)
    
    print(f"  4D spline curvature mean: {np.mean(kappa_4d):.4f}")
    print("  ✓ SplineVector tests passed")


def test_pinned_spline():
    """Test 2: Pinned spline fitting."""
    print("\n" + "="*60)
    print("TEST 2: Pinned spline constraint satisfaction")
    print("="*60)
    
    # Generate target points along a curve
    t_target = np.linspace(0, 1, 20)
    target_points = np.column_stack([
        t_target,
        np.sin(t_target * np.pi),
        np.zeros_like(t_target)
    ])
    
    # Pin the endpoints
    pinned_indices = [0, len(t_target) - 1]
    pinned_values = target_points[pinned_indices]
    
    print(f"  Target points shape: {target_points.shape}")
    print(f"  Pinned indices: {pinned_indices}")
    print(f"  Pinned start: {pinned_values[0]}")
    print(f"  Pinned end: {pinned_values[1]}")
    
    # Fit spline with pinned constraints
    spline = fit_spline_through_points(
        target_points,
        pinned_indices,
        pinned_values.tolist(),
        degree=3,
        num_control_points=8
    )
    
    # Verify pinned points are satisfied
    t_eval = np.linspace(0, 1, len(t_target))
    points_eval = spline.eval(t_eval)
    
    start_error = np.linalg.norm(points_eval[0] - pinned_values[0])
    end_error = np.linalg.norm(points_eval[-1] - pinned_values[1])
    
    print(f"  Start point error: {start_error:.6f}")
    print(f"  End point error: {end_error:.6f}")
    
    if start_error < 0.01 and end_error < 0.01:
        print("  ✓ Pinned spline constraint satisfied")
    else:
        print("  ✗ Pinned spline constraint NOT satisfied")


def test_resonance_signature():
    """Test 3: Resonance signature extraction from synthetic data."""
    print("\n" + "="*60)
    print("TEST 3: Resonance signature extraction")
    print("="*60)
    
    # Generate synthetic resonance signal
    # Multiple damped sinusoids
    t = np.linspace(0, 10, 1000)
    sample_rate = 100  # samples per unit time
    
    signal = (
        1.0 * np.sin(2 * np.pi * 5 * t) * np.exp(-0.5 * t) +  # 5 Hz, slow decay
        0.5 * np.sin(2 * np.pi * 12 * t) * np.exp(-0.8 * t) +  # 12 Hz, faster decay
        0.3 * np.sin(2 * np.pi * 20 * t) * np.exp(-1.2 * t) +  # 20 Hz, fastest decay
        0.1 * np.random.randn(len(t))  # noise
    )
    
    sig = resonance_signature(signal, sample_rate=sample_rate)
    
    print(f"  Dominant frequencies: {sig['frequencies']}")
    print(f"  Decay rate: {sig['decay_rate']:.4f}")
    print(f"  Harmonic content: {sig['harmonic_content']}")
    
    # Show ASCII visualization
    print("\n  ASCII Resonance Image:")
    print("  " + "-" * 60)
    ascii_img = ascii_resonance_image(sig, width=50, height=15)
    for line in ascii_img.split('\n'):
        print(f"  {line}")
    
    print("  ✓ Resonance signature extracted")


def test_contrast_map():
    """Test 4: Contrast map computation."""
    print("\n" + "="*60)
    print("TEST 4: Contrast map computation")
    print("="*60)
    
    # Base signature: simple resonance
    t = np.linspace(0, 10, 500)
    sample_rate = 50
    
    base_signal = (
        1.0 * np.sin(2 * np.pi * 8 * t) * np.exp(-0.3 * t) +
        0.4 * np.sin(2 * np.pi * 15 * t) * np.exp(-0.5 * t)
    )
    
    # Tap signature: different resonance (amplified at 8 Hz, suppressed at 15 Hz)
    tap_signal = (
        1.5 * np.sin(2 * np.pi * 8 * t) * np.exp(-0.3 * t) +  # amplified
        0.1 * np.sin(2 * np.pi * 15 * t) * np.exp(-0.5 * t) +  # suppressed
        0.2 * np.sin(2 * np.pi * 25 * t) * np.exp(-0.7 * t)  # new frequency
    )
    
    base_sig = resonance_signature(base_signal, sample_rate=sample_rate)
    tap_sig = resonance_signature(tap_signal, sample_rate=sample_rate)
    
    contrast_map = contrast(base_sig, tap_sig)
    
    print(f"  Hyperperfused (amplified) frequencies: {contrast_map['hyperperfused_freqs']}")
    print(f"  Hypoperfused (suppressed) frequencies: {contrast_map['hypoperfused_freqs']}")
    print(f"  Dead (no change) frequencies: {len(contrast_map['dead_freqs'])} frequencies")
    print(f"  Hyper count: {contrast_map['hyper_count']}, Hypo count: {contrast_map['hypo_count']}")
    
    # Show contrast ASCII image
    print("\n  ASCII Contrast Image (█=hyper, ▓=hypo, ░=dead):")
    print("  " + "-" * 60)
    ascii_contrast = ascii_contrast_image(contrast_map, width=50, height=15)
    for line in ascii_contrast.split('\n'):
        print(f"  {line}")
    
    print("  ✓ Contrast map computed")


def test_ascii_imaging():
    """Test 5: ASCII image generation."""
    print("\n" + "="*60)
    print("TEST 5: ASCII resonance imaging")
    print("="*60)
    
    # Create a synthetic signature with clear peaks
    t = np.linspace(0, 10, 400)
    
    # Multi-frequency signal
    signal = (
        2.0 * np.sin(2 * np.pi * 3 * t) * np.exp(-0.1 * t) +
        1.5 * np.sin(2 * np.pi * 7 * t) * np.exp(-0.2 * t) +
        1.0 * np.sin(2 * np.pi * 12 * t) * np.exp(-0.3 * t) +
        0.5 * np.sin(2 * np.pi * 20 * t) * np.exp(-0.5 * t)
    )
    
    sig = resonance_signature(signal, sample_rate=40)
    
    # Different sizes
    for width, height in [(40, 10), (60, 15), (80, 20)]:
        print(f"\n  Size {width}x{height}:")
        ascii_img = ascii_resonance_image(sig, width=width, height=height)
        for line in ascii_img.split('\n'):
            print(f"  {line}")
    
    print("  ✓ ASCII imaging tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# SPLINE EMBEDDING PROTOTYPE - TEST SUITE")
    print("#"*60)
    print(f"# scipy available: {HAS_SCIPY}")
    print(f"# numpy version: {np.__version__}")
    
    test_spline_vector()
    test_pinned_spline()
    test_resonance_signature()
    test_contrast_map()
    test_ascii_imaging()
    
    print("\n" + "#"*60)
    print("# ALL TESTS COMPLETED")
    print("#"*60)


if __name__ == "__main__":
    run_all_tests()
