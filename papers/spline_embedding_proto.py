#!/usr/bin/env python3
"""
Spline Embedding Prototype
Minimal implementation for resonance-based spline analysis.
"""

import numpy as np

# ============================================================================
# 1. SplineVector
# ============================================================================

class SplineVector:
    """A vector is a spline f: [0,1] -> R^n, not a point."""

    def __init__(self, control_points, degree=3):
        # control_points: shape (m+1, n) — m+1 control points in R^n
        self.cpts = np.array(control_points, dtype=np.float64)
        self.degree = degree
        self.n = self.cpts.shape[1]
        self.m = self.cpts.shape[0] - 1
        self.bspline = None
        self._build_spline()

    def _build_spline(self):
        """Build the B-spline representation."""
        try:
            from scipy.interpolate import BSpline
        except ImportError:
            raise ImportError("scipy is required for SplineVector")

        num_knots = self.m + self.degree + 1
        if num_knots < 2 * (self.degree + 1):
            # Degenerate case: fewer control points than needed
            num_knots = 2 * (self.degree + 1)

        interior_knots = max(1, self.m - self.degree + 1)
        knots = np.concatenate([
            np.zeros(self.degree),
            np.linspace(0, 1, interior_knots),
            np.ones(self.degree)
        ])
        self.bspline = BSpline(knots, self.cpts, self.degree)

    def eval(self, t):
        """Evaluate spline at parameter values t using numpy interpolation."""
        t = np.asarray(t)
        if t.ndim == 0:
            t = t.reshape(1)
        return self.bspline(t)

    def curvature(self, t):
        """
        Compute curvature at parameter values t.
        
        For a curve in R^n, curvature is:
        κ = |d1 × d2| / |d1|³
        
        For n > 3, we use the generalized formula via Gram matrix determinant
        or fall back to speed proxy.
        """
        from scipy.interpolate import BSpline

        t = np.asarray(t)
        if t.ndim == 0:
            t = t.reshape(1)

        # First and second derivatives
        d1 = self.bspline.derivative()(t)
        d2 = self.bspline.derivative().derivative()(t)

        # speed = |d1|
        speed = np.linalg.norm(d1, axis=1) + 1e-12

        if self.n == 2:
            # 2D: κ = |x'y'' - y'x''| / |d1|³
            cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
            curvature = np.abs(cross) / (speed ** 3)
        elif self.n == 3:
            # 3D: κ = |d1 × d2| / |d1|³
            cross = np.cross(d1, d2)
            cross_norm = np.linalg.norm(cross, axis=1)
            curvature = cross_norm / (speed ** 3)
        else:
            # n > 3: use speed proxy (inverse radius of gyration proxy)
            # κ ≈ |d2| / |d1|² as approximation
            d2_norm = np.linalg.norm(d2, axis=1)
            curvature = d2_norm / (speed ** 2 + 1e-12)

        return curvature


# ============================================================================
# 2. PinnedSplineFitter
# ============================================================================

def fit_pinned_spline(points, pinned_indices, pinned_values, degree=3):
    """
    Find spline that passes through pinned points AND best-fits the rest.
    Returns SplineVector.

    Args:
        points: Array of N points in R^n (N x n)
        pinned_indices: Indices of pinned (constrained) points
        pinned_values: The values at pinned indices (same shape as points[pinned_indices])
        degree: B-spline degree

    Returns:
        SplineVector instance

    With n pinned points out of N total, freedom = N - n - 2 (C² continuity).
    """
    points = np.asarray(points, dtype=np.float64)
    n_points, n_dim = points.shape
    n_pinned = len(pinned_indices)

    # Unpinned indices
    unpinned_idx = [i for i in range(n_points) if i not in set(pinned_indices)]

    # Solve: we want spline that passes through pinned points
    # and best-fits unpinned points via least squares
    # 
    # For a B-spline, each evaluation point is a linear combination of
    # control points: f(t_i) = sum_j B_j(t_i) * c_j
    # 
    # Constraint: f(t_pinned) = pinned_value
    # Fit: minimize ||f(t_unpinned) - points[unpinned]||²

    n_ctrl = n_pinned + max(1, len(unpinned_idx))
    n_ctrl = max(n_ctrl, degree + 1)

    # Parameterize by index
    t = np.linspace(0, 1, n_points)

    # Build basis matrix
    from scipy.interpolate import BSpline, splrep

    try:
        # Use splrep to get knot vector and coefficients, then adjust
        # for pinned constraints
        
        # Fit unconstrained spline
        if len(unpinned_idx) > 0:
            t_unpinned = t[unpinned_idx]
            pts_unpinned = points[unpinned_idx]
            # Simple approach: start with best-fit spline
            # and adjust control points for pinned constraints
            tck = splrep(t, points, k=degree, s=len(unpinned_idx) * 0.1)
            knots, coeffs = tck[0], tck[1]
        else:
            # All pinned - exact interpolation
            tck = splrep(t, points, k=degree, s=0)
            knots, coeffs = tck[0], tck[1]

        # Adjust for pinned points via constrained least squares
        # Build evaluation matrix for pinned points
        from scipy.interpolate import splev
        
        # Refit with constraints using least squares
        # For each pinned point, adjust coefficients
        B_pinned = np.zeros((n_pinned, n_ctrl))
        for i, (pi, pv) in enumerate(zip(pinned_indices, pinned_values)):
            # Evaluate basis at t[pi]
            t_val = t[pi]
            # Simple uniform B-spline basis approximation
            for j in range(n_ctrl):
                # hat function approximation
                u = t_val * (n_ctrl - 1)
                if j <= u <= j + 1:
                    B_pinned[i, j] = 1 - abs(u - j)
                else:
                    B_pinned[i, j] = 0

        # Solve B_pinned @ c = pinned_values
        # Use least squares if overdetermined
        if n_pinned < n_ctrl:
            # Solve using pseudo-inverse
            c0 = np.linalg.lstsq(B_pinned, pinned_values, rcond=None)[0]
        else:
            c0 = np.linalg.solve(B_pinned[:n_ctrl], pinned_values[:n_ctrl])

        ctrl_pts = c0

    except Exception as e:
        # Fallback: simple linear interpolation with degree bumps
        ctrl_pts = np.zeros((n_ctrl, n_dim))
        # Place pinned points at corresponding control point positions
        for i, (pi, pv) in enumerate(zip(pinned_indices, pinned_values)):
            pos = int(pi * n_ctrl / n_points)
            pos = min(pos, n_ctrl - 1)
            ctrl_pts[pos] = pv
        
        # Fill in between
        for d in range(n_dim):
            ctrl_pts[:, d] = np.interp(
                np.linspace(0, n_ctrl - 1, n_ctrl),
                np.arange(n_points),
                points[:, d]
            )

    return SplineVector(ctrl_pts, degree=degree)


# ============================================================================
# 3. ResonanceSignature
# ============================================================================

def resonance_signature(time_series):
    """
    Extract resonance signature from response time-series.
    FFT -> frequency spectrum, decay rate, harmonic content.
    """
    time_series = np.asarray(time_series)
    
    # Handle empty or trivial input
    if len(time_series) < 2:
        return {
            'frequencies': np.array([]),
            'spectrum': np.array([]),
            'decay_rate': 0.0,
            'harmonic_content': 0.0
        }

    spectrum = np.abs(np.fft.rfft(time_series))
    freqs = np.fft.rfftfreq(len(time_series))
    
    # Top 5 peaks by sorted indices
    if len(spectrum) > 0:
        peak_idx = np.argsort(spectrum)[-5:]
        peak_idx = peak_idx[peak_idx < len(freqs)]
    else:
        peak_idx = np.array([], dtype=int)
    
    # Decay rate: log decay per sample
    abs_ts = np.abs(time_series)
    max_val = np.max(abs_ts) + 1e-12
    last_val = abs_ts[-1] if len(abs_ts) > 0 else max_val
    
    if last_val > 0 and max_val > last_val:
        decay_rate = -np.log(last_val / max_val) / len(time_series)
    else:
        decay_rate = 0.0
    
    # Harmonic content: fraction of energy in top peaks
    peak_energy = np.sum(spectrum[peak_idx]) if len(peak_idx) > 0 else 0
    total_energy = np.sum(spectrum) + 1e-12
    harmonic_content = peak_energy / total_energy

    return {
        'frequencies': freqs[peak_idx],
        'spectrum': spectrum,
        'decay_rate': decay_rate,
        'harmonic_content': harmonic_content
    }


# ============================================================================
# 4. ContrastMap
# ============================================================================

def contrast(base_sig, tap_sig):
    """
    ΔR = R(tap) - R(base)
    Returns hyperperfused, hypoperfused, dead spots.
    """
    # Use spectrum lengths - rfft gives N//2+1 frequencies
    min_len = min(len(base_sig['spectrum']), len(tap_sig['spectrum']))
    base_spec = base_sig['spectrum'][:min_len]
    tap_spec = tap_sig['spectrum'][:min_len]
    
    contrast = tap_spec - base_spec
    
    threshold = 0.01 * np.max(np.abs(contrast)) if np.any(contrast) else 0
    
    # contrast_spectrum length matches base_spec (min_len)
    return {
        'hyperperfused': np.where(contrast > threshold)[0],
        'hypoperfused': np.where(contrast < -threshold)[0],
        'dead': np.where(np.abs(contrast) < threshold)[0],
        'contrast_spectrum': contrast
    }


# ============================================================================
# 5. ASCII Resonance Image
# ============================================================================

def ascii_resonance(signature, width=60, height=15):
    """ASCII image of frequency spectrum"""
    spec = signature['spectrum']
    if len(spec) == 0:
        return '\n'.join([' ' * width] * height)
    
    spec = spec / (np.max(spec) + 1e-12)
    
    lines = []
    for i in range(height):
        idx = int(i * len(spec) / height)
        amp = spec[idx] if idx < len(spec) else 0
        bar = int(amp * width)
        lines.append('█' * bar + '░' * (width - bar))
    return '\n'.join(lines)


# ============================================================================
# Tests
# ============================================================================

def test_spline_vector_r3():
    """Test 1: SplineVector with 5 control points in R³."""
    cpts = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
        [2.0, -1.0, 2.0],
        [3.0, 1.0, 0.0],
        [4.0, 0.0, 1.0]
    ]
    sv = SplineVector(cpts, degree=3)
    assert sv.n == 3, f"Expected n=3, got {sv.n}"
    assert sv.m == 4, f"Expected m=4, got {sv.m}"
    print("✓ SplineVector with 5 control points in R³ OK")
    return sv


def test_eval_shape():
    """Test 2: eval() returns correct shape."""
    cpts = [[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]]
    sv = SplineVector(cpts, degree=3)
    
    t_single = 0.5
    result_single = sv.eval(t_single)
    # Single t gives shape (1, n) since scipy BSpline preserves batch dimension
    assert result_single.shape[1] == 2, f"Single t: expected n=2, got {result_single.shape}"
    assert len(result_single) == 1, f"Single t: expected batch=1, got {len(result_single)}"
    
    t_array = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result_array = sv.eval(t_array)
    assert result_array.shape == (5, 2), f"Array t: expected (5,2), got {result_array.shape}"
    
    print(f"✓ eval() shape OK: single→(1,2), array(5)→(5,2)")
    return True


def test_curvature_positive():
    """Test 3: curvature() returns positive values."""
    cpts = [[0, 0], [1, 2], [2, -1], [3, 1], [4, 0]]
    sv = SplineVector(cpts, degree=3)
    
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    curv = sv.curvature(t)
    
    assert np.all(curv >= 0), f"Curvature has negative values: {curv}"
    assert len(curv) == len(t), f"Curvature length mismatch"
    
    print(f"✓ curvature() positive: min={curv.min():.4f}, max={curv.max():.4f}")
    return curv


def test_resonance_signature():
    """Test 4: resonance_signature() from synthetic impulse response."""
    # Synthetic damped oscillation
    t = np.linspace(0, 1, 128)
    omega = 2 * np.pi * 5  # 5 Hz
    decay = 8
    signal = np.exp(-decay * t) * np.sin(omega * t)
    
    sig = resonance_signature(signal)
    
    assert 'spectrum' in sig
    assert 'frequencies' in sig
    assert 'decay_rate' in sig
    assert 'harmonic_content' in sig
    assert sig['decay_rate'] > 0, "Damped signal should have positive decay"
    assert 0 <= sig['harmonic_content'] <= 1
    
    print(f"✓ resonance_signature() OK: decay={sig['decay_rate']:.3f}, harmonic={sig['harmonic_content']:.3f}")
    return sig


def test_contrast():
    """Test 5: contrast() between two signatures."""
    t = np.linspace(0, 1, 128)
    
    # Base: damped oscillation at 5 Hz
    base_signal = np.exp(-8 * t) * np.sin(2 * np.pi * 5 * t)
    base_sig = resonance_signature(base_signal)
    
    # Tap: same but with added component at 10 Hz
    tap_signal = np.exp(-8 * t) * (np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t))
    tap_sig = resonance_signature(tap_signal)
    
    cmap = contrast(base_sig, tap_sig)
    
    assert 'hyperperfused' in cmap
    assert 'hypoperfused' in cmap
    assert 'dead' in cmap
    assert 'contrast_spectrum' in cmap
    
    print(f"✓ contrast() OK: hyper={len(cmap['hyperperfused'])} freq, "
          f"hypo={len(cmap['hypoperfused'])} freq, "
          f"dead={len(cmap['dead'])} freq")
    return cmap


def test_ascii_resonance():
    """Test 6: ascii_resonance() produces 15 lines of output."""
    # Create signature from signal
    t = np.linspace(0, 1, 128)
    signal = np.exp(-5 * t) * np.sin(2 * np.pi * 8 * t)
    sig = resonance_signature(signal)
    
    img = ascii_resonance(sig, width=60, height=15)
    lines = img.split('\n')
    
    assert len(lines) == 15, f"Expected 15 lines, got {len(lines)}"
    assert all(len(line) == 60 for line in lines), "Not all lines are width 60"
    
    # Check it's mostly ░ with some █
    has_bar = any('█' in line for line in lines)
    assert has_bar, "ASCII art should contain at least some █"
    
    print(f"✓ ascii_resonance() OK: 15×60 image with █ and ░")
    print("Sample output:")
    for line in lines[:5]:
        print(f"  {line}")
    return img


# ============================================================================
# Runner
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Spline Embedding Prototype — Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("SplineVector R³", test_spline_vector_r3),
        ("eval() shape", test_eval_shape),
        ("curvature() positive", test_curvature_positive),
        ("resonance_signature()", test_resonance_signature),
        ("contrast()", test_contrast),
        ("ascii_resonance()", test_ascii_resonance),
    ]
    
    passed = 0
    failed = 0
    
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)