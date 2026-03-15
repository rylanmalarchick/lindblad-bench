"""
lindblad_bench — Python bindings for the bare-metal C Lindblad library.

Provides a zero-copy NumPy interface to lb_propagate_step and lb_evolve.
Build with: pip install -e python/

Requires the C library to be built first:
    cmake -B build && cmake --build build
"""

# Attempt to import the C extension; fall back to a clear error.
try:
    from . import _lindblad_ext as _ext  # noqa: F401
    _C_AVAILABLE = True
except ImportError:
    _C_AVAILABLE = False

__version__ = "0.1.0-dev"
__all__ = ["propagate_step", "evolve", "build_lindbladian"]


def _require_c() -> None:
    if not _C_AVAILABLE:
        raise ImportError(
            "C extension not built. Run: cmake -B build && cmake --build build\n"
            "Then: pip install -e python/"
        )


def propagate_step(P: "np.ndarray", rho_in: "np.ndarray") -> "np.ndarray":
    """
    Apply one Lindblad propagation step.

    Args:
        P:      Propagator matrix (d²×d²), complex128, C-contiguous.
        rho_in: Density matrix (d×d), complex128.

    Returns:
        rho_out: Density matrix after one step (d×d), complex128.
    """
    _require_c()
    return _ext.propagate_step(P, rho_in)


def evolve(H: "np.ndarray", cops: "list[np.ndarray]",
           rho0: "np.ndarray", t_end: float, dt: float) -> "np.ndarray":
    """
    Evolve a density matrix from t=0 to t=t_end.

    Args:
        H:     Hamiltonian (d×d), complex128.
        cops:  List of collapse operators (d×d each).
        rho0:  Initial density matrix (d×d).
        t_end: End time (same units as dt).
        dt:    Timestep.

    Returns:
        rho_final: Final density matrix (d×d).
    """
    _require_c()
    return _ext.evolve(H, cops, rho0, t_end, dt)
