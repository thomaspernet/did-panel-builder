"""Shared visualization style and color palette."""

from __future__ import annotations

COLORS = {
    "treated": "#E07A5F",
    "control": "#3D405B",
    "pre": "#81B29A",
    "post": "#F2CC8F",
    "highlight": "#E63946",
    "neutral": "#A8DADC",
    "success": "#2ecc71",
    "danger": "#e74c3c",
    "info": "#3498db",
    "muted": "#9b59b6",
}

# CI z-values — avoids scipy dependency
Z_VALUES = {
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


def get_z(ci: float) -> float:
    """Get z-value for a confidence level."""
    if ci in Z_VALUES:
        return Z_VALUES[ci]
    raise ValueError(f"Unsupported CI level: {ci}. Use one of {list(Z_VALUES.keys())}")


def apply_style() -> None:
    """Apply the did-panel-builder default matplotlib style."""
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        })
    except ImportError:
        pass
