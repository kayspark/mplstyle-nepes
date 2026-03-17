"""Nepes corporate color palette for matplotlib.

Usage:
    import nepes_mplstyle
    nepes_mplstyle.use("light")   # or "dark"

    # Or directly:
    import matplotlib.pyplot as plt
    plt.style.use(nepes_mplstyle.style_path("light"))
"""

from importlib.resources import files


def style_path(theme: str = "light") -> str:
    """Return the absolute path to a nepes .mplstyle file."""
    name = f"nepes-{theme}.mplstyle"
    return str(files("nepes_mplstyle").joinpath(name))


def use(theme: str = "light") -> None:
    """Apply the nepes matplotlib style."""
    import matplotlib.pyplot as plt
    plt.style.use(style_path(theme))
