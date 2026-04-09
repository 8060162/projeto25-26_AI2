"""
Visualization helpers for embedding inspection workflows.
"""

from typing import TYPE_CHECKING, Any

from embedding.visualization.spotlight_export import (
    SpotlightExportResult,
    export_spotlight_dataset,
)

if TYPE_CHECKING:
    from embedding.visualization.spotlight_viewer import SpotlightLaunchResult

__all__ = [
    "SpotlightExportResult",
    "SpotlightLaunchResult",
    "export_spotlight_dataset",
    "launch_spotlight_viewer",
]


def __getattr__(name: str) -> Any:
    """
    Lazily expose viewer helpers without importing the module at package import time.

    Parameters
    ----------
    name : str
        Attribute requested from the package namespace.

    Returns
    -------
    Any
        Lazily imported viewer helper.
    """

    if name in {"SpotlightLaunchResult", "launch_spotlight_viewer"}:
        from embedding.visualization.spotlight_viewer import (
            SpotlightLaunchResult,
            launch_spotlight_viewer,
        )

        exported_members = {
            "SpotlightLaunchResult": SpotlightLaunchResult,
            "launch_spotlight_viewer": launch_spotlight_viewer,
        }
        return exported_members[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
