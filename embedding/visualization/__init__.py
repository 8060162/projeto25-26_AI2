"""
Visualization helpers for embedding inspection workflows.
"""

from typing import TYPE_CHECKING, Any

from embedding.visualization.spotlight_export import (
    SpotlightExportResult,
    export_spotlight_dataset,
)

if TYPE_CHECKING:
    from embedding.visualization.benchmark_overlay_export import (
        BenchmarkOverlayExportResult,
    )
    from embedding.visualization.spotlight_viewer import SpotlightLaunchResult

__all__ = [
    "BenchmarkOverlayExportResult",
    "SpotlightExportResult",
    "export_benchmark_overlay_dataset",
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

    if name in {
        "BenchmarkOverlayExportResult",
        "export_benchmark_overlay_dataset",
    }:
        from embedding.visualization.benchmark_overlay_export import (
            BenchmarkOverlayExportResult,
            export_benchmark_overlay_dataset,
        )

        exported_members = {
            "BenchmarkOverlayExportResult": BenchmarkOverlayExportResult,
            "export_benchmark_overlay_dataset": export_benchmark_overlay_dataset,
        }
        return exported_members[name]

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
