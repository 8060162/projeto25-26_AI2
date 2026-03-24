from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Sequence

from Chunking.config.settings import PipelineSettings


@dataclass(slots=True)
class SpotlightLaunchResult:
    """
    Describe the dataset loaded into the Spotlight viewer.

    Attributes
    ----------
    dataset_path : Path
        JSONL dataset opened in Spotlight.

    record_count : int
        Number of rows available in the dataset.
    """

    dataset_path: Path
    record_count: int


def launch_spotlight_viewer(
    dataset_path: Optional[Path | str] = None,
    settings: Optional[PipelineSettings] = None,
    strategy_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> SpotlightLaunchResult:
    """
    Resolve one exported Spotlight dataset and open it in Spotlight.

    Parameters
    ----------
    dataset_path : Optional[Path | str]
        Explicit dataset path to open. When omitted, the latest exported
        dataset is resolved from the embedding output folder.

    settings : Optional[PipelineSettings]
        Shared runtime settings. When omitted, default settings are loaded.

    strategy_name : Optional[str]
        Optional strategy folder used when resolving a dataset automatically.

    run_id : Optional[str]
        Optional run folder used when resolving a dataset automatically.

    Returns
    -------
    SpotlightLaunchResult
        Summary of the opened dataset.
    """

    resolved_settings = settings or PipelineSettings()
    resolved_dataset_path = _resolve_dataset_path(
        dataset_path=dataset_path,
        output_root=resolved_settings.embedding_output_root,
        strategy_name=strategy_name,
        run_id=run_id,
    )
    record_count = _count_dataset_rows(resolved_dataset_path)
    spotlight_module = _load_spotlight_module()
    _open_dataset_in_spotlight(spotlight_module, resolved_dataset_path)

    return SpotlightLaunchResult(
        dataset_path=resolved_dataset_path,
        record_count=record_count,
    )


def _resolve_dataset_path(
    dataset_path: Optional[Path | str],
    output_root: Path,
    strategy_name: Optional[str],
    run_id: Optional[str],
) -> Path:
    """
    Resolve the Spotlight dataset path from explicit input or output folders.

    Parameters
    ----------
    dataset_path : Optional[Path | str]
        Explicit dataset path requested by the caller.

    output_root : Path
        Root folder where embedding runs are stored.

    strategy_name : Optional[str]
        Optional strategy filter applied during automatic resolution.

    run_id : Optional[str]
        Optional run identifier filter applied during automatic resolution.

    Returns
    -------
    Path
        Existing dataset path ready to be opened in Spotlight.
    """

    if dataset_path is not None:
        return _validate_dataset_path(Path(dataset_path).expanduser().resolve())

    return _resolve_latest_dataset_path(
        output_root=output_root,
        strategy_name=strategy_name,
        run_id=run_id,
    )


def _resolve_latest_dataset_path(
    output_root: Path,
    strategy_name: Optional[str],
    run_id: Optional[str],
) -> Path:
    """
    Resolve the newest exported Spotlight dataset under the embedding output root.

    Parameters
    ----------
    output_root : Path
        Root folder where embedding runs are stored.

    strategy_name : Optional[str]
        Optional strategy folder filter.

    run_id : Optional[str]
        Optional run folder filter.

    Returns
    -------
    Path
        Newest matching Spotlight dataset.
    """

    if not output_root.exists():
        raise FileNotFoundError(
            "Embedding output root does not exist. Run the embedding pipeline "
            "with Spotlight export enabled before launching the viewer."
        )

    if run_id and not strategy_name:
        raise ValueError("A strategy name is required when a run id is provided.")

    search_root = output_root
    if strategy_name:
        normalized_strategy_name = strategy_name.strip()
        if not normalized_strategy_name:
            raise ValueError("Strategy name cannot be empty.")
        search_root = search_root / normalized_strategy_name

    if run_id:
        normalized_run_id = run_id.strip()
        if not normalized_run_id:
            raise ValueError("Run id cannot be empty.")
        search_root = search_root / normalized_run_id

    if not search_root.exists():
        raise FileNotFoundError(
            f"No Spotlight exports were found under '{search_root}'."
        )

    candidate_paths = sorted(
        search_root.rglob("spotlight_dataset.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidate_paths:
        raise FileNotFoundError(
            f"No Spotlight dataset was found under '{search_root}'."
        )

    return _validate_dataset_path(candidate_paths[0].resolve())


def _validate_dataset_path(dataset_path: Path) -> Path:
    """
    Validate that one dataset path points to a readable Spotlight export file.

    Parameters
    ----------
    dataset_path : Path
        Candidate dataset path to validate.

    Returns
    -------
    Path
        Normalized validated dataset path.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Spotlight dataset '{dataset_path}' does not exist."
        )

    if not dataset_path.is_file():
        raise ValueError(f"Spotlight dataset '{dataset_path}' is not a file.")

    if dataset_path.name != "spotlight_dataset.jsonl":
        raise ValueError(
            "Spotlight viewer expects a 'spotlight_dataset.jsonl' export file."
        )

    return dataset_path


def _count_dataset_rows(dataset_path: Path) -> int:
    """
    Count valid JSONL rows in one Spotlight dataset export.

    Parameters
    ----------
    dataset_path : Path
        Dataset path to inspect.

    Returns
    -------
    int
        Number of non-empty JSON rows available for Spotlight.
    """

    record_count = 0

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, row in enumerate(dataset_file, start=1):
            normalized_row = row.strip()
            if not normalized_row:
                continue

            try:
                json.loads(normalized_row)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Spotlight dataset '{dataset_path}' contains invalid JSON "
                    f"on line {line_number}."
                ) from exc

            record_count += 1

    if record_count == 0:
        raise ValueError(f"Spotlight dataset '{dataset_path}' is empty.")

    return record_count


def _load_spotlight_module() -> ModuleType:
    """
    Import the installed Spotlight module using known package names.

    Returns
    -------
    ModuleType
        Imported module that exposes the Spotlight viewer API.
    """

    candidate_module_names = ("renumics.spotlight", "spotlight")

    for module_name in candidate_module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

    raise RuntimeError(
        "Renumics Spotlight is not installed. Install the Spotlight package "
        "before launching the embedding viewer."
    )


def _open_dataset_in_spotlight(
    spotlight_module: ModuleType,
    dataset_path: Path,
) -> None:
    """
    Open one exported dataset through the installed Spotlight API.

    Parameters
    ----------
    spotlight_module : ModuleType
        Imported Spotlight module.

    dataset_path : Path
        Dataset path to open.
    """

    show_function = getattr(spotlight_module, "show", None)
    if not callable(show_function):
        raise RuntimeError(
            "The installed Spotlight module does not expose a callable 'show' API."
        )

    dataset_argument = str(dataset_path)
    call_patterns = (
        ((dataset_argument,), {}),
        ((), {"data": dataset_argument}),
        ((), {"dataset": dataset_argument}),
    )

    last_error: Optional[Exception] = None

    for args, kwargs in call_patterns:
        try:
            show_function(*args, **kwargs)
            return
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            raise RuntimeError(
                f"Spotlight failed to open dataset '{dataset_path}'."
            ) from exc

    raise RuntimeError(
        "Unable to launch Spotlight because the installed 'show' API does not "
        "match the supported call patterns."
    ) from last_error


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for the Spotlight launcher script.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for standalone Spotlight launching.
    """

    parser = argparse.ArgumentParser(
        description="Open one exported embedding dataset in Renumics Spotlight."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Explicit path to one spotlight_dataset.jsonl export.",
    )
    parser.add_argument(
        "--strategy",
        dest="strategy_name",
        help="Strategy folder used when resolving the latest exported dataset.",
    )
    parser.add_argument(
        "--run-id",
        help="Run identifier used when resolving the exported dataset.",
    )
    return parser


def _print_launch_summary(result: SpotlightLaunchResult) -> None:
    """
    Print a concise summary of the Spotlight dataset that was opened.

    Parameters
    ----------
    result : SpotlightLaunchResult
        Result returned by the launcher helper.
    """

    print(f"[INFO] Spotlight dataset opened: {result.dataset_path}")
    print(f"[INFO] Spotlight dataset rows: {result.record_count}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the Spotlight viewer launcher from the command line.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Optional argument list used instead of `sys.argv`.

    Returns
    -------
    int
        Process exit code for shell execution.
    """

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        result = launch_spotlight_viewer(
            dataset_path=arguments.dataset_path,
            strategy_name=arguments.strategy_name,
            run_id=arguments.run_id,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    _print_launch_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
