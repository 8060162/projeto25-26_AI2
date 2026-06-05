"""Módulo Memory Reporting — métricas de memória longa para administração."""
from .aggregator import MongoMemoryAggregator, StubMemoryAggregator, MemorySnapshotAggregator
from .reader     import MongoMemoryReportReader, StubMemoryReportReader, MemoryReportReader
from .scheduler  import create_memory_scheduler, MemorySchedulerConfig
from .router     import router as memory_reports_router

__all__ = [
    "MongoMemoryAggregator", "StubMemoryAggregator", "MemorySnapshotAggregator",
    "MongoMemoryReportReader", "StubMemoryReportReader", "MemoryReportReader",
    "create_memory_scheduler", "MemorySchedulerConfig",
    "memory_reports_router",
]