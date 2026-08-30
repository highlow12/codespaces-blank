"""Portable clustering core intended for Pyodide/Web Worker use."""

from .api import cluster_documents
from .discovery import dependency_status
from .types import DiscoveryDependencyError

__all__ = ["cluster_documents", "dependency_status", "DiscoveryDependencyError"]
