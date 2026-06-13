"""writer — agentic, provider-agnostic long-form novel writer."""

from __future__ import annotations

from .config import Provider, get_provider, load_providers
from .engine import StageError, Writer
from .models import Characters, Concept, Outline, World
from .project import Project, ProjectConfig

__version__ = "0.2.0"

__all__ = [
    "Writer",
    "Project",
    "ProjectConfig",
    "StageError",
    "Provider",
    "get_provider",
    "load_providers",
    "Concept",
    "Characters",
    "World",
    "Outline",
    "__version__",
]
