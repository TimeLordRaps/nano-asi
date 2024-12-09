"""Core ASI implementation and configuration."""

from .asi import ASI
from .config import Config
from .architecture import ComponentProtocol, ComponentRegistry
from .config_manager import ConfigManager
from .dependency_injection import DependencyContainer, inject
from .interfaces import (
    ComponentProtocol,
    ModelAdapterProtocol,
    JudgmentProtocol,
    ComponentConfig
)

__all__ = [
    "ASI",
    "Config",
    "ComponentProtocol",
    "ComponentRegistry",
    "ConfigManager",
    "DependencyContainer",
    "inject",
    "ComponentProtocol",
    "ModelAdapterProtocol",
    "JudgmentProtocol",
    "ComponentConfig"
]
