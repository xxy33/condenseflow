"""
Configuration Management Utilities

Provides configuration file loading and merging functionality.
"""

import os
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: configuration file path

    Returns:
        configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to a YAML file.

    Args:
        config: configuration dictionary
        config_path: configuration file path
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Later configs override earlier ones.

    Args:
        *configs: list of configuration dictionaries

    Returns:
        merged configuration
    """
    result = {}

    for config in configs:
        result = _deep_merge(result, config)

    return result


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Args:
        base: base dictionary
        override: override dictionary

    Returns:
        merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value by dot-separated path.

    Args:
        config: configuration dictionary
        key_path: dot-separated key path, e.g., "model.num_layers"
        default: default value

    Returns:
        configuration value or default
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def set_config_value(config: Dict[str, Any], key_path: str, value: Any):
    """
    Set configuration value by dot-separated path.

    Args:
        config: configuration dictionary
        key_path: dot-separated key path
        value: value to set
    """
    keys = key_path.split('.')
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


class Config:
    """Configuration class, provides convenient configuration access"""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: configuration dictionary
        """
        self._config = config_dict or {}

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from file"""
        return cls(load_config(config_path))

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value"""
        return get_config_value(self._config, key_path, default)

    def set(self, key_path: str, value: Any):
        """Set configuration value"""
        set_config_value(self._config, key_path, value)

    def merge(self, other: "Config"):
        """Merge another configuration"""
        self._config = merge_configs(self._config, other._config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self._config.copy()

    def save(self, config_path: str):
        """Save to file"""
        save_config(self._config, config_path)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._config
