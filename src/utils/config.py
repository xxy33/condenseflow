"""
配置管理工具

提供配置文件加载和合并功能。
"""

import os
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件。

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], config_path: str):
    """
    保存配置到YAML文件。

    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典。

    后面的配置会覆盖前面的配置。

    Args:
        *configs: 配置字典列表

    Returns:
        合并后的配置
    """
    result = {}

    for config in configs:
        result = _deep_merge(result, config)

    return result


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个字典。

    Args:
        base: 基础字典
        override: 覆盖字典

    Returns:
        合并后的字典
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
    通过点分隔的路径获取配置值。

    Args:
        config: 配置字典
        key_path: 点分隔的键路径，如 "model.num_layers"
        default: 默认值

    Returns:
        配置值或默认值
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
    通过点分隔的路径设置配置值。

    Args:
        config: 配置字典
        key_path: 点分隔的键路径
        value: 要设置的值
    """
    keys = key_path.split('.')
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


class Config:
    """配置类，提供便捷的配置访问"""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化配置。

        Args:
            config_dict: 配置字典
        """
        self._config = config_dict or {}

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """从文件加载配置"""
        return cls(load_config(config_path))

    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值"""
        return get_config_value(self._config, key_path, default)

    def set(self, key_path: str, value: Any):
        """设置配置值"""
        set_config_value(self._config, key_path, value)

    def merge(self, other: "Config"):
        """合并另一个配置"""
        self._config = merge_configs(self._config, other._config)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()

    def save(self, config_path: str):
        """保存到文件"""
        save_config(self._config, config_path)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._config
