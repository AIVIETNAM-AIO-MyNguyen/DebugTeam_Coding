"""
Hierarchical YAML Configuration System.

Provides:
- Easy attribute access: `cfg.model.hidden_dim` as well as `cfg["model"]["hidden_dim"]`
- Deep merging of defaults and specialized configs
- Automatic CLI overrides (e.g. `--training.lr 1e-4 --data.batch_size 128`)
- YAML serialization for experiment provenance
"""

import argparse
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class ConfigDict(dict):
    """
    Dictionary subclass supporting recursive attribute-style access.
    Example:
        cfg = ConfigDict({'model': {'dim': 512}})
        print(cfg.model.dim)  # 512
        cfg.model.dim = 256
    """

    def __init__(self, mapping: Optional[Union[Dict[str, Any], "ConfigDict"]] = None, **kwargs):
        super().__init__()
        if mapping:
            if isinstance(mapping, dict):
                for key, val in mapping.items():
                    self[key] = val
        for key, val in kwargs.items():
            self[key] = val

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Recursively converts ConfigDict back to standard Python dict."""
        out = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [item.to_dict() if isinstance(item, ConfigDict) else item for item in v]
            else:
                out[k] = v
        return out

    def clone(self) -> "ConfigDict":
        """Creates a deep copy of the configuration."""
        import copy
        return ConfigDict(copy.deepcopy(self.to_dict()))


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges `update` dictionary into `base`.
    Preserves existing keys in base unless overwritten by update.
    """
    merged = dict(base)
    for key, val in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _parse_value(val_str: str) -> Any:
    """Parses a string into a typed Python object (int, float, bool, list, or str)."""
    # Check booleans
    if val_str.lower() == "true":
        return True
    if val_str.lower() == "false":
        return False
    if val_str.lower() in ("none", "null"):
        return None

    # Try literal eval for numbers, lists, tuples, dicts
    try:
        return ast.literal_eval(val_str)
    except (ValueError, SyntaxError):
        # Fallback to raw string
        return val_str


def parse_cli_overrides(args_list: List[str]) -> Dict[str, Any]:
    """
    Parses a list of CLI arguments with dot-notation overrides into a nested dictionary.
    Example:
        ['--training.lr', '1e-4', '--data.batch_size', '128', '--training.deterministic', 'True']
        -> {'training': {'lr': 0.0001, 'deterministic': True}, 'data': {'batch_size': 128}}
    """
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg.startswith("--"):
            key_path = arg[2:]
            if "=" in key_path:
                key_path, val_str = key_path.split("=", 1)
                i += 1
            else:
                if i + 1 < len(args_list) and not args_list[i + 1].startswith("--"):
                    val_str = args_list[i + 1]
                    i += 2
                else:
                    # Flag boolean True
                    val_str = "true"
                    i += 1

            keys = key_path.split(".")
            curr = overrides
            for k in keys[:-1]:
                curr = curr.setdefault(k, {})
            curr[keys[-1]] = _parse_value(val_str)
        else:
            i += 1
    return overrides


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    default_config_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[List[str]] = None,
) -> ConfigDict:
    """
    Loads and merges configuration from YAML files and optional CLI overrides.

    Args:
        config_path: Path to experiment-specific YAML config.
        default_config_path: Path to base/default YAML config (optional).
        cli_overrides: Optional list of command-line override strings.

    Returns:
        ConfigDict object ready for use.
    """
    merged_dict: Dict[str, Any] = {}

    # 1. Load default config if provided
    if default_config_path:
        default_p = Path(default_config_path)
        if default_p.is_file():
            with open(default_p, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
                merged_dict = deep_merge(merged_dict, content)

    # 2. Load experiment config
    if config_path:
        exp_p = Path(config_path)
        if not exp_p.is_file():
            raise FileNotFoundError(f"Configuration file not found: {exp_p}")
        with open(exp_p, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
            merged_dict = deep_merge(merged_dict, content)

    # 3. Apply CLI overrides
    if cli_overrides:
        overrides = parse_cli_overrides(cli_overrides)
        merged_dict = deep_merge(merged_dict, overrides)

    return ConfigDict(merged_dict)


def save_config(config: Union[ConfigDict, Dict[str, Any]], out_path: Union[str, Path]) -> None:
    """Saves a ConfigDict or dict to a YAML file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dict = config.to_dict() if isinstance(config, ConfigDict) else config
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw_dict, f, default_flow_style=False, sort_keys=False)
