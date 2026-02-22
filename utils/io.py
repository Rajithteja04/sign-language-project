from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path, local_config_path: str | Path | None = None) -> dict:
    """
    Load base config and optionally overlay a local machine-specific config.
    Local config keys override base keys.
    """
    config = load_yaml(config_path)

    if local_config_path is None:
        return config

    local_path = Path(local_config_path)
    if local_path.exists():
        local_cfg = load_yaml(local_path)
        config.update(local_cfg)

    return config
