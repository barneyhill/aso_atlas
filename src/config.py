from pathlib import Path
from typing import Optional
import yaml
DEFAULT_CONFIG_PATH = Path.home() / "dphil" / "asogym" / ".asogym" / "config.yaml"

def get_api_key(key_name: str, config_path: Path = DEFAULT_CONFIG_PATH) -> str:
    """Get API key from YAML config file.
    
    Args:
        key_name: Name of the API key to retrieve (e.g., 'openai', 'azure')
        config_path: Path to the YAML config file
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found. Please create {config_path} "
            "with your API keys in YAML format."
        )
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if not config or key_name not in config:
        raise ValueError(f"API key '{key_name}' not found in {config_path}")
        
    api_key = config[key_name]
    if not api_key:
        raise ValueError(f"API key '{key_name}' in {config_path} is empty")
        
    return api_key
