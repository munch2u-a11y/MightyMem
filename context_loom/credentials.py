"""Safe Credentials & Secret Key Store Management for MightyMem.

Manages credentials with multi-tiered resolution, prioritizing hidden user-level
environment files (~/.mightymem/.env) and project overrides.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard credential keys used by MightyMem channels and adapters
CREDENTIAL_KEYS = {
    "TELEGRAM_BOT_TOKEN": "Telegram Bot API Token (from @BotFather)",
    "DISCORD_BOT_TOKEN": "Discord Bot API Token (from Discord Developer Portal)",
    "DISCORD_CHANNEL_ID": "Discord Channel ID for bot polling & message routing",
    "PIPER_MODEL_PATH": "Path to local Piper TTS ONNX model",
}


# Prefix patterns allowed for environment variable loading from .env files
ALLOWED_ENV_PREFIXES = ("MIGHTYMEM_", "TELEGRAM_", "DISCORD_", "PIPER_")


def get_default_credentials_dir() -> Path:
    """Return the secure hidden user credentials directory (~/.mightymem) with 0700 permissions."""
    dir_path = Path.home() / ".mightymem"
    dir_path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(dir_path, 0o700)
    except Exception as err:
        logger.debug("Could not set 0700 permissions on %s: %s", dir_path, err)
    return dir_path


def get_credentials_file_paths() -> List[Path]:
    """Return ordered list of candidate .env credential file paths to search.
    
    Priority order:
    1. $MIGHTYMEM_ENV_FILE environment variable (if defined)
    2. ~/.mightymem/.env (hidden user home directory credentials)
    3. ~/.env (user home root environment file)
    4. .env in current working directory
    """
    paths: List[Path] = []
    
    custom_env = os.environ.get("MIGHTYMEM_ENV_FILE")
    if custom_env:
        paths.append(Path(custom_env).expanduser().resolve())
        
    hidden_mightymem_env = Path.home() / ".mightymem" / ".env"
    paths.append(hidden_mightymem_env)
    
    user_home_env = Path.home() / ".env"
    paths.append(user_home_env)
    
    cwd_env = Path.cwd() / ".env"
    paths.append(cwd_env)
    
    return paths


def parse_env_file(file_path: Path) -> Dict[str, str]:
    """Safely parse key=value lines from a .env file without external dependencies."""
    credentials: Dict[str, str] = {}
    if not file_path.exists() or not file_path.is_file():
        return credentials

    try:
        content = file_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            
            # Remove surrounding matching single or double quotes
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
                
            credentials[key] = val
    except Exception as err:
        logger.warning("Failed to parse credentials file %s: %s", file_path, err)
        
    return credentials


def _is_allowed_credential_key(key: str) -> bool:
    """Check if key is in known CREDENTIAL_KEYS or matches allowed prefixes."""
    if key in CREDENTIAL_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in ALLOWED_ENV_PREFIXES)


def load_credentials(override_existing: bool = False) -> Dict[str, str]:
    """Load filtered credentials from all available .env files into os.environ."""
    loaded: Dict[str, str] = {}
    
    # Iterate in reverse priority so higher priority files overwrite lower priority values
    for env_path in reversed(get_credentials_file_paths()):
        parsed = parse_env_file(env_path)
        for k, v in parsed.items():
            # Filter to known credential keys or allowed prefixes to prevent env pollution
            if _is_allowed_credential_key(k):
                if override_existing or k not in os.environ:
                    os.environ[k] = v
                loaded[k] = v
            
    return loaded


def get_credential(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve a credential key, checking os.environ and credential files if missing."""
    if key in os.environ:
        return os.environ[key]
        
    # Attempt lazy loading from credential files
    load_credentials(override_existing=False)
    return os.environ.get(key, default)


def save_credential(key: str, value: str, target_file: Optional[Path] = None) -> Path:
    """Save or update a credential key=value pair into a target file using atomic 0600 permissions."""
    if target_file is None:
        target_file = get_default_credentials_dir() / ".env"
        
    target_file.parent.mkdir(parents=True, exist_ok=True)
    
    lines: List[str] = []
    key_updated = False
    
    if target_file.exists():
        content = target_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, _ = stripped.split("=", 1)
                if k.strip() == key:
                    lines.append(f"{key}={value}")
                    key_updated = True
                    continue
            lines.append(line)
            
    if not key_updated:
        lines.append(f"{key}={value}")
        
    content_str = "\n".join(lines) + "\n"
    
    # Atomic creation with strict 0600 permissions on POSIX systems
    if os.name == "posix":
        fd = os.open(str(target_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content_str)
    else:
        target_file.write_text(content_str, encoding="utf-8")
        
    os.environ[key] = value
    return target_file
