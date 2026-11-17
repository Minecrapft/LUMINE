"""
LUMINE Utilities
Helper functions for common operations.
"""

import os
import re
import json
import uuid
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# ========================================
# ID Generation
# ========================================

def generate_patient_id() -> str:
    """Generate a unique patient ID."""
    return f"P{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"

def generate_assessment_id() -> str:
    """Generate a unique assessment ID."""
    return f"A{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"

def generate_calibration_id() -> str:
    """Generate a unique calibration ID."""
    return f"CAL-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

# ========================================
# Data Validation
# ========================================

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    pattern = r'^[\d\s\-+()]{10,15}$'
    return bool(re.match(pattern, phone))

def validate_date(date_str: str) -> bool:
    """Validate date format (YYYY-MM-DD)."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def sanitize_input(text: str) -> str:
    """Sanitize user input."""
    if not text:
        return ''
    # Remove HTML tags and special characters
    return re.sub(r'[<>&\'"]', '', text).strip()

# ========================================
# Data Formatting
# ========================================

def format_date(date_str: str, format: str = '%b %d, %Y') -> str:
    """Format date string."""
    if not date_str:
        return ''
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime(format)
    except:
        return date_str

def format_datetime(dt_str: str, format: str = '%b %d, %Y %I:%M %p') -> str:
    """Format datetime string."""
    if not dt_str:
        return ''
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime(format)
    except:
        return dt_str

def format_phone(phone: str) -> str:
    """Format phone number."""
    if not phone:
        return ''
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + '...'

# ========================================
# Security Utilities
# ========================================

def hash_string(value: str) -> str:
    """Hash a string using SHA256."""
    return hashlib.sha256(value.encode()).hexdigest()

def generate_token() -> str:
    """Generate a secure random token."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode()

def mask_sensitive_data(text: str, visible_start: int = 4, visible_end: int = 4) -> str:
    """Mask sensitive data with asterisks."""
    if len(text) <= visible_start + visible_end:
        return '*' * len(text)
    return text[:visible_start] + '*' * (len(text) - visible_start - visible_end) + text[-visible_end:]

# ========================================
# JSON Helpers
# ========================================

def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely load JSON data."""
    if not data:
        return default
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(data: Any, default: Any = None) -> str:
    """Safely dump JSON data."""
    try:
        return json.dumps(data)
    except TypeError:
        return json.dumps(default) if default is not None else '{}'

# ========================================
# File Helpers
# ========================================

def ensure_directory(path: str) -> bool:
    """Ensure directory exists."""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(path)
    except:
        return 0

def get_file_extension(path: str) -> str:
    """Get file extension."""
    return os.path.splitext(path)[1].lower()

def is_image_file(path: str) -> bool:
    """Check if file is an image."""
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    return get_file_extension(path) in extensions

# ========================================
# Color Utilities
# ========================================

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def is_light_color(rgb: tuple) -> bool:
    """Check if color is light."""
    return (rgb[0] + rgb[1] + rgb[2]) / 3 > 128

def get_contrast_color(rgb: tuple) -> tuple:
    """Get contrasting color."""
    return (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])

# ========================================
# Metric Helpers
# ========================================

def calculate_percentage(value: float, total: float) -> float:
    """Calculate percentage."""
    if total == 0:
        return 0.0
    return (value / total) * 100

def calculate_change(current: float, previous: float) -> Dict[str, Any]:
    """Calculate change between values."""
    if previous == 0:
        return {
            'absolute': current,
            'percentage': 100.0 if current > 0 else 0.0,
            'direction': 'increase' if current > 0 else 'stable'
        }
    
    absolute = current - previous
    percentage = (absolute / previous) * 100
    
    if percentage > 5:
        direction = 'increase'
    elif percentage < -5:
        direction = 'decrease'
    else:
        direction = 'stable'
    
    return {
        'absolute': round(absolute, 2),
        'percentage': round(percentage, 1),
        'direction': direction
    }

# ========================================
# Status Helpers
# ========================================

def get_status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        'normal': '#22c55e',
        'attention': '#f59e0b',
        'followup_recommended': '#ef4444',
        'invalid': '#6b7280',
        'pending': '#8b5cf6',
        'pass': '#22c55e',
        'fail': '#ef4444',
        'warning': '#f59e0b'
    }
    return colors.get(status.lower(), '#6b7280')

def get_status_icon(status: str) -> str:
    """Get icon for status."""
    icons = {
        'normal': '✅',
        'attention': '⚠️',
        'followup_recommended': '🔴',
        'invalid': '❌',
        'pending': '⏳',
        'pass': '✅',
        'fail': '❌',
        'warning': '⚠️'
    }
    return icons.get(status.lower(), '❓')

def get_status_label(status: str) -> str:
    """Get human-readable status label."""
    labels = {
        'normal': 'Normal',
        'attention': 'Attention',
        'followup_recommended': 'Follow-up Recommended',
        'invalid': 'Invalid',
        'pending': 'Pending',
        'pass': 'Pass',
        'fail': 'Fail'
    }
    return labels.get(status.lower(), status.title())

# ========================================
# Age Calculation
# ========================================

def calculate_age(birth_date: str) -> Optional[int]:
    """Calculate age from birth date."""
    if not birth_date:
        return None
    try:
        birth = datetime.fromisoformat(birth_date)
        today = datetime.now()
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except:
        return None

# ========================================
# Dictionary Helpers
# ========================================

def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """Merge two dictionaries."""
    result = dict1.copy()
    if deep:
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
    else:
        result.update(dict2)
    return result

def filter_dict(data: Dict, keys: List[str]) -> Dict:
    """Filter dictionary to only include specified keys."""
    return {k: v for k, v in data.items() if k in keys}

def exclude_keys(data: Dict, keys: List[str]) -> Dict:
    """Exclude specified keys from dictionary."""
    return {k: v for k, v in data.items() if k not in keys}

# ========================================
# Class for lazy loading
# ========================================

class LazyLoader:
    """Lazy loader for expensive operations."""
    
    def __init__(self, load_func, *args, **kwargs):
        self.load_func = load_func
        self.args = args
        self.kwargs = kwargs
        self._loaded = False
        self._data = None
    
    def __call__(self):
        if not self._loaded:
            self._data = self.load_func(*self.args, **self.kwargs)
            self._loaded = True
        return self._data
    
    def get(self):
        return self.__call__()
    
    def clear(self):
        self._loaded = False
        self._data = None

# ========================================
# Version Helpers
# ========================================

def parse_version(version: str) -> tuple:
    """Parse version string to tuple."""
    try:
        return tuple(int(x) for x in version.split('.'))
    except:
        return (0, 0, 0)

def compare_versions(version1: str, version2: str) -> int:
    """Compare two versions. Returns -1, 0, or 1."""
    v1 = parse_version(version1)
    v2 = parse_version(version2)
    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    return 0

# ========================================
# Export all utilities
# ========================================

__all__ = [
    'generate_patient_id',
    'generate_assessment_id',
    'generate_calibration_id',
    'validate_email',
    'validate_phone',
    'validate_date',
    'sanitize_input',
    'format_date',
    'format_datetime',
    'format_phone',
    'truncate_text',
    'hash_string',
    'generate_token',
    'mask_sensitive_data',
    'safe_json_loads',
    'safe_json_dumps',
    'ensure_directory',
    'get_file_size',
    'get_file_extension',
    'is_image_file',
    'hex_to_rgb',
    'rgb_to_hex',
    'is_light_color',
    'get_contrast_color',
    'calculate_percentage',
    'calculate_change',
    'get_status_color',
    'get_status_icon',
    'get_status_label',
    'calculate_age',
    'merge_dicts',
    'filter_dict',
    'exclude_keys',
    'LazyLoader',
    'parse_version',
    'compare_versions'
]