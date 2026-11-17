"""
LUMINE Image Quality Checker
Comprehensive image quality assessment for strip analysis.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

class ImageQualityChecker:
    """Check image quality for strip analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'min_brightness': 80,
            'max_brightness': 220,
            'min_contrast': 30,
            'min_blur_threshold': 100.0,
            'glare_threshold': 0.3,
            'min_resolution': (1280, 720)
        }
    
    def check(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive quality check."""
        results = {
            'pass': False,
            'checks': {},
            'issues': [],
            'score': 0
        }
        
        if image is None:
            results['issues'].append('No image data')
            return results
        
        h, w = image.shape[:2]
        
        # Check resolution
        resolution_ok = self._check_resolution(w, h)
        results['checks']['resolution'] = {
            'pass': resolution_ok,
            'width': w,
            'height': h,
            'required': self.config['min_resolution']
        }
        if not resolution_ok:
            results['issues'].append(f'Resolution too low: {w}x{h}')
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Check brightness
        brightness = np.mean(gray)
        brightness_ok = self.config['min_brightness'] < brightness < self.config['max_brightness']
        results['checks']['brightness'] = {
            'pass': brightness_ok,
            'value': float(brightness),
            'range': (self.config['min_brightness'], self.config['max_brightness'])
        }
        if not brightness_ok:
            if brightness <= self.config['min_brightness']:
                results['issues'].append('Underexposed image')
            else:
                results['issues'].append('Overexposed image')
        
        # Check contrast
        contrast = np.std(gray)
        contrast_ok = contrast >= self.config['min_contrast']
        results['checks']['contrast'] = {
            'pass': contrast_ok,
            'value': float(contrast),
            'minimum': self.config['min_contrast']
        }
        if not contrast_ok:
            results['issues'].append('Low contrast')
        
        # Check focus (blur)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        focus_ok = blur_score >= self.config['min_blur_threshold']
        results['checks']['focus'] = {
            'pass': focus_ok,
            'value': float(blur_score),
            'threshold': self.config['min_blur_threshold']
        }
        if not focus_ok:
            results['issues'].append('Image blurry')
        
        # Check glare
        glare_ok, glare_score = self._check_glare(image)
        results['checks']['glare'] = {
            'pass': glare_ok,
            'value': float(glare_score),
            'threshold': self.config['glare_threshold']
        }
        if not glare_ok:
            results['issues'].append('Excessive glare detected')
        
        # Overall pass/fail
        critical_checks = ['resolution', 'brightness', 'focus']
        all_critical_pass = all(results['checks'].get(c, {}).get('pass', False) for c in critical_checks)
        
        # Calculate score
        passed = sum(1 for c in results['checks'].values() if c.get('pass', False))
        total = len(results['checks'])
        results['score'] = int((passed / total) * 100) if total > 0 else 0
        
        results['pass'] = all_critical_pass and len(results['issues']) == 0
        
        return results
    
    def _check_resolution(self, width: int, height: int) -> bool:
        """Check if image meets minimum resolution."""
        min_w, min_h = self.config['min_resolution']
        return width >= min_w and height >= min_h
    
    def _check_glare(self, image: np.ndarray) -> Tuple[bool, float]:
        """Check for excessive glare/specular highlights."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use histogram to detect saturation
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        
        # Percentage of pixels in high brightness range
        glare_ratio = float(np.sum(hist[240:256]))
        
        return glare_ratio < self.config['glare_threshold'], glare_ratio
    
    def check_strip_position(self, image: np.ndarray, strip_roi: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Check if strip is properly positioned."""
        h, w = image.shape[:2]
        x, y, sw, sh = strip_roi
        
        results = {
            'pass': True,
            'issues': [],
            'position': {
                'x': x, 'y': y, 'width': sw, 'height': sh,
                'image_width': w, 'image_height': h
            }
        }
        
        # Check if strip is centered
        center_x = x + sw / 2
        center_y = y + sh / 2
        image_center_x = w / 2
        image_center_y = h / 2
        
        offset_x = abs(center_x - image_center_x) / image_center_x
        offset_y = abs(center_y - image_center_y) / image_center_y
        
        if offset_x > 0.3 or offset_y > 0.3:
            results['pass'] = False
            results['issues'].append('Strip not centered')
        
        # Check if strip is complete
        if sw < w * 0.3 or sh < h * 0.3:
            results['pass'] = False
            results['issues'].append('Strip appears incomplete')
        
        if sw > w * 0.9 or sh > h * 0.9:
            results['pass'] = False
            results['issues'].append('Strip too large for frame')
        
        return results
    
    def get_quality_summary(self, quality_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get human-readable quality summary."""
        summary = {
            'overall': 'Pass' if quality_results.get('pass', False) else 'Fail',
            'details': {}
        }
        
        for check_name, check_data in quality_results.get('checks', {}).items():
            status = 'Pass' if check_data.get('pass', False) else 'Fail'
            if check_name == 'brightness':
                summary['details'][check_name] = f"{status} ({check_data.get('value', 0):.0f})"
            elif check_name == 'focus':
                summary['details'][check_name] = f"{status} ({check_data.get('value', 0):.0f})"
            elif check_name == 'contrast':
                summary['details'][check_name] = f"{status} ({check_data.get('value', 0):.0f})"
            elif check_name == 'glare':
                summary['details'][check_name] = f"{status} ({check_data.get('value', 0):.2f})"
            else:
                summary['details'][check_name] = status
        
        if quality_results.get('issues'):
            summary['issues'] = quality_results['issues']
        
        summary['score'] = quality_results.get('score', 0)
        
        return summary