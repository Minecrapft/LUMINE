"""
LUMINE Color Analysis - SALIVA MODE
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ColorAnalyzer:
    """Color analysis with perceptual comparison - SALIVA MODE."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'color_space': 'rgb',
            'distance_metric': 'euclidean',
            'normalize': True,
            'reference_mode': 'rgb'
        }
    
    def compare_to_reference(self, color: np.ndarray, reference_colors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare color to reference colors and find best match."""
        result = {
            'best_match': None,
            'match_quality': 'poor',
            'distance': float('inf'),
            'all_matches': []
        }
        
        if color is None or not reference_colors:
            return result
        
        # Ensure color is numpy array
        if isinstance(color, list):
            color = np.array(color)
        
        # Get RGB values
        if len(color) >= 3:
            r, g, b = color[0], color[1], color[2]
        else:
            r, g, b = 128, 128, 128
        
        for ref in reference_colors:
            ref_color = ref.get('rgb')
            if ref_color is None:
                continue
            
            # Ensure ref_color is list/tuple
            if isinstance(ref_color, str):
                try:
                    ref_color = json.loads(ref_color)
                except:
                    continue
            
            if len(ref_color) >= 3:
                rr, rg, rb = ref_color[0], ref_color[1], ref_color[2]
            else:
                continue
            
            # Calculate color distance
            distance = np.sqrt((r - rr)**2 + (g - rg)**2 + (b - rb)**2)
            
            # Convert RGB to perceived brightness
            brightness_r = (r * 0.299 + g * 0.587 + b * 0.114)
            brightness_ref = (rr * 0.299 + rg * 0.587 + rb * 0.114)
            brightness_diff = abs(brightness_r - brightness_ref)
            
            # Combined distance (color + brightness)
            combined_distance = distance + brightness_diff * 0.3
            
            match_info = {
                'level': ref.get('level', 'unknown'),
                'distance': combined_distance,
                'color': ref_color,
                'reference_id': ref.get('reference_id', '')
            }
            result['all_matches'].append(match_info)
            
            if combined_distance < result['distance']:
                result['distance'] = combined_distance
                result['best_match'] = match_info
        
        # Determine match quality - more forgiving for saliva
        if result['distance'] < 60:
            result['match_quality'] = 'excellent'
        elif result['distance'] < 100:
            result['match_quality'] = 'good'
        elif result['distance'] < 150:
            result['match_quality'] = 'fair'
        else:
            result['match_quality'] = 'poor'
        
        return result
    
    def determine_marker_result(self, color: np.ndarray, reference_colors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine marker result from color comparison."""
        comparison = self.compare_to_reference(color, reference_colors)
        
        result = {
            'value': 'unknown',
            'level': 'unknown',
            'match_quality': 'poor',
            'distance': float('inf'),
            'confidence': 0.0
        }
        
        if comparison['best_match']:
            result['level'] = comparison['best_match']['level']
            result['value'] = comparison['best_match']['level']
            result['distance'] = comparison['distance']
            result['match_quality'] = comparison['match_quality']
            
            # Map distance to confidence (more forgiving)
            if comparison['distance'] < 60:
                result['confidence'] = 0.9
            elif comparison['distance'] < 100:
                result['confidence'] = 0.7
            elif comparison['distance'] < 150:
                result['confidence'] = 0.5
            else:
                result['confidence'] = 0.3
        
        return result
    
    def normalize_color(self, color: np.ndarray, reference_color: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalize color for consistent analysis."""
        if color is None:
            return np.array([128, 128, 128])
        
        if isinstance(color, list):
            color = np.array(color)
        
        # Ensure we have RGB
        if len(color) < 3:
            return np.array([128, 128, 128])
        
        # Simple normalization
        normalized = color.copy()
        
        # If reference color provided, adjust
        if reference_color is not None:
            if isinstance(reference_color, list):
                reference_color = np.array(reference_color)
            if len(reference_color) >= 3:
                # Simple white balance
                scale = np.mean(reference_color) / (np.mean(normalized) + 1)
                normalized = normalized * scale
                normalized = np.clip(normalized, 0, 255)
        
        return normalized.astype(int)
    
    def analyze_marker_color(self, marker_name: str, color: np.ndarray, 
                            reference_library: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze a single marker with reference library."""
        result = {
            'marker': marker_name,
            'color': color.tolist() if color is not None else None,
            'result': 'unknown',
            'match_quality': 'poor',
            'confidence': 0.0
        }
        
        if marker_name not in reference_library:
            # Try to find by marker in any reference
            found = False
            for key, refs in reference_library.items():
                if marker_name.lower() in key.lower():
                    reference_colors = refs
                    found = True
                    break
            
            if not found:
                return result
        else:
            reference_colors = reference_library[marker_name]
        
        if not reference_colors:
            return result
        
        # Determine result from color comparison
        determination = self.determine_marker_result(color, reference_colors)
        
        result['result'] = determination['level']
        result['match_quality'] = determination['match_quality']
        result['confidence'] = determination['confidence']
        
        # If confidence is low, mark as unknown
        if result['confidence'] < 0.3:
            result['result'] = 'unknown'
        
        return result
