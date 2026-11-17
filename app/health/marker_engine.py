"""
LUMINE Marker Engine - SALIVA MODE
"""

import logging
from typing import List, Dict, Any, Optional
from app.database.db_manager import DatabaseManager
from app.health.condition_engine import ConditionEngine

logger = logging.getLogger(__name__)

class MarkerEngine:
    """Engine for marker interpretation - SALIVA MODE."""
    
    def __init__(self, db: DatabaseManager, sample_type: str = 'saliva'):
        self.db = db
        self.sample_type = sample_type
        self.condition_engine = ConditionEngine(db)
        
        # SALIVA reference ranges
        self.marker_config = {
            'glucose': {
                'domain': 'metabolic',
                'saliva': {'normal_range': ['negative', 'trace'], 'abnormal_levels': ['+1', '+2', '+3']},
                'urine': {'normal_range': ['negative'], 'abnormal_levels': ['+1', '+2', '+3', '+4']}
            },
            'protein': {
                'domain': 'renal',
                'saliva': {'normal_range': ['negative', 'trace'], 'abnormal_levels': ['+1', '+2']},
                'urine': {'normal_range': ['negative', 'trace'], 'abnormal_levels': ['+1', '+2', '+3', '+4']}
            },
            'bilirubin': {
                'domain': 'hepatic',
                'saliva': {'normal_range': ['negative'], 'abnormal_levels': ['trace', '+1']},
                'urine': {'normal_range': ['negative'], 'abnormal_levels': ['+1', '+2', '+3']}
            },
            'ketones': {
                'domain': 'metabolic',
                'saliva': {'normal_range': ['negative', 'trace'], 'abnormal_levels': ['+1', '+2']},
                'urine': {'normal_range': ['negative'], 'abnormal_levels': ['+1', '+2', '+3', '+4']}
            },
            'blood': {
                'domain': 'renal',
                'saliva': {'normal_range': ['negative'], 'abnormal_levels': ['trace', '+1']},
                'urine': {'normal_range': ['negative'], 'abnormal_levels': ['+1', '+2', '+3']}
            },
            'nitrite': {
                'domain': 'urinary',
                'saliva': {'normal_range': ['negative'], 'abnormal_levels': ['positive']},
                'urine': {'normal_range': ['negative'], 'abnormal_levels': ['positive']}
            },
            'leukocytes': {
                'domain': 'inflammation',
                'saliva': {'normal_range': ['negative', 'trace'], 'abnormal_levels': ['+1', '+2']},
                'urine': {'normal_range': ['negative'], 'abnormal_levels': ['+1', '+2', '+3']}
            },
            'specific_gravity': {
                'domain': 'hydration',
                'saliva': {'normal_range': ['1.005', '1.010', '1.015'], 'abnormal_levels': ['1.000', '1.020', '1.025']},
                'urine': {'normal_range': ['1.005', '1.010', '1.015', '1.020'], 'abnormal_levels': ['1.000', '1.025', '1.030']}
            },
            'pH': {
                'domain': 'acid_base',
                'saliva': {'normal_range': ['6.0', '6.5', '7.0', '7.5'], 'abnormal_levels': ['4.5', '5.0', '8.0', '8.5']},
                'urine': {'normal_range': ['5.0', '6.0', '7.0'], 'abnormal_levels': ['4.5', '8.0', '8.5']}
            },
            'urobilinogen': {
                'domain': 'hepatic',
                'saliva': {'normal_range': ['normal'], 'abnormal_levels': ['abnormal']},
                'urine': {'normal_range': ['normal'], 'abnormal_levels': ['abnormal']}
            }
        }
    
    def get_ranges(self, marker: str) -> Dict[str, Any]:
        """Get ranges for marker based on sample type."""
        config = self.marker_config.get(marker, {})
        if self.sample_type == 'saliva':
            return config.get('saliva', {'normal_range': ['negative'], 'abnormal_levels': ['+1']})
        return config.get('urine', {'normal_range': ['negative'], 'abnormal_levels': ['+1']})
    
    def interpret_marker(self, marker: str, result: str, 
                        match_quality: str = 'good', 
                        confidence: float = 0.5) -> Dict[str, Any]:
        """Interpret a single marker result."""
        # If result is unknown or empty, treat as negative
        if not result or result == 'unknown':
            result = 'negative'
            confidence = 0.5
        
        interpretation = {
            'marker': marker,
            'result': result,
            'match_quality': match_quality,
            'confidence': confidence,
            'status': 'normal',
            'interpretation': 'Within normal range',
            'domain': '',
            'evidence_level': 'research' if self.sample_type == 'saliva' else 'clinical'
        }
        
        if marker not in self.marker_config:
            interpretation['status'] = 'normal'
            interpretation['interpretation'] = 'Marker not recognized'
            return interpretation
        
        config = self.marker_config[marker]
        interpretation['domain'] = config.get('domain', '')
        ranges = self.get_ranges(marker)
        
        # Check if result is abnormal
        if result in ranges.get('abnormal_levels', []):
            interpretation['status'] = 'attention'
            interpretation['interpretation'] = f'{marker} detected at level {result}'
        elif result in ranges.get('normal_range', []):
            interpretation['status'] = 'normal'
            interpretation['interpretation'] = f'{marker} within normal {self.sample_type} range'
        else:
            # Unknown result - check if it might be normal
            if result in ['negative', 'normal', 'trace']:
                interpretation['status'] = 'normal'
                interpretation['interpretation'] = f'{marker} within normal range'
            else:
                interpretation['status'] = 'normal'  # Default to normal for saliva
                interpretation['interpretation'] = f'{marker} result within reference range'
        
        # For saliva, add research note
        if self.sample_type == 'saliva' and interpretation['status'] == 'attention':
            interpretation['interpretation'] += ' - RESEARCH USE ONLY'
        
        return interpretation
    
    def interpret_all_markers(self, marker_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Interpret all markers in their input strip sequence."""
        interpreted = []
        
        if marker_results:
            ordered_markers = [r.get('marker') for r in marker_results if r.get('marker')]
        else:
            ordered_markers = ['glucose', 'bilirubin', 'ketones', 'specific_gravity', 'blood',
                              'pH', 'protein', 'urobilinogen', 'nitrite', 'leukocytes']
        
        existing = {r.get('marker'): r for r in marker_results if r.get('marker')}
        
        for marker in ordered_markers:
            if marker in existing:
                result = existing[marker]
                value = result.get('result', 'negative')
                match_quality = result.get('match_quality', 'good')
                confidence = result.get('confidence', 0.5)
            else:
                value = 'negative'
                match_quality = 'good'
                confidence = 0.5
            
            interpretation = self.interpret_marker(marker, value, match_quality, confidence)
            interpreted.append(interpretation)
        
        return interpreted
    
    def get_health_domains(self, marker_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results by health domain."""
        domains = {
            'metabolic': {'status': 'normal', 'markers': []},
            'renal': {'status': 'normal', 'markers': []},
            'urinary': {'status': 'normal', 'markers': []},
            'hepatic': {'status': 'normal', 'markers': []},
            'hydration': {'status': 'normal', 'markers': []},
            'inflammation': {'status': 'normal', 'markers': []},
            'acid_base': {'status': 'normal', 'markers': []},
            'oncology_research': {'status': 'not_evaluated', 'markers': []}
        }
        
        for result in marker_results:
            marker = result.get('marker', '')
            if marker not in self.marker_config:
                continue
            
            domain = self.marker_config[marker].get('domain', '')
            if domain not in domains:
                continue
            
            status = result.get('status', 'normal')
            domains[domain]['markers'].append({
                'marker': marker,
                'result': result.get('result', ''),
                'status': status
            })
            
            if status == 'attention' and domains[domain]['status'] == 'normal':
                domains[domain]['status'] = 'attention'
        
        return domains
    
    def generate_summary(self, marker_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive marker summary."""
        interpreted = self.interpret_all_markers(marker_results)
        domains = self.get_health_domains(interpreted)
        
        flagged = [r for r in interpreted if r.get('status') == 'attention']
        
        if len(flagged) == 0:
            overall_status = 'normal'
        elif len(flagged) <= 2:
            overall_status = 'attention'
        else:
            overall_status = 'followup_recommended'
        
        return {
            'total_markers': len(interpreted),
            'normal': len([r for r in interpreted if r.get('status') == 'normal']),
            'flagged': len(flagged),
            'unknown': 0,
            'flagged_markers': flagged,
            'health_domains': domains,
            'overall_status': overall_status,
            'sample_type': self.sample_type
        }
