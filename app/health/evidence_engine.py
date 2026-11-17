"""
LUMINE Evidence Engine
Manages evidence levels and weighting for health associations.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class EvidenceEngine:
    """
    Evidence Engine for health association weighting and transparency.
    
    Evidence Levels:
    - Level 1: Validated Screening Association
    - Level 2: Clinical Association
    - Level 3: Research Association
    - Level 4: Experimental
    - Level 5: Unsupported (not shown)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'min_evidence_level': 1,
            'max_evidence_level': 5,
            'default_evidence': 'research'
        }
        
        # Evidence level weights for scoring
        self.evidence_weights = {
            'validated': 1.0,
            'clinical': 0.8,
            'research': 0.5,
            'experimental': 0.3,
            'unsupported': 0.0
        }
        
        # Level labels
        self.level_labels = {
            'validated': 'Validated Screening Association',
            'clinical': 'Clinical Association',
            'research': 'Research Association',
            'experimental': 'Experimental',
            'unsupported': 'Unsupported'
        }
        
        # Level descriptions
        self.level_descriptions = {
            'validated': 'Supported by the selected assay and established interpretation.',
            'clinical': 'Marker may be associated with condition but is not disease-specific.',
            'research': 'Published research exists but the assay is not clinically validated.',
            'experimental': 'Research hypothesis or prototype.',
            'unsupported': 'Not shown as an interpretation.'
        }
    
    def get_evidence_level(self, evidence_str: str) -> int:
        """Convert evidence string to numeric level."""
        mapping = {
            'validated': 1,
            'clinical': 2,
            'research': 3,
            'experimental': 4,
            'unsupported': 5
        }
        return mapping.get(evidence_str.lower(), 3)
    
    def get_evidence_string(self, level: int) -> str:
        """Convert numeric level to evidence string."""
        mapping = {
            1: 'validated',
            2: 'clinical',
            3: 'research',
            4: 'experimental',
            5: 'unsupported'
        }
        return mapping.get(level, 'research')
    
    def get_evidence_label(self, evidence_level: str) -> str:
        """Get human-readable label for evidence level."""
        return self.level_labels.get(evidence_level, 'Unknown')
    
    def get_evidence_description(self, evidence_level: str) -> str:
        """Get description for evidence level."""
        return self.level_descriptions.get(evidence_level, '')
    
    def get_evidence_weight(self, evidence_level: str) -> float:
        """Get weight for evidence level."""
        return self.evidence_weights.get(evidence_level, 0.5)
    
    def calculate_evidence_score(self, associations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall evidence score from associations."""
        if not associations:
            return {
                'score': 0.0,
                'level': 'unsupported',
                'label': 'No evidence',
                'confidence': 0.0
            }
        
        # Calculate weighted score
        total_weight = 0.0
        total_score = 0.0
        
        for assoc in associations:
            level = assoc.get('evidence_level', 'research')
            weight = self.get_evidence_weight(level)
            confidence = assoc.get('confidence', 0.5)
            
            total_weight += weight
            total_score += weight * confidence
        
        if total_weight == 0:
            return {
                'score': 0.0,
                'level': 'unsupported',
                'label': 'No evidence',
                'confidence': 0.0
            }
        
        avg_score = total_score / total_weight
        
        # Determine evidence level from score
        if avg_score >= 0.8:
            level = 'validated'
        elif avg_score >= 0.6:
            level = 'clinical'
        elif avg_score >= 0.4:
            level = 'research'
        elif avg_score >= 0.2:
            level = 'experimental'
        else:
            level = 'unsupported'
        
        return {
            'score': round(avg_score, 2),
            'level': level,
            'label': self.get_evidence_label(level),
            'confidence': round(avg_score, 2)
        }
    
    def grade_evidence(self, marker_results: List[Dict[str, Any]], 
                      conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Grade evidence based on marker results and conditions."""
        grades = {
            'overall': 'unsupported',
            'markers': [],
            'conditions': [],
            'score': 0.0
        }
        
        # Grade each marker
        for marker in marker_results:
            marker_name = marker.get('marker', '')
            status = marker.get('status', 'normal')
            confidence = marker.get('confidence', 0.5)
            
            # Find matching conditions
            matching_conditions = [c for c in conditions if marker_name in c.get('associated_markers', [])]
            
            if matching_conditions:
                # Use highest evidence level from matching conditions
                levels = [c.get('evidence_level', 'research') for c in matching_conditions]
                best_level = self.get_highest_evidence(levels)
                
                grades['markers'].append({
                    'marker': marker_name,
                    'evidence_level': best_level,
                    'label': self.get_evidence_label(best_level),
                    'confidence': confidence,
                    'status': status
                })
            else:
                grades['markers'].append({
                    'marker': marker_name,
                    'evidence_level': 'unsupported',
                    'label': 'Unsupported',
                    'confidence': confidence,
                    'status': status
                })
        
        # Grade conditions
        for condition in conditions:
            level = condition.get('evidence_level', 'research')
            grades['conditions'].append({
                'condition': condition.get('condition', ''),
                'evidence_level': level,
                'label': self.get_evidence_label(level),
                'description': condition.get('description', '')
            })
        
        # Calculate overall score
        scores = [m.get('confidence', 0.0) for m in grades['markers'] if m.get('evidence_level') != 'unsupported']
        if scores:
            grades['score'] = round(sum(scores) / len(scores), 2)
            
            if grades['score'] >= 0.8:
                grades['overall'] = 'validated'
            elif grades['score'] >= 0.6:
                grades['overall'] = 'clinical'
            elif grades['score'] >= 0.4:
                grades['overall'] = 'research'
            elif grades['score'] >= 0.2:
                grades['overall'] = 'experimental'
        
        return grades
    
    def get_highest_evidence(self, levels: List[str]) -> str:
        """Get the highest evidence level from a list."""
        priority = ['validated', 'clinical', 'research', 'experimental']
        for level in priority:
            if level in levels:
                return level
        return 'unsupported'
    
    def get_evidence_summary(self, associations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of evidence levels."""
        summary = {
            'total': len(associations),
            'validated': 0,
            'clinical': 0,
            'research': 0,
            'experimental': 0,
            'unsupported': 0,
            'by_domain': {}
        }
        
        for assoc in associations:
            level = assoc.get('evidence_level', 'research')
            domain = assoc.get('domain', 'unknown')
            
            if level in summary:
                summary[level] += 1
            
            if domain not in summary['by_domain']:
                summary['by_domain'][domain] = {'validated': 0, 'clinical': 0, 'research': 0, 'experimental': 0}
            if level in summary['by_domain'][domain]:
                summary['by_domain'][domain][level] += 1
        
        return summary
    
    def should_show_association(self, evidence_level: str) -> bool:
        """Determine if an association should be shown to the user."""
        show_levels = ['validated', 'clinical', 'research', 'experimental']
        return evidence_level in show_levels
    
    def get_confidence_interval(self, score: float, evidence_level: str) -> Tuple[float, float]:
        """Get confidence interval based on evidence level."""
        # Confidence intervals based on evidence level
        intervals = {
            'validated': (0.85, 0.98),
            'clinical': (0.70, 0.90),
            'research': (0.50, 0.75),
            'experimental': (0.30, 0.60),
            'unsupported': (0.0, 0.30)
        }
        
        low, high = intervals.get(evidence_level, (0.0, 0.5))
        
        # Adjust based on score
        if score >= 0.8:
            return (max(low, score - 0.1), min(high, score + 0.1))
        elif score >= 0.6:
            return (max(low, score - 0.15), min(high, score + 0.15))
        else:
            return (low, min(high, score + 0.2))
    
    def generate_evidence_report(self, associations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive evidence report."""
        summary = self.get_evidence_summary(associations)
        score_data = self.calculate_evidence_score(associations)
        
        report = {
            'summary': summary,
            'score': score_data,
            'associations': []
        }
        
        for assoc in associations:
            level = assoc.get('evidence_level', 'research')
            confidence = assoc.get('confidence', 0.5)
            low, high = self.get_confidence_interval(confidence, level)
            
            report['associations'].append({
                'condition': assoc.get('condition', ''),
                'condition_id': assoc.get('condition_id', ''),
                'evidence_level': level,
                'evidence_label': self.get_evidence_label(level),
                'confidence': confidence,
                'confidence_interval': {'low': low, 'high': high},
                'description': assoc.get('description', ''),
                'confirmation_method': assoc.get('confirmation_method', ''),
                'show': self.should_show_association(level)
            })
        
        return report