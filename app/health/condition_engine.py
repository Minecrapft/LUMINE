"""
LUMINE Condition Engine
Associates marker patterns with health conditions.
"""

import logging
from typing import List, Dict, Any, Optional
from app.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class ConditionEngine:
    """Engine for health condition associations."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.condition_cache = {}
        self._load_conditions()
    
    def _load_conditions(self):
        """Load all conditions into cache."""
        try:
            # Get all conditions from database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM conditions')
                rows = cursor.fetchall()
                for row in rows:
                    condition = dict(row)
                    if condition.get('associated_markers'):
                        try:
                            import json
                            condition['associated_markers'] = json.loads(condition['associated_markers'])
                        except:
                            condition['associated_markers'] = []
                    self.condition_cache[condition['condition_id']] = condition
            logger.info(f"Loaded {len(self.condition_cache)} conditions")
        except Exception as e:
            logger.error(f"Failed to load conditions: {e}")
    
    def get_condition(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """Get condition by ID."""
        return self.condition_cache.get(condition_id)
    
    def get_conditions_by_marker(self, marker: str) -> List[Dict[str, Any]]:
        """Get conditions associated with a marker."""
        conditions = []
        for cond in self.condition_cache.values():
            if marker in cond.get('associated_markers', []):
                conditions.append(cond)
        return conditions
    
    def get_conditions_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get conditions by domain."""
        return [c for c in self.condition_cache.values() if c.get('domain') == domain]
    
    def find_associations(self, marker_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find health condition associations from marker results."""
        associations = []
        
        # Get flagged markers
        flagged = [r for r in marker_results if r.get('status') == 'attention']
        
        if not flagged:
            return associations
        
        # Find conditions matching flagged markers
        for marker_result in flagged:
            marker = marker_result.get('marker', '')
            conditions = self.get_conditions_by_marker(marker)
            
            for condition in conditions:
                # Check if all required markers are present
                required_markers = condition.get('associated_markers', [])
                present_markers = [r.get('marker') for r in flagged]
                
                # Check if all required markers are present
                if all(m in present_markers for m in required_markers):
                    # Check evidence level
                    evidence_level = condition.get('evidence_level', 'research')
                    
                    association = {
                        'condition_id': condition.get('condition_id'),
                        'condition': condition.get('condition'),
                        'domain': condition.get('domain'),
                        'evidence_level': evidence_level,
                        'evidence_label': self._get_evidence_label(evidence_level),
                        'description': condition.get('description', ''),
                        'limitations': condition.get('limitations', ''),
                        'confirmation_method': condition.get('confirmation_method', ''),
                        'associated_markers': required_markers,
                        'confidence': self._calculate_confidence(marker_results, required_markers)
                    }
                    associations.append(association)
        
        # Remove duplicates
        seen = set()
        unique_associations = []
        for assoc in associations:
            key = assoc.get('condition_id')
            if key not in seen:
                seen.add(key)
                unique_associations.append(assoc)
        
        return unique_associations
    
    def _get_evidence_label(self, evidence_level: str) -> str:
        """Get human-readable evidence label."""
        labels = {
            'validated': 'Validated Screening Association',
            'clinical': 'Clinical Association',
            'research': 'Research Association',
            'experimental': 'Experimental',
            'unsupported': 'Unsupported'
        }
        return labels.get(evidence_level, 'Unknown')
    
    def _calculate_confidence(self, marker_results: List[Dict[str, Any]], 
                            required_markers: List[str]) -> float:
        """Calculate confidence in association."""
        if not required_markers:
            return 0.0
        
        # Calculate based on match quality of markers
        total_confidence = 0.0
        marker_count = 0
        
        for marker in required_markers:
            for result in marker_results:
                if result.get('marker') == marker:
                    total_confidence += result.get('confidence', 0.5)
                    marker_count += 1
                    break
        
        if marker_count == 0:
            return 0.0
        
        return total_confidence / len(required_markers)
    
    def get_evidence_summary(self, associations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of evidence levels for associations."""
        summary = {
            'total': len(associations),
            'validated': 0,
            'clinical': 0,
            'research': 0,
            'experimental': 0,
            'unsupported': 0
        }
        
        for assoc in associations:
            level = assoc.get('evidence_level', 'research')
            if level in summary:
                summary[level] += 1
        
        return summary
    
    def get_recommendations(self, associations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on associations."""
        recommendations = []
        
        # Check if any validated associations
        has_validated = any(a.get('evidence_level') == 'validated' for a in associations)
        has_clinical = any(a.get('evidence_level') == 'clinical' for a in associations)
        has_research = any(a.get('evidence_level') == 'research' for a in associations)
        has_experimental = any(a.get('evidence_level') == 'experimental' for a in associations)
        
        if has_validated or has_clinical:
            recommendations.append("Consider discussing these results with a healthcare professional.")
            recommendations.append("Confirmatory laboratory testing may be appropriate.")
        
        if has_research:
            recommendations.append("Research associations have been identified. Clinical correlation is recommended.")
        
        if has_experimental:
            recommendations.append("These results are from experimental analysis. Professional evaluation is recommended for any clinical concern.")
        
        if associations:
            # Add specific recommendations from associations
            for assoc in associations:
                method = assoc.get('confirmation_method')
                if method and method not in recommendations:
                    recommendations.append(f"Confirmation: {method}")
        
        if not recommendations:
            recommendations = [
                "No significant associations identified.",
                "Regular screening is recommended for ongoing health monitoring."
            ]
        
        return recommendations
    
    def get_limitations(self, associations: List[Dict[str, Any]]) -> List[str]:
        """Get limitations based on associations."""
        limitations = [
            "This is a screening tool, not a diagnostic device.",
            "Results should be confirmed with laboratory testing.",
            "False positives and false negatives can occur.",
            "Multiple factors can affect test results."
        ]
        
        # Add specific limitations from associations
        for assoc in associations:
            if assoc.get('limitations'):
                limitations.append(assoc['limitations'])
        
        # Add experimental warnings if needed
        if any(a.get('evidence_level') in ['research', 'experimental'] for a in associations):
            limitations.append("Some associations are based on research evidence only.")
            limitations.append("Clinical validation may be required for clinical use.")
        
        return limitations