"""
LUMINE Data Models - COMPLETE
"""

from datetime import datetime
import sqlite3
import json
from typing import Dict, List, Optional, Any

class Patient:
    """Patient model with all fields - COMPLETE."""
    
    def __init__(self, data: Dict[str, Any]):
        self.patient_id = data.get('patient_id')
        self.first_name = data.get('first_name', '')
        self.last_name = data.get('last_name', '')
        self.date_of_birth = data.get('date_of_birth')
        self.gender = data.get('gender', '')
        self.email = data.get('email', '')
        self.phone = data.get('phone', '')
        self.address = data.get('address', '')
        self.medical_history = data.get('medical_history', '')
        self.allergies = data.get('allergies', '')
        self.emergency_contact = data.get('emergency_contact', '')
        self.notes = data.get('notes', '')
        self.consent_status = data.get('consent_status', False)
        self.created_at = data.get('created_at', datetime.now().isoformat())
        self.updated_at = data.get('updated_at', datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'patient_id': self.patient_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'date_of_birth': self.date_of_birth,
            'gender': self.gender,
            'email': self.email,
            'phone': self.phone,
            'address': self.address,
            'medical_history': self.medical_history,
            'allergies': self.allergies,
            'emergency_contact': self.emergency_contact,
            'notes': self.notes,
            'consent_status': self.consent_status,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()
    
    @property
    def age(self) -> Optional[int]:
        if not self.date_of_birth:
            return None
        try:
            birth = datetime.fromisoformat(self.date_of_birth)
            today = datetime.now()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except:
            return None

class Assessment:
    """Assessment model for screening results."""
    
    def __init__(self, data: Dict[str, Any]):
        self.assessment_id = data.get('assessment_id')
        self.patient_id = data.get('patient_id')
        self.date = data.get('date', datetime.now().isoformat())
        self.sample_type = data.get('sample_type', 'saliva')
        self.test_profile = data.get('test_profile', 'SALIVA-4K')
        self.overall_status = data.get('overall_status', 'pending')
        self.image_quality = data.get('image_quality', {})
        self.analysis_quality = data.get('analysis_quality', {})
        self.marker_results = data.get('marker_results', [])
        self.flagged_markers = data.get('flagged_markers', [])
        self.health_domains = data.get('health_domains', {})
        self.possible_associations = data.get('possible_associations', [])
        self.recommendations = data.get('recommendations', [])
        self.limitations = data.get('limitations', [])
        self.evidence_levels = data.get('evidence_levels', {})
        self.algorithm_version = data.get('algorithm_version', '1.0.0')
        self.calibration_version = data.get('calibration_version', '')
        self.image_path = data.get('image_path', '')
        self.notes = data.get('notes', '')
        self.created_at = data.get('created_at', datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'assessment_id': self.assessment_id,
            'patient_id': self.patient_id,
            'date': self.date,
            'sample_type': self.sample_type,
            'test_profile': self.test_profile,
            'overall_status': self.overall_status,
            'image_quality': self.image_quality,
            'analysis_quality': self.analysis_quality,
            'marker_results': self.marker_results,
            'flagged_markers': self.flagged_markers,
            'health_domains': self.health_domains,
            'possible_associations': self.possible_associations,
            'recommendations': self.recommendations,
            'limitations': self.limitations,
            'evidence_levels': self.evidence_levels,
            'algorithm_version': self.algorithm_version,
            'calibration_version': self.calibration_version,
            'image_path': self.image_path,
            'notes': self.notes,
            'created_at': self.created_at
        }
    
    @property
    def is_normal(self) -> bool:
        return self.overall_status == 'normal'
    
    @property
    def needs_followup(self) -> bool:
        return self.overall_status in ['attention', 'followup_recommended']
    
    @property
    def is_valid(self) -> bool:
        return self.overall_status != 'invalid'

class TestProfile:
    """Test strip profile definition."""
    
    def __init__(self, data: Dict[str, Any]):
        self.profile_id = data.get('profile_id')
        self.name = data.get('name', '')
        self.manufacturer = data.get('manufacturer', '')
        self.strip_type = data.get('strip_type', '')
        self.sample_type = data.get('sample_type', 'saliva')
        self.pad_count = data.get('pad_count', 10)
        self.reaction_time = data.get('reaction_time', 60)
        self.configuration = data.get('configuration', {})
        self.validation_status = data.get('validation_status', 'research')
        self.version = data.get('version', '1.0.0')
        self.created_at = data.get('created_at', datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'profile_id': self.profile_id,
            'name': self.name,
            'manufacturer': self.manufacturer,
            'strip_type': self.strip_type,
            'sample_type': self.sample_type,
            'pad_count': self.pad_count,
            'reaction_time': self.reaction_time,
            'configuration': self.configuration,
            'validation_status': self.validation_status,
            'version': self.version,
            'created_at': self.created_at
        }
    
    @property
    def is_validated(self) -> bool:
        return self.validation_status == 'validated'
    
    @property
    def is_experimental(self) -> bool:
        return self.validation_status in ['experimental', 'research']

class MarkerResult:
    """Individual marker result."""
    
    def __init__(self, data: Dict[str, Any]):
        self.marker = data.get('marker', '')
        self.raw_color = data.get('raw_color', {})
        self.normalized_color = data.get('normalized_color', {})
        self.result = data.get('result', '')
        self.match_quality = data.get('match_quality', '')
        self.reference_level = data.get('reference_level', '')
        self.status = data.get('status', 'normal')
        self.interpretation = data.get('interpretation', '')
        self.domain = data.get('domain', '')
        self.evidence_level = data.get('evidence_level', '')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'marker': self.marker,
            'raw_color': self.raw_color,
            'normalized_color': self.normalized_color,
            'result': self.result,
            'match_quality': self.match_quality,
            'reference_level': self.reference_level,
            'status': self.status,
            'interpretation': self.interpretation,
            'domain': self.domain,
            'evidence_level': self.evidence_level
        }
    
    @property
    def is_flagged(self) -> bool:
        return self.status in ['attention', 'abnormal']
    
    @property
    def is_normal(self) -> bool:
        return self.status == 'normal'

class ConditionAssociation:
    """Health condition association model."""
    
    def __init__(self, data: Dict[str, Any]):
        self.condition_id = data.get('condition_id')
        self.condition = data.get('condition', '')
        self.domain = data.get('domain', '')
        self.associated_markers = data.get('associated_markers', [])
        self.evidence_level = data.get('evidence_level', 'research')
        self.description = data.get('description', '')
        self.limitations = data.get('limitations', '')
        self.confirmation_method = data.get('confirmation_method', '')
        self.research_references = data.get('research_references', '')
        self.created_at = data.get('created_at', datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'condition_id': self.condition_id,
            'condition': self.condition,
            'domain': self.domain,
            'associated_markers': self.associated_markers,
            'evidence_level': self.evidence_level,
            'description': self.description,
            'limitations': self.limitations,
            'confirmation_method': self.confirmation_method,
            'research_references': self.research_references,
            'created_at': self.created_at
        }
    
    @property
    def evidence_label(self) -> str:
        labels = {
            'validated': 'Validated Screening Association',
            'clinical': 'Clinical Association',
            'research': 'Research Association',
            'experimental': 'Experimental',
            'unsupported': 'Unsupported'
        }
        return labels.get(self.evidence_level, 'Unknown')
