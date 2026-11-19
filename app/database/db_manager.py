"""
LUMINE Database Manager — with chart-based reference colour seeding.
"""

import sqlite3
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)

URS_4K_MARKERS = ["glucose", "pH", "protein", "ketones"]


class TrainingCapacityError(RuntimeError):
    """Raised when the configured training-job limit has been reached."""


# ---------------------------------------------------------------------------
# Reference colour charts (imported here, used in _seed_reference_colors)
# ---------------------------------------------------------------------------
try:
    from app.ml.chart_colors import PROFILE_CHARTS
    _CHARTS_AVAILABLE = True
except ImportError:
    _CHARTS_AVAILABLE = False
    PROFILE_CHARTS = {}

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create tables
                cursor.executescript('''
                    CREATE TABLE IF NOT EXISTS patients (
                        patient_id TEXT PRIMARY KEY,
                        first_name TEXT NOT NULL,
                        last_name TEXT NOT NULL,
                        date_of_birth TEXT,
                        gender TEXT,
                        email TEXT,
                        phone TEXT,
                        address TEXT,
                        medical_history TEXT,
                        allergies TEXT,
                        emergency_contact TEXT,
                        notes TEXT,
                        consent_status INTEGER DEFAULT 0,
                        created_at TEXT,
                        updated_at TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS assessments (
                        assessment_id TEXT PRIMARY KEY,
                        patient_id TEXT NOT NULL,
                        date TEXT NOT NULL,
                        sample_type TEXT NOT NULL,
                        test_profile TEXT NOT NULL,
                        overall_status TEXT NOT NULL,
                        image_quality TEXT,
                        analysis_quality TEXT,
                        marker_results TEXT,
                        flagged_markers TEXT,
                        health_domains TEXT,
                        possible_associations TEXT,
                        recommendations TEXT,
                        limitations TEXT,
                        evidence_levels TEXT,
                        algorithm_version TEXT,
                        calibration_version TEXT,
                        image_path TEXT,
                        notes TEXT,
                        created_at TEXT,
                        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS test_profiles (
                        profile_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        manufacturer TEXT,
                        strip_type TEXT,
                        sample_type TEXT NOT NULL,
                        pad_count INTEGER DEFAULT 10,
                        reaction_time INTEGER DEFAULT 60,
                        configuration TEXT,
                        validation_status TEXT,
                        version TEXT,
                        created_at TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS reference_colors (
                        color_id TEXT PRIMARY KEY,
                        profile_id TEXT,
                        marker TEXT NOT NULL,
                        level TEXT NOT NULL,
                        rgb TEXT,
                        hsv TEXT,
                        lab TEXT,
                        batch TEXT,
                        calibration_version TEXT,
                        created_at TEXT,
                        FOREIGN KEY (profile_id) REFERENCES test_profiles(profile_id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS conditions (
                        condition_id TEXT PRIMARY KEY,
                        condition TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        associated_markers TEXT,
                        evidence_level TEXT NOT NULL,
                        sample_type TEXT NOT NULL DEFAULT 'urine',
                        description TEXT,
                        limitations TEXT,
                        confirmation_method TEXT,
                        research_references TEXT,
                        created_at TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS calibrations (
                        calibration_id TEXT PRIMARY KEY,
                        operator TEXT,
                        camera_settings TEXT,
                        lighting_reading TEXT,
                        reference_values TEXT,
                        timestamp TEXT,
                        status TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS audit_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        operator TEXT,
                        action TEXT NOT NULL,
                        details TEXT,
                        assessment_id TEXT
                    );

                    -- ── Phase 4: Dataset management ──────────────────────────
                    CREATE TABLE IF NOT EXISTS training_datasets (
                        dataset_id   TEXT PRIMARY KEY,
                        name         TEXT NOT NULL,
                        description  TEXT,
                        strip_type   TEXT NOT NULL,
                        created_by   TEXT DEFAULT 'system',
                        created_at   TEXT NOT NULL,
                        updated_at   TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS training_samples (
                        sample_id      TEXT PRIMARY KEY,
                        dataset_id     TEXT NOT NULL,
                        image_path     TEXT NOT NULL,
                        strip_type     TEXT NOT NULL,
                        capture_meta   TEXT,           -- JSON: phone, lighting, date
                        split          TEXT DEFAULT 'unassigned', -- train/val/test/unassigned
                        is_synthetic   INTEGER DEFAULT 0,
                        created_at     TEXT NOT NULL,
                        FOREIGN KEY (dataset_id) REFERENCES training_datasets(dataset_id)
                    );

                    CREATE TABLE IF NOT EXISTS training_pad_labels (
                        label_id       TEXT PRIMARY KEY,
                        sample_id      TEXT NOT NULL,
                        marker         TEXT NOT NULL,
                        pad_index      INTEGER NOT NULL,
                        true_level     TEXT NOT NULL,
                        measured_value TEXT,           -- optional numeric (e.g. "7.0" for pH)
                        label_source   TEXT DEFAULT 'visual', -- visual | standard_solution | chart
                        annotator      TEXT DEFAULT 'unknown',
                        pad_color      TEXT,           -- JSON [R,G,B] from detection
                        white_ref_rgb  TEXT,           -- JSON [R,G,B] white reference at capture
                        created_at     TEXT NOT NULL,
                        FOREIGN KEY (sample_id) REFERENCES training_samples(sample_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_samples_dataset ON training_samples(dataset_id);
                    CREATE INDEX IF NOT EXISTS idx_labels_sample   ON training_pad_labels(sample_id);
                    CREATE INDEX IF NOT EXISTS idx_labels_marker   ON training_pad_labels(marker);

                    -- ── Phase 5: ML Models & Training Jobs ─────────────────
                    CREATE TABLE IF NOT EXISTS ml_models (
                        model_id        TEXT PRIMARY KEY,
                        strip_type      TEXT NOT NULL,
                        marker          TEXT NOT NULL,
                        version         INTEGER NOT NULL,
                        algorithm       TEXT NOT NULL,
                        feature_version INTEGER NOT NULL,
                        dataset_ids     TEXT,
                        metrics         TEXT NOT NULL,
                        baseline_metrics TEXT,
                        model_path      TEXT NOT NULL,
                        sha256_hash     TEXT NOT NULL,
                        is_active       INTEGER DEFAULT 0,
                        created_by      TEXT DEFAULT 'system',
                        created_at      TEXT NOT NULL,
                        activated_at    TEXT
                    );

                    CREATE TABLE IF NOT EXISTS training_jobs (
                        job_id          TEXT PRIMARY KEY,
                        strip_type      TEXT NOT NULL,
                        marker          TEXT NOT NULL,
                        algorithm       TEXT NOT NULL,
                        status          TEXT DEFAULT 'pending',
                        progress        INTEGER DEFAULT 0,
                        log_output      TEXT DEFAULT '',
                        error_message   TEXT,
                        model_id        TEXT,
                        created_at      TEXT NOT NULL,
                        updated_at      TEXT NOT NULL,
                        completed_at    TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_ml_models_strip_marker ON ml_models(strip_type, marker);
                    CREATE INDEX IF NOT EXISTS idx_ml_models_active ON ml_models(is_active);
                    CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
                    -- ─────────────────────────────────────────────────────────

                    CREATE INDEX IF NOT EXISTS idx_assessments_patient ON assessments(patient_id);
                    CREATE INDEX IF NOT EXISTS idx_assessments_date ON assessments(date);
                    CREATE INDEX IF NOT EXISTS idx_patients_name ON patients(last_name, first_name);
                ''')
                
                # Add missing columns if not exists
                try:
                    cursor.execute("ALTER TABLE patients ADD COLUMN emergency_contact TEXT")
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute("ALTER TABLE patients ADD COLUMN notes TEXT")
                except sqlite3.OperationalError:
                    pass

                training_job_columns = {
                    row["name"] for row in cursor.execute(
                        "PRAGMA table_info(training_jobs)"
                    ).fetchall()
                }
                if "process_id" not in training_job_columns:
                    cursor.execute(
                        "ALTER TABLE training_jobs ADD COLUMN process_id INTEGER"
                    )

                condition_columns = {
                    row["name"] for row in cursor.execute(
                        "PRAGMA table_info(conditions)"
                    ).fetchall()
                }
                if "sample_type" not in condition_columns:
                    cursor.execute(
                        "ALTER TABLE conditions ADD COLUMN sample_type "
                        "TEXT NOT NULL DEFAULT 'urine'"
                    )

                cursor.execute("""
                    UPDATE conditions
                    SET sample_type = 'saliva'
                    WHERE condition_id IN ('SC001', 'SC002', 'SC003', 'SC004', 'SC005')
                       OR lower(condition) LIKE '%saliva%'
                       OR lower(condition) LIKE '%salivary%'
                """)
                
                # Add SALIVA test profile
                cursor.execute('SELECT COUNT(*) FROM test_profiles WHERE profile_id = ?', ('SALIVA-4K',))
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO test_profiles (
                            profile_id, name, manufacturer, strip_type, sample_type,
                            pad_count, reaction_time, configuration, validation_status, version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        'SALIVA-4K', 'SALIVA-4K Saliva Strip', 'Research', 'saliva_strip', 'saliva',
                        10, 120, json.dumps({
                            'markers': [
                                'glucose', 'protein', 'ketones', 'pH',
                                'blood', 'nitrite', 'leukocytes', 'specific_gravity',
                                'bilirubin', 'urobilinogen'
                            ]
                        }), 'experimental', '1.0.0', datetime.now().isoformat()
                    ))

                cursor.execute(
                    'SELECT COUNT(*) FROM test_profiles WHERE profile_id = ?',
                    ('SALIVA-PH4',)
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO test_profiles (
                            profile_id, name, manufacturer, strip_type, sample_type,
                            pad_count, reaction_time, configuration, validation_status, version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        'SALIVA-PH4', 'Universal pH 0-14 Saliva Strip (4-pad panel)',
                        'User-supplied chart', 'ph_strip', 'saliva', 4, 0,
                        json.dumps({
                            'markers': ['pH'],
                            'reading_method': 'four_pad_combined'
                        }),
                        'experimental', '1.0.0', datetime.now().isoformat()
                    ))
                else:
                    cursor.execute('''
                        UPDATE test_profiles
                        SET name = ?, manufacturer = ?, strip_type = ?, sample_type = ?,
                            pad_count = ?, reaction_time = ?, configuration = ?,
                            validation_status = ?
                        WHERE profile_id = ?
                    ''', (
                        'Universal pH 0-14 Saliva Strip (4-pad panel)',
                        'User-supplied chart', 'ph_strip', 'saliva', 4, 0,
                        json.dumps({
                            'markers': ['pH'],
                            'reading_method': 'four_pad_combined'
                        }),
                        'experimental', 'SALIVA-PH4'
                    ))
                
                # URS-4K is the four-pad urine strip used by this installation.
                cursor.execute('SELECT COUNT(*) FROM test_profiles WHERE profile_id = ?', ('URS-4K',))
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO test_profiles (
                            profile_id, name, manufacturer, strip_type, sample_type,
                            pad_count, reaction_time, configuration, validation_status, version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        'URS-4K', 'URS-4K Urine Strip (4 parameters)', 'Generic',
                        'urine_strip', 'urine', len(URS_4K_MARKERS), 60,
                        json.dumps({'markers': URS_4K_MARKERS}),
                        'experimental', '1.0.0', datetime.now().isoformat()
                    ))

                # Upgrade existing databases that previously treated URS-4K as ten-pad.
                cursor.execute('''
                    UPDATE test_profiles
                    SET name = ?, pad_count = ?, configuration = ?, validation_status = ?
                    WHERE profile_id = ?
                ''', (
                    'URS-4K Urine Strip (4 parameters)',
                    len(URS_4K_MARKERS),
                    json.dumps({'markers': URS_4K_MARKERS}),
                    'experimental',
                    'URS-4K'
                ))

                # URS-10T retains the ten-marker configuration for compatibility.
                cursor.execute('SELECT COUNT(*) FROM test_profiles WHERE profile_id = ?', ('URS-10T',))
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO test_profiles (
                            profile_id, name, manufacturer, strip_type, sample_type,
                            pad_count, reaction_time, configuration, validation_status, version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        'URS-10T', 'URS-10T Urine Strip', 'Generic', 'urine_strip', 'urine',
                        10, 60, json.dumps({
                            'markers': [
                                'glucose', 'bilirubin', 'ketones', 'specific_gravity',
                                'blood', 'pH', 'protein', 'urobilinogen', 'nitrite', 'leukocytes'
                            ]
                        }), 'validated', '1.0.0', datetime.now().isoformat()
                    ))

                # Legacy four-marker profile ID retained for existing assessments.
                cursor.execute('SELECT COUNT(*) FROM test_profiles WHERE profile_id = ?', ('URS-4K-4P',))
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO test_profiles (
                            profile_id, name, manufacturer, strip_type, sample_type,
                            pad_count, reaction_time, configuration, validation_status, version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        'URS-4K-4P', '4-Parameter Urine Strip (Research)',
                        'Generic / user-specified', 'urine_strip', 'urine',
                        len(URS_4K_MARKERS), 60,
                        json.dumps({'markers': URS_4K_MARKERS}),
                        'experimental', '1.0.0', datetime.now().isoformat()
                    ))
                
                # Seed reference colours for every chart-backed profile.
                self._seed_reference_colors_from_chart(cursor)
                
                # Add conditions
                cursor.execute('SELECT COUNT(*) FROM conditions WHERE condition_id = ?', ('SC001',))
                if cursor.fetchone()[0] == 0:
                    conditions = [
                        ('SC001', 'salivary_glucose_abnormal', 'metabolic', '["glucose"]', 'research', 'saliva',
                         'Glucose detected above normal saliva reference range', 
                         'May indicate metabolic changes - RESEARCH USE ONLY',
                         'Blood glucose test, HbA1c', 'Research: Salivary glucose as biomarker'),
                        ('SC002', 'salivary_protein_abnormal', 'renal', '["protein"]', 'research', 'saliva',
                         'Protein detected in saliva above normal reference',
                         'May indicate oral or systemic changes - RESEARCH USE ONLY',
                         'Urine protein test, kidney function tests', 'Research: Salivary protein biomarkers'),
                        ('SC003', 'salivary_ketosis', 'metabolic', '["ketones"]', 'research', 'saliva',
                         'Ketones detected in saliva',
                         'May indicate ketosis or metabolic changes - RESEARCH USE ONLY',
                         'Blood ketones, medical evaluation', 'Research: Salivary ketone detection'),
                        ('SC004', 'salivary_inflammation', 'inflammation', '["leukocytes"]', 'research', 'saliva',
                         'Leukocytes detected in saliva',
                         'May indicate oral inflammation or infection - RESEARCH USE ONLY',
                         'Oral examination, medical evaluation', 'Research: Salivary leukocyte markers'),
                        ('SC005', 'salivary_pH_abnormal', 'acid_base', '["pH"]', 'research', 'saliva',
                         'Salivary pH outside normal reference range (6.0-7.5)',
                         'May indicate acid-base changes - RESEARCH USE ONLY',
                         'Blood pH, medical evaluation', 'Research: Salivary pH as biomarker'),
                    ]
                    
                    for cond in conditions:
                        cursor.execute('''
                            INSERT INTO conditions (
                                condition_id, condition, domain, associated_markers,
                                evidence_level, sample_type, description, limitations,
                                confirmation_method, research_references, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (*cond, datetime.now().isoformat()))
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Reference colour seeding — idempotent, called from _init_database
    # ------------------------------------------------------------------

    def _seed_reference_colors_from_chart(self, cursor) -> None:
        """
        Insert reference colours for all profiles defined in PROFILE_CHARTS.

        The operation is idempotent: existing (profile_id, marker, level)
        combinations are left untouched (no duplicates).
        """
        if not _CHARTS_AVAILABLE:
            logger.warning("chart_colors not available — skipping reference colour seeding")
            return

        # Map profile IDs to chart keys
        profile_chart_map = {
            'URS-4K':    'URS-4K',
            'URS-10T':   'URS-10T',
            'URS-4K-4P': 'URS-4K-4P',
            'SALIVA-4K': 'SALIVA-4K',
            'SALIVA-PH4': 'SALIVA-PH4',
        }

        for profile_id, chart_key in profile_chart_map.items():
            chart = PROFILE_CHARTS.get(chart_key, {})
            if not chart:
                logger.warning(f"No chart data found for {chart_key}")
                continue

            for marker, levels in chart.items():
                for level_name, rgb in levels:
                    # Normalise rgb: accept list or tuple of ints in 0–255
                    rgb_list = [int(max(0, min(255, v))) for v in rgb]

                    # Check for existing entry
                    cursor.execute(
                        'SELECT 1 FROM reference_colors '
                        'WHERE profile_id = ? AND marker = ? AND level = ?',
                        (profile_id, marker, level_name)
                    )
                    if cursor.fetchone():
                        if profile_id in {'URS-4K', 'URS-4K-4P'}:
                            cursor.execute(
                                'UPDATE reference_colors SET rgb = ? '
                                'WHERE profile_id = ? AND marker = ? AND level = ?',
                                (json.dumps(rgb_list), profile_id, marker, level_name)
                            )
                        continue  # already present — skip

                    color_id = f"RC_{profile_id}_{marker}_{level_name}_{uuid.uuid4().hex[:8]}"
                    cursor.execute(
                        '''
                        INSERT INTO reference_colors
                            (color_id, profile_id, marker, level, rgb, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            color_id,
                            profile_id,
                            marker,
                            level_name,
                            json.dumps(rgb_list),
                            datetime.now().isoformat(),
                        )
                    )

        logger.info("Reference colour seeding complete")

    def seed_reference_colors_idempotent(self) -> None:
        """Public method to re-seed reference colors at any time (e.g. after adding a new chart)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            self._seed_reference_colors_from_chart(cursor)
            conn.commit()
    
    # ========== PATIENT OPERATIONS ==========
    
    def create_patient(self, patient_data: Dict[str, Any]) -> str:
        patient_id = f"P{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO patients (
                    patient_id, first_name, last_name, date_of_birth, gender,
                    email, phone, address, medical_history, allergies,
                    emergency_contact, notes, consent_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_id,
                patient_data.get('first_name', ''),
                patient_data.get('last_name', ''),
                patient_data.get('date_of_birth'),
                patient_data.get('gender', ''),
                patient_data.get('email', ''),
                patient_data.get('phone', ''),
                patient_data.get('address', ''),
                patient_data.get('medical_history', ''),
                patient_data.get('allergies', ''),
                patient_data.get('emergency_contact', ''),
                patient_data.get('notes', ''),
                1 if patient_data.get('consent_status', False) else 0,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            conn.commit()
            return patient_id
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def search_patients(self, query: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM patients 
                WHERE first_name LIKE ? OR last_name LIKE ? OR patient_id LIKE ?
                ORDER BY last_name, first_name
            ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_patients(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM patients ORDER BY last_name, first_name')
            return [dict(row) for row in cursor.fetchall()]
    
    def update_patient(self, patient_id: str, data: Dict[str, Any]) -> bool:
        allowed_fields = ['first_name', 'last_name', 'date_of_birth', 'gender',
                         'email', 'phone', 'address', 'medical_history', 
                         'allergies', 'emergency_contact', 'notes', 'consent_status']
        
        updates = []
        values = []
        for key in allowed_fields:
            if key in data:
                updates.append(f'{key} = ?')
                values.append(data[key])
        
        if not updates:
            return False
        
        values.append(datetime.now().isoformat())
        values.append(patient_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE patients 
                SET {', '.join(updates)}, updated_at = ?
                WHERE patient_id = ?
            ''', values)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_patient(self, patient_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM patients WHERE patient_id = ?', (patient_id,))
            if cursor.fetchone() is None:
                return False
            cursor.execute('''
                DELETE FROM audit_log
                WHERE assessment_id IN (
                    SELECT assessment_id FROM assessments WHERE patient_id = ?
                )
            ''', (patient_id,))
            cursor.execute('DELETE FROM assessments WHERE patient_id = ?', (patient_id,))
            cursor.execute('DELETE FROM patients WHERE patient_id = ?', (patient_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def delete_assessment(self, patient_id: str, assessment_id: str) -> bool:
        """Delete one assessment and its audit entries for the specified patient."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 1 FROM assessments
                WHERE assessment_id = ? AND patient_id = ?
            ''', (assessment_id, patient_id))
            if cursor.fetchone() is None:
                return False
            cursor.execute(
                'DELETE FROM audit_log WHERE assessment_id = ?',
                (assessment_id,)
            )
            cursor.execute(
                'DELETE FROM assessments WHERE assessment_id = ? AND patient_id = ?',
                (assessment_id, patient_id)
            )
            deleted = cursor.rowcount == 1
            conn.commit()
            return deleted
    
    # ========== ASSESSMENT OPERATIONS ==========
    
    def create_assessment(self, assessment_data: Dict[str, Any]) -> str:
        assessment_id = f"A{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO assessments (
                    assessment_id, patient_id, date, sample_type, test_profile,
                    overall_status, image_quality, analysis_quality, marker_results,
                    flagged_markers, health_domains, possible_associations,
                    recommendations, limitations, evidence_levels,
                    algorithm_version, calibration_version, image_path, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                assessment_id,
                assessment_data.get('patient_id', ''),
                assessment_data.get('date', datetime.now().isoformat()),
                assessment_data.get('sample_type', 'saliva'),
                assessment_data.get('test_profile', 'SALIVA-4K'),
                assessment_data.get('overall_status', 'pending'),
                json.dumps(assessment_data.get('image_quality', {})),
                json.dumps(assessment_data.get('analysis_quality', {})),
                json.dumps(assessment_data.get('marker_results', [])),
                json.dumps(assessment_data.get('flagged_markers', [])),
                json.dumps(assessment_data.get('health_domains', {})),
                json.dumps(assessment_data.get('possible_associations', [])),
                json.dumps(assessment_data.get('recommendations', [])),
                json.dumps(assessment_data.get('limitations', [])),
                json.dumps(assessment_data.get('evidence_levels', {})),
                assessment_data.get('algorithm_version', '1.0.0'),
                assessment_data.get('calibration_version', ''),
                assessment_data.get('image_path', ''),
                assessment_data.get('notes', ''),
                datetime.now().isoformat()
            ))
            conn.commit()
            return assessment_id
    
    def get_assessment(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM assessments WHERE assessment_id = ?', (assessment_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                for field in ['image_quality', 'analysis_quality', 'marker_results',
                             'flagged_markers', 'health_domains', 'possible_associations',
                             'recommendations', 'limitations', 'evidence_levels']:
                    if result.get(field):
                        try:
                            result[field] = json.loads(result[field])
                        except:
                            result[field] = {}
                return result
            return None
    
    def get_patient_assessments(self, patient_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM assessments 
                WHERE patient_id = ?
                ORDER BY date DESC
            ''', (patient_id,))
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                for field in ['image_quality', 'analysis_quality', 'marker_results',
                             'flagged_markers', 'health_domains', 'possible_associations',
                             'recommendations', 'limitations', 'evidence_levels']:
                    if result.get(field):
                        try:
                            result[field] = json.loads(result[field])
                        except:
                            result[field] = {}
                results.append(result)
            return results
    
    def get_recent_assessments(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.*, p.first_name, p.last_name 
                FROM assessments a
                JOIN patients p ON a.patient_id = p.patient_id
                ORDER BY a.date DESC
                LIMIT ?
            ''', (limit,))
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                for field in ['image_quality', 'analysis_quality', 'marker_results',
                             'flagged_markers', 'health_domains', 'possible_associations',
                             'recommendations', 'limitations', 'evidence_levels']:
                    if result.get(field):
                        try:
                            result[field] = json.loads(result[field])
                        except:
                            result[field] = {}
                results.append(result)
            return results
    
    def get_test_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM test_profiles WHERE profile_id = ?', (profile_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('configuration'):
                    try:
                        result['configuration'] = json.loads(result['configuration'])
                    except:
                        result['configuration'] = {}
                return result
            return None
    
    def get_all_test_profiles(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM test_profiles ORDER BY name')
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('configuration'):
                    try:
                        result['configuration'] = json.loads(result['configuration'])
                    except:
                        result['configuration'] = {}
                results.append(result)
            return results
    
    def get_reference_colors(self, profile_id: str, marker: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM reference_colors 
                WHERE profile_id = ? AND marker = ?
                ORDER BY level
            ''', (profile_id, marker))
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('rgb'):
                    try:
                        result['rgb'] = json.loads(result['rgb'])
                    except:
                        result['rgb'] = [128, 128, 128]
                results.append(result)
            return results
    
    def log_action(self, operator: str, action: str, details: str = '', assessment_id: str = ''):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_log (timestamp, operator, action, details, assessment_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), operator, action, details, assessment_id))
            conn.commit()
    
    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM audit_log 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM patients')
            total_patients = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*)
                FROM assessments a
                INNER JOIN patients p ON p.patient_id = a.patient_id
            ''')
            total_assessments = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*)
                FROM assessments a
                INNER JOIN patients p ON p.patient_id = a.patient_id
                WHERE a.overall_status IN (?, ?)
            ''', ('attention', 'followup_recommended'))
            followup_count = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT DATE(a.date) as day, COUNT(*) as count 
                FROM assessments a
                INNER JOIN patients p ON p.patient_id = a.patient_id
                WHERE a.date >= DATE('now', '-7 days')
                GROUP BY day
                ORDER BY day
            ''')
            recent_activity = [dict(row) for row in cursor.fetchall()]
            
            cursor.execute('''
                SELECT a.overall_status, COUNT(*) as count 
                FROM assessments a
                INNER JOIN patients p ON p.patient_id = a.patient_id
                GROUP BY a.overall_status
            ''')
            status_counts = {row['overall_status']: row['count'] for row in cursor.fetchall()}
            
            return {
                'total_patients': total_patients,
                'total_assessments': total_assessments,
                'followup_needed': followup_count,
                'recent_screening_count': sum(
                    activity['count'] for activity in recent_activity
                ),
                'recent_activity': recent_activity,
                'status_counts': status_counts
            }

    # ========== PHASE 4: DATASET MANAGEMENT ==========

    def create_dataset(self, name: str, strip_type: str, description: str = '',
                       created_by: str = 'system') -> str:
        """Create a new training dataset. Returns dataset_id."""
        dataset_id = f"DS_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6].upper()}"
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_datasets
                    (dataset_id, name, description, strip_type, created_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (dataset_id, name, description, strip_type, created_by, now, now))
            conn.commit()
        return dataset_id

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM training_datasets WHERE dataset_id = ?', (dataset_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.*,
                       COUNT(DISTINCT s.sample_id)  AS sample_count,
                       COUNT(DISTINCT l.label_id)   AS label_count
                FROM training_datasets d
                LEFT JOIN training_samples    s ON s.dataset_id = d.dataset_id
                LEFT JOIN training_pad_labels l ON l.sample_id  = s.sample_id
                GROUP BY d.dataset_id
                ORDER BY d.created_at DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset and all its samples + labels (cascade)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # labels first (FK)
            cursor.execute('''
                DELETE FROM training_pad_labels
                WHERE sample_id IN (
                    SELECT sample_id FROM training_samples WHERE dataset_id = ?
                )
            ''', (dataset_id,))
            cursor.execute('DELETE FROM training_samples WHERE dataset_id = ?', (dataset_id,))
            cursor.execute('DELETE FROM training_datasets WHERE dataset_id = ?', (dataset_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ── Sample operations ─────────────────────────────────────────────────────

    def add_sample(self, dataset_id: str, image_path: str, strip_type: str,
                   capture_meta: Optional[Dict] = None, is_synthetic: bool = False,
                   split: str = 'unassigned') -> str:
        """Add a photo sample to a dataset. Returns sample_id."""
        sample_id = f"SMP_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6].upper()}"
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_samples
                    (sample_id, dataset_id, image_path, strip_type,
                     capture_meta, split, is_synthetic, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sample_id, dataset_id, image_path, strip_type,
                json.dumps(capture_meta or {}), split, 1 if is_synthetic else 0, now
            ))
            # touch dataset updated_at
            cursor.execute('UPDATE training_datasets SET updated_at = ? WHERE dataset_id = ?',
                           (now, dataset_id))
            conn.commit()
        return sample_id

    def get_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM training_samples WHERE sample_id = ?', (sample_id,))
            row = cursor.fetchone()
            if not row:
                return None
            result = dict(row)
            if result.get('capture_meta'):
                try:
                    result['capture_meta'] = json.loads(result['capture_meta'])
                except Exception:
                    result['capture_meta'] = {}
            return result

    def list_samples(self, dataset_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.*,
                       COUNT(l.label_id) AS label_count
                FROM training_samples s
                LEFT JOIN training_pad_labels l ON l.sample_id = s.sample_id
                WHERE s.dataset_id = ?
                GROUP BY s.sample_id
                ORDER BY s.created_at DESC
            ''', (dataset_id,))
            results = []
            for row in cursor.fetchall():
                r = dict(row)
                if r.get('capture_meta'):
                    try:
                        r['capture_meta'] = json.loads(r['capture_meta'])
                    except Exception:
                        r['capture_meta'] = {}
                results.append(r)
            return results

    def delete_sample(self, sample_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM training_pad_labels WHERE sample_id = ?', (sample_id,))
            cursor.execute('DELETE FROM training_samples WHERE sample_id = ?', (sample_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ── Label operations ──────────────────────────────────────────────────────

    def save_pad_labels(self, sample_id: str, labels: List[Dict[str, Any]],
                        annotator: str = 'unknown') -> int:
        """
        Upsert pad labels for a sample.
        Each dict in labels: {marker, pad_index, true_level, measured_value?, label_source?}
        Returns count of labels saved.
        """
        now = datetime.now().isoformat()
        saved = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for lbl in labels:
                marker    = lbl.get('marker', '')
                pad_index = int(lbl.get('pad_index', 0))
                true_level    = lbl.get('true_level', '')
                measured_value = lbl.get('measured_value')
                label_source   = lbl.get('label_source', 'visual')
                if not (marker and true_level):
                    continue
                # Delete old label for same (sample, marker, pad_index)
                cursor.execute('''
                    DELETE FROM training_pad_labels
                    WHERE sample_id = ? AND marker = ? AND pad_index = ?
                ''', (sample_id, marker, pad_index))
                label_id = f"LBL_{uuid.uuid4().hex[:12]}"
                pad_color     = lbl.get('pad_color')     # JSON string or list; may be None
                white_ref_rgb = lbl.get('white_ref_rgb') # JSON string or list; may be None
                if isinstance(pad_color, list):
                    import json as _json
                    pad_color = _json.dumps(pad_color)
                if isinstance(white_ref_rgb, list):
                    import json as _json
                    white_ref_rgb = _json.dumps(white_ref_rgb)
                cursor.execute('''
                    INSERT INTO training_pad_labels
                        (label_id, sample_id, marker, pad_index, true_level,
                         measured_value, label_source, annotator,
                         pad_color, white_ref_rgb, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (label_id, sample_id, marker, pad_index, true_level,
                      measured_value, label_source, annotator,
                      pad_color, white_ref_rgb, now))
                saved += 1
            conn.commit()
        return saved

    def get_pad_labels(self, sample_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM training_pad_labels WHERE sample_id = ?
                ORDER BY pad_index
            ''', (sample_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ── Export helpers ────────────────────────────────────────────────────────

    def export_dataset_csv(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Return all labelled pads for a dataset as a flat list of dicts
        suitable for CSV serialisation (one row per pad label).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    s.sample_id, s.image_path, s.strip_type, s.split, s.is_synthetic,
                    l.marker, l.pad_index, l.true_level, l.measured_value,
                    l.label_source, l.annotator, l.pad_color, l.white_ref_rgb,
                    l.created_at AS labeled_at
                FROM training_samples s
                JOIN training_pad_labels l ON l.sample_id = s.sample_id
                WHERE s.dataset_id = ?
                ORDER BY s.sample_id, l.pad_index
            ''', (dataset_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ── Group split ───────────────────────────────────────────────────────────

    def assign_group_split(self, dataset_id: str,
                            train_frac: float = 0.7,
                            val_frac: float = 0.15) -> Dict[str, int]:
        """
        Assign train/val/test split to samples by photo (group split).
        Shuffles samples deterministically (sorted by sample_id).
        No photo appears in more than one split.
        Returns counts: {'train': N, 'val': N, 'test': N}
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT sample_id FROM training_samples WHERE dataset_id = ? ORDER BY sample_id',
                (dataset_id,)
            )
            sample_ids = [row['sample_id'] for row in cursor.fetchall()]

        n = len(sample_ids)
        if n == 0:
            return {'train': 0, 'val': 0, 'test': 0}

        n_train = max(1, round(n * train_frac))
        n_val   = max(0, round(n * val_frac))
        n_test  = max(0, n - n_train - n_val)

        splits = (
            ['train'] * n_train +
            ['val']   * n_val   +
            ['test']  * n_test
        )
        # Pad/trim to exact length
        splits = (splits + ['test'] * n)[:n]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            for sid, split in zip(sample_ids, splits):
                cursor.execute(
                    'UPDATE training_samples SET split = ? WHERE sample_id = ?',
                    (split, sid)
                )
            conn.commit()

        counts: Dict[str, int] = {'train': 0, 'val': 0, 'test': 0}
        for s in splits:
            counts[s] = counts.get(s, 0) + 1
        return counts

    # ── Phase 5: Model Registry ───────────────────────────────────────────────

    def register_model(
        self,
        strip_type: str,
        marker: str,
        algorithm: str,
        feature_version: int,
        metrics: Dict[str, Any],
        model_path: str,
        sha256_hash: str,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        dataset_ids: Optional[List[str]] = None,
        created_by: str = 'system',
        is_active: bool = False
    ) -> str:
        """Register a newly trained model artifact in ml_models table."""
        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Determine next version number for this (strip_type, marker)
            cursor.execute(
                'SELECT COALESCE(MAX(version), 0) + 1 FROM ml_models WHERE strip_type = ? AND marker = ?',
                (strip_type, marker)
            )
            next_version = cursor.fetchone()[0]

            model_id = f"MDL_{uuid.uuid4().hex[:10].upper()}"
            cursor.execute('''
                INSERT INTO ml_models (
                    model_id, strip_type, marker, version, algorithm,
                    feature_version, dataset_ids, metrics, baseline_metrics,
                    model_path, sha256_hash, is_active, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                strip_type,
                marker,
                next_version,
                algorithm,
                feature_version,
                json.dumps(dataset_ids or []),
                json.dumps(metrics),
                json.dumps(baseline_metrics or {}),
                model_path,
                sha256_hash,
                1 if is_active else 0,
                created_by,
                now
            ))
            conn.commit()
            logger.info(f"Registered model {model_id} ({strip_type}/{marker} v{next_version})")
            return model_id

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single model by ID, deserializing JSON fields."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM ml_models WHERE model_id = ?', (model_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d['dataset_ids'] = json.loads(d['dataset_ids']) if d.get('dataset_ids') else []
            d['metrics'] = json.loads(d['metrics']) if d.get('metrics') else {}
            d['baseline_metrics'] = json.loads(d['baseline_metrics']) if d.get('baseline_metrics') else {}
            return d

    def list_models(
        self,
        strip_type: Optional[str] = None,
        marker: Optional[str] = None,
        active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """List models with optional filtering by strip_type, marker, or active status."""
        query = 'SELECT * FROM ml_models WHERE 1=1'
        params: List[Any] = []
        if strip_type:
            query += ' AND strip_type = ?'
            params.append(strip_type)
        if marker:
            query += ' AND marker = ?'
            params.append(marker)
        if active_only:
            query += ' AND is_active = 1'
        query += ' ORDER BY strip_type, marker, version DESC'

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d['dataset_ids'] = json.loads(d['dataset_ids']) if d.get('dataset_ids') else []
                d['metrics'] = json.loads(d['metrics']) if d.get('metrics') else {}
                d['baseline_metrics'] = json.loads(d['baseline_metrics']) if d.get('baseline_metrics') else {}
                result.append(d)
            return result

    def get_active_model(self, strip_type: str, marker: str) -> Optional[Dict[str, Any]]:
        """Return currently active model for the given (strip_type, marker), if any."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM ml_models
                WHERE strip_type = ? AND marker = ? AND is_active = 1
                ORDER BY version DESC LIMIT 1
            ''', (strip_type, marker))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d['dataset_ids'] = json.loads(d['dataset_ids']) if d.get('dataset_ids') else []
            d['metrics'] = json.loads(d['metrics']) if d.get('metrics') else {}
            d['baseline_metrics'] = json.loads(d['baseline_metrics']) if d.get('baseline_metrics') else {}
            return d

    def activate_model(self, model_id: str) -> bool:
        """
        Activate a model for inference. Deactivates other versions
        of the same (strip_type, marker).
        """
        model = self.get_model(model_id)
        if not model:
            logger.warning(f"Cannot activate non-existent model {model_id}")
            return False

        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Deactivate any active model for this marker/strip
            cursor.execute(
                'UPDATE ml_models SET is_active = 0 WHERE strip_type = ? AND marker = ?',
                (model['strip_type'], model['marker'])
            )
            # Activate chosen model
            cursor.execute(
                'UPDATE ml_models SET is_active = 1, activated_at = ? WHERE model_id = ?',
                (now, model_id)
            )
            conn.commit()
            logger.info(f"Activated model {model_id} for {model['strip_type']}/{model['marker']}")
            return True

    def deactivate_model(self, model_id: str) -> bool:
        """Deactivate a model."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE ml_models SET is_active = 0 WHERE model_id = ?', (model_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_model(self, model_id: str) -> bool:
        """Delete model metadata row."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM ml_models WHERE model_id = ?', (model_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ── Phase 5: Training Jobs ────────────────────────────────────────────────

    def create_training_job(
        self,
        strip_type: str,
        marker: str,
        algorithm: str,
        max_concurrent: int = 1
    ) -> str:
        """Create a new training job entry with 'pending' status."""
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")

        job_id = f"JOB_{uuid.uuid4().hex[:10].upper()}"
        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                """
                SELECT COUNT(*) FROM training_jobs
                WHERE status IN ('pending', 'running', 'paused')
                """
            )
            if cursor.fetchone()[0] >= max_concurrent:
                raise TrainingCapacityError(
                    "Another training job is already active. Wait for it to finish "
                    "or stop it before starting a new job."
                )
            cursor.execute('''
                INSERT INTO training_jobs (
                    job_id, strip_type, marker, algorithm, status, progress,
                    log_output, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', 0, '', ?, ?)
            ''', (job_id, strip_type, marker, algorithm, now, now))
            conn.commit()
            return job_id

    def update_training_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        log_append: Optional[str] = None,
        error_message: Optional[str] = None,
        model_id: Optional[str] = None,
        process_id: Optional[int] = None
    ):
        """Update status, progress, log output or outcome of a training job."""
        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT log_output, status FROM training_jobs WHERE job_id = ?', (job_id,))
            row = cursor.fetchone()
            if not row:
                return

            current_log = row['log_output'] or ''
            new_log = (current_log + (log_append or ''))[-100_000:]

            updates = ['updated_at = ?', 'log_output = ?']
            params: List[Any] = [now, new_log]

            if status is not None:
                updates.append('status = ?')
                params.append(status)
                if status in ('completed', 'failed', 'cancelled'):
                    updates.append('completed_at = ?')
                    params.append(now)

            if progress is not None:
                updates.append('progress = ?')
                params.append(progress)

            if error_message is not None:
                updates.append('error_message = ?')
                params.append(error_message)

            if model_id is not None:
                updates.append('model_id = ?')
                params.append(model_id)

            if process_id is not None:
                updates.append('process_id = ?')
                params.append(process_id)

            params.append(job_id)
            terminal_statuses = ('completed', 'failed', 'cancelled')
            terminal_guard = (
                " AND status IN ('pending', 'running', 'paused')"
                if status in terminal_statuses else ""
            )
            cursor.execute(
                f"UPDATE training_jobs SET {', '.join(updates)} "
                f"WHERE job_id = ?{terminal_guard}",
                params
            )
            conn.commit()

    def get_training_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve training job status and logs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM training_jobs WHERE job_id = ?', (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_training_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent training jobs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT ?',
                (limit,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def list_active_training_jobs(self) -> List[Dict[str, Any]]:
        """List every job that can still consume training resources."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM training_jobs
                WHERE status IN ('pending', 'running', 'paused')
                ORDER BY created_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]
