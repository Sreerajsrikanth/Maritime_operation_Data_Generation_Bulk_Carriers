"""
Enhanced Bridge Generator with Uncertainty Injection

This module extends the existing EnhancedBridgeGenerator to add realistic
uncertainty injection capabilities while maintaining all existing functionality.

Key Additions:
1. Add uncertainty_enabled parameter to constructor
2. Add bridge-specific failure scenario tracking and progression
3. Modify existing data generation methods to apply uncertainties
4. Add new uncertainty-specific methods for navigation equipment failures

Based on validated maritime industry data:
- 5-8% total operational anomalies (navigation equipment issues)
- 90% normal operations, 7% alerts/warnings, 2% degradation, 0.8% failures
- GPS issues are leading cause (52% of ECDIS position offsets)
- Navigation equipment failure rates: 3.59/year → 0.010/day probability
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import random
from dataclasses import dataclass
import math

# Imports from project structure
from core.voyage_plan import VoyagePlan
from core.vessel_profile import VesselProfile, AutomationLevel, CompanyType, CrewStability


class CrewTimingSimulator:
    """Simulates realistic crew timing variations for bridge operations"""
    
    @staticmethod
    def generate_crew_timestamp(day: int, hour: int, 
                               task_duration_mins: Tuple[int, int] = (5, 15),
                               handover_delay: bool = False) -> datetime:
        """Generate realistic timestamp with crew timing variations"""
        
        # Base timestamp
        base_time = datetime(2024, 1, day, hour, 0)
        
        # Add task duration variation
        duration_variation = random.randint(*task_duration_mins)
        
        # Add handover delay if watch change
        handover_variation = random.randint(5, 20) if handover_delay else 0
        
        # Total delay
        total_delay = duration_variation + handover_variation
        
        return base_time + timedelta(minutes=total_delay)
    
    @staticmethod
    def is_handover_time(hour: int) -> bool:
        """Check if hour is a watch handover time (every 4 hours)"""
        return hour in [0, 4, 8, 12, 16, 20]


@dataclass
class NavigationEquipmentState:
    """Tracks navigation equipment state with profile-based degradation"""
    
    # GPS characteristics
    gps_precision_factor: float = 1.0  # 1.0 = modern, >1.0 = degraded
    gps_update_rate: int = 1  # Updates per minute
    
    # Radar characteristics  
    radar_range_factor: float = 1.0  # 1.0 = full range, <1.0 = reduced
    radar_target_detection: float = 1.0  # Target detection capability
    
    # Autopilot characteristics
    autopilot_engagement_rate: float = 0.6  # Fraction of time engaged
    course_correction_precision: float = 1.0  # Degrees precision
    
    # Communication equipment
    ais_reliability: float = 1.0  # AIS system reliability
    vhf_quality: float = 1.0  # VHF radio quality
    
    # Integrated systems
    ecdis_capability: bool = True  # Electronic chart capability
    integrated_nav: bool = True  # Integrated navigation systems


class EnhancedBridgeGenerator:
    """
    Enhanced Bridge Operations Generator with Vessel Profile Integration and Uncertainty Injection
    
    Generates authentic navigation operational data that varies systematically 
    based on vessel characteristics while maintaining voyage-plan-driven realism.
    
    NEW: Includes realistic uncertainty injection for bridge equipment failures,
    navigation anomalies, and crew performance variations.
    """
    
    def __init__(self, voyage_plan: VoyagePlan, uncertainty_enabled: bool = False):
        """
        Initialize Enhanced Bridge Generator with optional uncertainty injection
        
        Args:
            voyage_plan: VoyagePlan object with vessel profile integration
            uncertainty_enabled: Enable realistic uncertainty injection (default: False)
        """
        
        self.voyage_plan = voyage_plan
        self.vessel_profile = voyage_plan.vessel_profile
        self.uncertainty_enabled = uncertainty_enabled
        
        # Core navigation parameters
        self.route_total_distance = 2800  # nautical miles (typical bulk carrier route)
        self.base_speed = 12.0  # knots
        self.current_speed = self.base_speed
        self.distance_remaining = self.route_total_distance
        
        # Initialize equipment state and uncertainties
        self._apply_vessel_profile_adjustments()
        self._calculate_logging_parameters()
        self._initialize_navigation_equipment_state()
        
        # NEW: Initialize uncertainty tracking systems
        if self.uncertainty_enabled:
            self._initialize_uncertainty_systems()
        
        print(f"   Bridge Generator initialized for {self.vessel_profile.profile_name}")
        if self.uncertainty_enabled:
            print(f"   ⚠️  Uncertainty injection ENABLED")
    
    def _initialize_uncertainty_systems(self):
        """Initialize uncertainty tracking and failure simulation systems"""
        
        # Active navigation equipment failures
        self.active_failures = {}
        
        # Failure history for analysis
        self.failure_history = []
        
        # Human factors tracking
        self.human_factors = {
            'fatigue_accumulation': 0.0,      # Increases over time
            'watch_changes_today': 0,         # Resets daily
            'procedure_compliance': 0.9,      # Decreases with fatigue
            'navigation_accuracy': 1.0        # Affected by fatigue/failures
        }
        
        # Bridge-specific failure rates (MARCAT validated)
        self.base_failure_rates = {
            'gps_degradation': 0.005,    # 1.83/year → 0.005/day
            'radar_issues': 0.004,       # 1.46/year → 0.004/day  
            'autopilot_malfunction': 0.003,  # 1.10/year → 0.003/day
            'communication_loss': 0.002, # 0.73/year → 0.002/day
            'integrated_bridge_failure': 0.0003  # 0.11/year → 0.0003/day (rare but high impact)
        }
        
        # Apply vessel profile multipliers to failure rates
        profile_multipliers = {
            'Modern Major': {'gps': 0.7, 'radar': 0.8, 'autopilot': 0.6, 'comms': 0.8, 'integrated': 1.5},
            'Aging Midtier': {'gps': 1.0, 'radar': 1.0, 'autopilot': 1.0, 'comms': 1.0, 'integrated': 1.0},
            'Legacy Small': {'gps': 1.5, 'radar': 1.8, 'autopilot': 1.4, 'comms': 1.3, 'integrated': 0.3}
        }
        
        multiplier = profile_multipliers.get(self.vessel_profile.profile_name, 
                                           profile_multipliers['Aging Midtier'])
        
        # Apply multipliers
        self.adjusted_failure_rates = {
            'gps_degradation': self.base_failure_rates['gps_degradation'] * multiplier['gps'],
            'radar_issues': self.base_failure_rates['radar_issues'] * multiplier['radar'],
            'autopilot_malfunction': self.base_failure_rates['autopilot_malfunction'] * multiplier['autopilot'],
            'communication_loss': self.base_failure_rates['communication_loss'] * multiplier['comms'],
            'integrated_bridge_failure': self.base_failure_rates['integrated_bridge_failure'] * multiplier['integrated']
        }
    
    def _apply_vessel_profile_adjustments(self):
        """Apply systematic variations based on vessel profile"""
        
        profile = self.vessel_profile
        
        # Apply speed variations based on vessel profile
        speed_factors = {
            'Modern Major': 1.05,    # 5% faster (better efficiency)
            'Aging Midtier': 1.0,    # Baseline speed
            'Legacy Small': 0.92     # 8% slower (older engines)
        }
        
        self.base_speed *= speed_factors.get(profile.profile_name, 1.0)
        self.current_speed = self.base_speed
        
        # Apply operational timing variations
        timing_factors = {
            'Modern Major': 0.85,    # 15% faster operations (better training)
            'Aging Midtier': 1.0,    # Baseline timing
            'Legacy Small': 1.25     # 25% slower (less familiarity)
        }
        
        self.timing_variation_factor = timing_factors.get(profile.profile_name, 1.0)
    
    def _calculate_logging_parameters(self):
        """Calculate logging frequency based on vessel profile"""
        
        # Get bridge-specific logging multiplier from vessel profile
        # Use base frequency of 1.0 and let the vessel profile apply its multiplier
        bridge_logging_multiplier = self.vessel_profile.get_logging_frequency("bridge", 1.0)
        
        # Base logging multipliers by automation level
        automation_bonus = {
            AutomationLevel.HIGH: 1.3,    # More automated logging
            AutomationLevel.MEDIUM: 1.0,  # Standard logging
            AutomationLevel.LOW: 0.7      # More manual logging
        }
        
        self.logging_frequency_multiplier = (
            bridge_logging_multiplier * 
            automation_bonus[self.vessel_profile.automation_level]
        )
        
        # Calculate watch-specific intervals
        self.watch_end_logging = True  # Always log at watch end
        self.routine_check_interval = max(1, int(4 / self.logging_frequency_multiplier))
    
    def _initialize_navigation_equipment_state(self):
        """Initialize navigation equipment state based on vessel profile"""
        
        self.equipment_state = NavigationEquipmentState()
        profile = self.vessel_profile
        
        # 1. Vessel age affects equipment precision and reliability
        if profile.vessel_age <= 5:
            # Modern vessels: latest equipment standards
            self.equipment_state.gps_precision_factor = 0.9
            self.equipment_state.radar_range_factor = 1.1
            self.equipment_state.ecdis_capability = True
            self.equipment_state.integrated_nav = True
            
        elif profile.vessel_age <= 15:
            # Mid-age vessels: compliant standard equipment  
            self.equipment_state.gps_precision_factor = 1.0
            self.equipment_state.radar_range_factor = 1.0
            self.equipment_state.ecdis_capability = True
            self.equipment_state.integrated_nav = True
            
        else:
            # Aging vessels: older but compliant equipment
            self.equipment_state.gps_precision_factor = 1.2
            self.equipment_state.radar_range_factor = 0.9
            self.equipment_state.ecdis_capability = True
            self.equipment_state.integrated_nav = False
        
        # 2. Automation level affects operational procedures
        if profile.automation_level == AutomationLevel.HIGH:
            self.equipment_state.autopilot_engagement_rate = 0.80
            self.equipment_state.course_correction_precision = 0.4
            
        elif profile.automation_level == AutomationLevel.MEDIUM:
            self.equipment_state.autopilot_engagement_rate = 0.60
            self.equipment_state.course_correction_precision = 0.8
            
        else:  # LOW automation
            self.equipment_state.autopilot_engagement_rate = 0.35
            self.equipment_state.course_correction_precision = 1.5
        
        # 3. Company type affects equipment maintenance and reliability
        if profile.company_type == CompanyType.MAJOR:
            self.equipment_state.ais_reliability = 0.98
            self.equipment_state.vhf_quality = 0.95
        elif profile.company_type == CompanyType.MIDTIER:
            self.equipment_state.ais_reliability = 0.95
            self.equipment_state.vhf_quality = 0.90
        else:  # SMALL
            self.equipment_state.ais_reliability = 0.90
            self.equipment_state.vhf_quality = 0.85
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete bridge operations dataset for the voyage"""
        
        print(f"\n🚢 Generating Bridge Operations Data...")
        
        # Handle different attribute names for ports - use correct VoyagePlan attributes
        origin_port = getattr(self.voyage_plan, 'origin_port', 'Unknown Origin')
        destination_port = getattr(self.voyage_plan, 'destination_port', 'Unknown Destination')
        
        print(f"   Voyage: {origin_port} → {destination_port}")
        
        # Handle different attribute names for duration
        if hasattr(self.voyage_plan, 'voyage_duration_days'):
            duration_days = self.voyage_plan.voyage_duration_days
        else:
            # Calculate from start and end dates
            duration = self.voyage_plan.end_date - self.voyage_plan.start_date
            duration_days = duration.days
        
        print(f"   Duration: {duration_days} days")
        print(f"   Profile: {self.vessel_profile.profile_name}")
        if self.uncertainty_enabled:
            print(f"   ⚠️  Uncertainty injection: ACTIVE")
        
        all_data = []
        
        # Standard watch times (4-hour watches)
        watch_times = [0, 4, 8, 12, 16, 20]
        
        for day in range(1, duration_days + 1):
            # Get phase for this day - handle different VoyagePlan structures
            if hasattr(self.voyage_plan, 'get_phase_for_day'):
                # Original simple structure
                phase = self.voyage_plan.get_phase_for_day(day)
            else:
                # New VoyagePlan structure with timestamps
                day_timestamp = self.voyage_plan.start_date + timedelta(days=day-1)
                current_phase = self.voyage_plan.get_current_phase(day_timestamp)
                
                # Convert to dictionary format expected by generator
                phase = {
                    "phase": current_phase.value if hasattr(current_phase, 'value') else str(current_phase),
                    "logging_intensity": "high" if current_phase.value in ["loading", "unloading"] else "moderate"
                }
            
            # NEW: Update uncertainty systems daily
            if self.uncertainty_enabled:
                self._update_daily_uncertainty_state(day)
            
            # 1. Watch end logs (every 4 hours)
            for watch_hour in watch_times:
                timestamp = CrewTimingSimulator.generate_crew_timestamp(
                    day, watch_hour, handover_delay=True
                )
                
                bridge_data = self._generate_bridge_log(timestamp, phase, "watch_end")
                
                # NEW: Apply uncertainties if enabled
                if self.uncertainty_enabled:
                    bridge_data = self._apply_uncertainties(bridge_data, timestamp)
                
                all_data.append(bridge_data)
            
            # 2. Position fix logs (every 2 hours during sea transit, every hour during port ops)
            if phase.get("phase") in ["loading", "unloading"]:
                position_hours = range(1, 24, 1)  # Every hour in port
            else:
                position_hours = range(2, 24, 2)  # Every 2 hours at sea
            
            for pos_hour in position_hours:
                if pos_hour not in watch_times:  # Don't duplicate watch end logs
                    timestamp = CrewTimingSimulator.generate_crew_timestamp(
                        day, pos_hour, task_duration_mins=(2, 8)
                    )
                    
                    bridge_data = self._generate_bridge_log(timestamp, phase, "position_fix")
                    
                    # NEW: Apply uncertainties if enabled
                    if self.uncertainty_enabled:
                        bridge_data = self._apply_uncertainties(bridge_data, timestamp)
                    
                    all_data.append(bridge_data)
            
            # 3. Course change logs (random, 1-3 per day)
            course_changes = random.randint(1, 3)
            for _ in range(course_changes):
                change_hour = random.randint(1, 23)
                timestamp = CrewTimingSimulator.generate_crew_timestamp(
                    day, change_hour, task_duration_mins=(5, 15)
                )
                
                bridge_data = self._generate_bridge_log(timestamp, phase, "course_change")
                
                # NEW: Apply uncertainties if enabled
                if self.uncertainty_enabled:
                    bridge_data = self._apply_uncertainties(bridge_data, timestamp)
                
                all_data.append(bridge_data)
        
        # Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(all_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # NEW: Add uncertainty metadata if enabled
        if self.uncertainty_enabled:
            self._add_uncertainty_metadata(df)
        
        print(f"   Generated: {len(df):,} bridge operation records")
        if self.uncertainty_enabled:
            failure_count = len([r for r in all_data if r.get('risk_score', 0) > 0])
            print(f"   Anomalies: {failure_count:,} records ({failure_count/len(df)*100:.1f}%)")
        
        return df
    
    def _update_daily_uncertainty_state(self, day: int):
        """Update uncertainty tracking systems daily"""
        
        # Update human factors (fatigue accumulation)
        self.human_factors['fatigue_accumulation'] = min(1.0, 
            self.human_factors['fatigue_accumulation'] + 0.05)  # 5% per day
        
        # Reset daily counters
        self.human_factors['watch_changes_today'] = 0
        
        # Update procedure compliance based on fatigue
        base_compliance = 0.9
        fatigue_penalty = self.human_factors['fatigue_accumulation'] * 0.2
        self.human_factors['procedure_compliance'] = max(0.6, base_compliance - fatigue_penalty)
        
        # Check for new equipment failures
        self._check_equipment_failures(day)
        
        # Update existing failure progressions
        self._update_failure_progressions(day)
    
    def _check_equipment_failures(self, day: int):
        """Check for new equipment failures based on MARCAT failure rates"""
        
        for failure_type, daily_probability in self.adjusted_failure_rates.items():
            if random.random() < daily_probability:
                self._initiate_equipment_failure(failure_type, day)
    
    def _initiate_equipment_failure(self, failure_type: str, day: int):
        """Initiate a new equipment failure scenario"""
        
        # Determine failure severity
        severity = random.uniform(0.1, 1.0)
        
        # Create failure record
        failure_data = {
            'type': failure_type,
            'severity': severity,
            'initiated_day': day,
            'stage': 'early_warning',
            'duration_days': random.randint(1, 7),  # Most failures resolve within a week
            'affected_parameters': self._get_affected_parameters(failure_type)
        }
        
        # Add to active failures
        self.active_failures[failure_type] = failure_data
        
        # Log to failure history
        self.failure_history.append({
            'day': day,
            'event': 'failure_initiated',
            'failure_type': failure_type,
            'severity': severity,
            'vessel_id': getattr(self.voyage_plan, 'vessel_id', 'unknown')
        })
    
    def _get_affected_parameters(self, failure_type: str) -> List[str]:
        """Get list of parameters affected by each failure type"""
        
        parameter_map = {
            'gps_degradation': ['gps_accuracy_m', 'navigation_equipment_failures'],
            'radar_issues': ['radar_range_nm', 'radar_contacts_count', 'navigation_equipment_failures'],
            'autopilot_malfunction': ['autopilot_engaged', 'course_over_ground', 'navigation_equipment_failures'],
            'communication_loss': ['communication_status', 'ais_contacts_count'],
            'integrated_bridge_failure': ['gps_accuracy_m', 'radar_range_nm', 'autopilot_engaged', 
                                        'navigation_equipment_failures', 'bridge_alarm_count']
        }
        
        return parameter_map.get(failure_type, [])
    
    def _update_failure_progressions(self, day: int):
        """Update progression of existing failures"""
        
        failures_to_remove = []
        
        for failure_type, failure_data in self.active_failures.items():
            days_since_start = day - failure_data['initiated_day']
            
            # Update failure stage
            if days_since_start >= failure_data['duration_days']:
                # Failure resolved
                failures_to_remove.append(failure_type)
                self.failure_history.append({
                    'day': day,
                    'event': 'failure_resolved',
                    'failure_type': failure_type,
                    'severity': failure_data['severity'],
                    'vessel_id': getattr(self.voyage_plan, 'vessel_id', 'unknown')
                })
            elif days_since_start >= 3:
                failure_data['stage'] = 'critical'
            elif days_since_start >= 1:
                failure_data['stage'] = 'degraded'
        
        # Remove resolved failures
        for failure_type in failures_to_remove:
            del self.active_failures[failure_type]
    
    def _apply_uncertainties(self, record: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """Apply uncertainty effects to a bridge operation record"""
        
        if not self.uncertainty_enabled:
            return record
        
        # Create modified record copy
        modified_record = record.copy()
        
        # Apply active equipment failures
        for failure_type, failure_data in self.active_failures.items():
            severity = failure_data['severity']
            stage = failure_data['stage']
            
            # Apply stage-based severity multipliers
            stage_multipliers = {
                'early_warning': 0.3,
                'degraded': 0.6,
                'critical': 1.0
            }
            effective_severity = severity * stage_multipliers[stage]
            
            self._apply_failure_effects(modified_record, failure_type, effective_severity)
        
        # Apply human factors effects
        fatigue_effect = self.human_factors['fatigue_accumulation']
        compliance_effect = self.human_factors['procedure_compliance']
        
        # Modify navigation accuracy based on human factors
        if 'gps_accuracy_m' in modified_record:
            human_error_factor = 1.0 + (fatigue_effect * 0.5)  # Up to 50% accuracy loss
            modified_record['gps_accuracy_m'] *= human_error_factor
        
        # FIXED: Populate active_failures column properly
        if self.active_failures:
            active_failure_names = list(self.active_failures.keys())
            modified_record['active_failures'] = ','.join(active_failure_names)
        else:
            modified_record['active_failures'] = ''
        
        # Calculate composite risk score
        modified_record['risk_score'] = self._calculate_risk_score()
        
        return modified_record
    
    def _apply_failure_effects(self, record: Dict[str, Any], failure_type: str, severity: float):
        """Apply specific failure effects to a record"""
        
        if failure_type == 'gps_degradation':
            # GPS accuracy degradation
            if 'gps_accuracy_m' in record:
                # Normal: 1.6-8.2m → Degraded: 8-100m → Critical: 100-999m
                if severity < 0.3:
                    record['gps_accuracy_m'] *= (1 + severity * 5)  # 8-25m
                elif severity < 0.7:
                    record['gps_accuracy_m'] *= (1 + severity * 20)  # 25-100m
                else:
                    record['gps_accuracy_m'] = min(999, record['gps_accuracy_m'] * (1 + severity * 100))  # 100-999m
            
            if 'navigation_equipment_failures' in record:
                record['navigation_equipment_failures'] = min(4, record['navigation_equipment_failures'] + 1)
        
        elif failure_type == 'radar_issues':
            # Radar performance degradation
            if 'radar_range_nm' in record:
                record['radar_range_nm'] *= (1 - severity * 0.6)  # Up to 60% range reduction
            
            if 'radar_contacts_count' in record:
                record['radar_contacts_count'] = max(0, 
                    int(record['radar_contacts_count'] * (1 - severity * 0.4)))  # Missed contacts
            
            if 'navigation_equipment_failures' in record:
                record['navigation_equipment_failures'] = min(4, record['navigation_equipment_failures'] + 1)
        
        elif failure_type == 'autopilot_malfunction':
            # Autopilot issues
            if 'autopilot_engaged' in record:
                if severity > 0.5:
                    record['autopilot_engaged'] = 0  # Forced manual steering
            
            if 'course_over_ground' in record:
                # Course hunting/oscillation
                course_error = random.uniform(-severity * 10, severity * 10)
                record['course_over_ground'] += course_error
            
            if 'navigation_equipment_failures' in record:
                record['navigation_equipment_failures'] = min(4, record['navigation_equipment_failures'] + 1)
        
        elif failure_type == 'communication_loss':
            # Communication equipment issues
            if 'communication_status' in record:
                record['communication_status'] = max(0, record['communication_status'] - severity)
            
            if 'ais_contacts_count' in record:
                record['ais_contacts_count'] = max(0, 
                    int(record['ais_contacts_count'] * (1 - severity * 0.8)))  # AIS reception loss
        
        elif failure_type == 'integrated_bridge_failure':
            # Multiple system failure (rare but high impact)
            if 'gps_accuracy_m' in record:
                record['gps_accuracy_m'] = 999  # Dead reckoning mode
            
            if 'radar_range_nm' in record:
                record['radar_range_nm'] *= 0.4  # Emergency radar only
            
            if 'autopilot_engaged' in record:
                record['autopilot_engaged'] = 0  # Manual steering only
            
            if 'navigation_equipment_failures' in record:
                record['navigation_equipment_failures'] = 4  # Maximum failures
            
            if 'bridge_alarm_count' in record:
                record['bridge_alarm_count'] = min(15, record['bridge_alarm_count'] + 8)  # Alarm flood
    
    def _calculate_risk_score(self) -> float:
        """Calculate composite risk score based on all active factors"""
        
        # Base risk from active failures
        failure_risk = sum(f['severity'] * 0.4 for f in self.active_failures.values())
        
        # Human factors risk
        human_risk = (self.human_factors['fatigue_accumulation'] * 0.3 + 
                     (1 - self.human_factors['procedure_compliance']) * 0.2)
        
        # Cascade amplification for multiple failures
        cascade_count = len(self.active_failures)
        cascade_risk = (cascade_count - 1) * 0.1 if cascade_count > 1 else 0
        
        total_risk = min(1.0, failure_risk + human_risk + cascade_risk)
        return round(total_risk, 3)
    
    def _add_uncertainty_metadata(self, df: pd.DataFrame):
        """Add uncertainty tracking columns to dataframe"""
        
        # Ensure all uncertainty columns exist and are properly initialized
        uncertainty_columns = {
            'active_failures': '',
            'failure_stage': 'normal', 
            'risk_score': 0.0,
            'maintenance_urgency': 'routine'
        }
        
        for col, default_value in uncertainty_columns.items():
            if col not in df.columns:
                df[col] = default_value
        
        # Update failure stages based on risk scores
        df.loc[df['risk_score'] > 0.8, 'failure_stage'] = 'critical'
        df.loc[df['risk_score'] > 0.8, 'maintenance_urgency'] = 'immediate'
        df.loc[(df['risk_score'] > 0.6) & (df['risk_score'] <= 0.8), 'failure_stage'] = 'severe'
        df.loc[(df['risk_score'] > 0.6) & (df['risk_score'] <= 0.8), 'maintenance_urgency'] = 'urgent'
        df.loc[(df['risk_score'] > 0.3) & (df['risk_score'] <= 0.6), 'failure_stage'] = 'moderate'
        df.loc[(df['risk_score'] > 0.3) & (df['risk_score'] <= 0.6), 'maintenance_urgency'] = 'scheduled'
        
        # Ensure no null values in active_failures column
        df['active_failures'] = df['active_failures'].fillna('')
        
        print(f"   📊 Uncertainty metadata added:")
        print(f"      - Active failures: {len(df[df['active_failures'] != ''])} records")
        print(f"      - Risk events: {len(df[df['risk_score'] > 0])} records")
        print(f"      - Failure stages: {df['failure_stage'].value_counts().to_dict()}")
    
    # EXISTING METHODS (keeping all original generation methods unchanged)
    def _generate_bridge_log(self, timestamp: datetime, phase: Dict, log_type: str) -> Dict[str, Any]:
        """Generate bridge operational log entry (EXISTING METHOD - UNCHANGED)"""
        
        # Calculate position
        progress = self._calculate_voyage_progress(timestamp)
        lat, lon = self._calculate_position(progress)
        
        # Calculate navigation parameters
        course = self._calculate_course_over_ground(phase)
        speed = self._calculate_speed_over_ground(phase)
        
        # Calculate equipment status
        autopilot_engaged = self._determine_autopilot_status(phase, log_type)
        ais_contacts, radar_contacts = self._calculate_traffic_info(phase)
        
        # Environmental conditions
        wind_speed, wind_direction = self._generate_weather_conditions()
        wave_height = self._calculate_wave_height(wind_speed)
        visibility = self._calculate_visibility()
        barometric_pressure = self._generate_barometric_pressure()
        air_temperature = self._generate_air_temperature()
        
        # Equipment precision and status
        gps_accuracy = self._calculate_gps_accuracy()
        radar_range = self._calculate_radar_range()
        communication_status = self._determine_communication_status()
        
        # Navigation equipment status
        navigation_failures = 0  # Base value, modified by uncertainties
        bridge_alarms = self._calculate_bridge_alarms()
        
        # Cargo monitoring status
        cargo_monitoring_active = 1 if phase.get("phase") in ["loading", "unloading"] else 0
        
        # ECDIS operational status
        ecdis_operational = 1 if self.equipment_state.ecdis_capability else 0
        
        # Crew information
        crew_member = self._get_current_watch_officer(timestamp)
        
        # Distance calculations
        distance_to_destination = self.distance_remaining
        
        return {
            'timestamp': timestamp,
            'operational_phase': phase.get("phase", "unknown"),
            'log_type': log_type,
            'latitude': lat,
            'longitude': lon,
            'course_over_ground': course,
            'speed_over_ground_knots': speed,
            'distance_to_destination_nm': distance_to_destination,
            'radar_contacts_count': radar_contacts,
            'ais_contacts_count': ais_contacts,
            'autopilot_engaged': 1 if autopilot_engaged else 0,
            'wind_speed_knots': wind_speed,
            'wind_direction': wind_direction,
            'wave_height_m': wave_height,
            'visibility_nm': visibility,
            'barometric_pressure_hpa': barometric_pressure,
            'air_temperature_c': air_temperature,
            'navigation_equipment_failures': navigation_failures,
            'bridge_alarm_count': bridge_alarms,
            'communication_status': communication_status,
            'cargo_monitoring_active': cargo_monitoring_active,
            'gps_accuracy_m': gps_accuracy,
            'radar_range_nm': radar_range,
            'ecdis_operational': ecdis_operational,
            'crew_member': crew_member,
            'vessel_id': getattr(self.voyage_plan, 'vessel_id', 'unknown'),
            'voyage_id': getattr(self.voyage_plan, 'voyage_id', 'unknown'),
            'cargo_type': getattr(self.voyage_plan, 'cargo_type', 'unknown'),
            'operational_area': 'bridge',
            'vessel_profile': self.vessel_profile.profile_name
        }
    
    def _calculate_voyage_progress(self, timestamp: datetime) -> float:
        """Calculate voyage progress as fraction (0.0 to 1.0)"""
        
        # Get voyage start time
        if hasattr(self.voyage_plan, 'start_date'):
            start_time = self.voyage_plan.start_date
        else:
            # Fallback to day 1
            start_time = datetime(2024, 1, 1)
        
        # Calculate elapsed time
        elapsed = timestamp - start_time
        elapsed_hours = elapsed.total_seconds() / 3600
        
        # Get total voyage duration
        if hasattr(self.voyage_plan, 'voyage_duration_days'):
            total_hours = self.voyage_plan.voyage_duration_days * 24
        else:
            duration = self.voyage_plan.end_date - self.voyage_plan.start_date
            total_hours = duration.total_seconds() / 3600
        
        # Calculate progress
        progress = min(1.0, elapsed_hours / total_hours)
        
        # Update distance remaining
        self.distance_remaining = self.route_total_distance * (1 - progress)
        
        return progress
    
    def _calculate_position(self, progress: float) -> Tuple[float, float]:
        """Calculate current position based on voyage progress"""
        
        # Example route: Rotterdam (51.9225, 4.4792) to Shanghai (31.2304, 121.4737)
        start_lat, start_lon = 51.9225, 4.4792
        end_lat, end_lon = 31.2304, 121.4737
        
        # Linear interpolation (simplified great circle)
        current_lat = start_lat + (end_lat - start_lat) * progress
        current_lon = start_lon + (end_lon - start_lon) * progress
        
        # Add GPS precision error based on equipment state
        precision_error = 0.001 * self.equipment_state.gps_precision_factor
        lat_error = random.uniform(-precision_error, precision_error)
        lon_error = random.uniform(-precision_error, precision_error)
        
        # Calculate final position
        final_lat = current_lat + lat_error
        final_lon = current_lon + lon_error
        
        # COORDINATE VALIDATION FIX - Clamp to valid ranges
        final_lat = max(-90.0, min(90.0, final_lat))      # Latitude: -90 to +90
        final_lon = max(-180.0, min(180.0, final_lon))    # Longitude: -180 to +180
        
        return final_lat, final_lon
    
    def _calculate_course_over_ground(self, phase: Dict) -> float:
        """Calculate course over ground with variations"""
        
        # Base course (approximately southeast Rotterdam to Shanghai)
        base_course = 135.0  # degrees
        
        # Add course variations based on phase
        if phase.get("phase") in ["loading", "unloading"]:
            # Port maneuvering
            course_variation = random.uniform(-30, 30)
        else:
            # Sea transit - minor course corrections
            course_variation = random.uniform(-5, 5)
        
        # Apply autopilot precision
        if hasattr(self, 'equipment_state'):
            precision_error = self.equipment_state.course_correction_precision
            course_variation += random.uniform(-precision_error, precision_error)
        
        final_course = (base_course + course_variation) % 360
        return round(final_course, 1)
    
    def _calculate_speed_over_ground(self, phase: Dict) -> float:
        """Calculate speed over ground with environmental factors"""
        
        base_speed = self.current_speed
        
        # Phase-specific speed adjustments
        if phase.get("phase") in ["loading", "unloading"]:
            # Port operations - stationary or very slow maneuvering (FIXED: was 0.5-3.0)
            base_speed = random.uniform(0.0, 0.5)  # 0-0.5 knots (nearly stationary)
        elif phase.get("phase") == "sea_transit":
            # Sea transit - normal speed with variations
            base_speed = self.base_speed + random.uniform(-1.0, 1.0)
        
        # Apply current and weather effects only if not in port
        if phase.get("phase") not in ["loading", "unloading"]:
            current_effect = random.uniform(-0.5, 0.5)
            weather_effect = random.uniform(-1.0, 0.2)
            base_speed += current_effect + weather_effect
        
        sog = max(0.0, base_speed)  # Ensure non-negative speed
        return round(sog, 1)
    
    def _determine_autopilot_status(self, phase: Dict, log_type: str) -> bool:
        """Determine if autopilot is engaged"""
        
        phase_name = phase.get("phase")
        
        # Force manual steering in port
        if phase_name in ["loading", "unloading"]:
            return False
        
        # Force manual during course changes
        if log_type == "course_change":
            return False
        
        # Otherwise use profile-based engagement rate
        return random.random() < self.equipment_state.autopilot_engagement_rate
    
    def _calculate_traffic_info(self, phase: Dict) -> Tuple[int, int]:
        """Calculate AIS contacts and radar targets"""
        
        phase_name = phase.get("phase")
        
        # Base traffic depends on phase
        if phase_name in ["loading", "unloading"]:
            # Port areas: high traffic
            base_ais = 12
            base_radar = 18
        else:
            # Open ocean: lower traffic
            base_ais = 4
            base_radar = 7
        
        # Apply equipment reliability
        ais_multiplier = self.equipment_state.ais_reliability
        radar_multiplier = self.equipment_state.radar_range_factor
        
        # Add realistic variation
        ais_contacts = max(0, int(np.random.poisson(base_ais * ais_multiplier)))
        radar_targets = max(0, int(np.random.poisson(base_radar * radar_multiplier)))
        
        return ais_contacts, radar_targets
    
    def _generate_weather_conditions(self) -> Tuple[float, float]:
        """Generate realistic weather conditions"""
        
        # Wind speed (knots) - typical ocean conditions
        wind_speed = max(0, np.random.normal(15, 8))
        
        # Wind direction (degrees)
        wind_direction = random.uniform(0, 360)
        
        return round(wind_speed, 1), round(wind_direction, 1)
    
    def _calculate_wave_height(self, wind_speed: float) -> float:
        """Calculate wave height based on wind speed"""
        
        # Simplified wave height calculation (Douglas Sea Scale approximation)
        if wind_speed < 4:
            wave_height = random.uniform(0.1, 0.5)
        elif wind_speed < 11:
            wave_height = random.uniform(0.5, 1.5)
        elif wind_speed < 22:
            wave_height = random.uniform(1.5, 3.5)
        elif wind_speed < 34:
            wave_height = random.uniform(3.5, 6.0)
        else:
            wave_height = random.uniform(6.0, 12.0)
        
        return round(wave_height, 1)
    
    def _calculate_visibility(self) -> float:
        """Calculate visibility in nautical miles"""
        
        # Most conditions have good visibility
        if random.random() < 0.8:
            visibility = random.uniform(8, 15)  # Good visibility
        elif random.random() < 0.9:
            visibility = random.uniform(3, 8)   # Moderate visibility
        else:
            visibility = random.uniform(0.5, 3) # Poor visibility (fog, rain)
        
        return round(visibility, 1)
    
    def _generate_barometric_pressure(self) -> float:
        """Generate realistic barometric pressure"""
        
        # Normal atmospheric pressure with variations
        pressure = np.random.normal(1013.25, 15)  # hPa
        return round(pressure, 1)
    
    def _generate_air_temperature(self) -> float:
        """Generate air temperature based on location/season"""
        
        # Simplified temperature model (assuming moderate climate)
        temperature = np.random.normal(18, 8)  # Celsius
        return round(temperature, 1)
    
    def _calculate_gps_accuracy(self) -> float:
        """Calculate GPS accuracy with equipment-based variations"""
        
        # Base accuracy depends on equipment state
        if self.equipment_state.gps_precision_factor <= 1.0:
            # Modern GPS
            base_accuracy = random.uniform(1.6, 4.0)
        elif self.equipment_state.gps_precision_factor <= 1.2:
            # Standard GPS
            base_accuracy = random.uniform(2.0, 6.0)
        else:
            # Legacy GPS
            base_accuracy = random.uniform(3.0, 8.2)
        
        # Apply precision factor
        final_accuracy = base_accuracy * self.equipment_state.gps_precision_factor
        
        return round(final_accuracy, 1)
    
    def _calculate_radar_range(self) -> float:
        """Calculate effective radar range"""
        
        # Base range depends on equipment
        base_range = 30.0  # nautical miles
        
        # Apply range factor from equipment state
        effective_range = base_range * self.equipment_state.radar_range_factor
        
        # Add minor variations
        variation = random.uniform(-2, 2)
        final_range = max(5.0, effective_range + variation)
        
        return round(final_range, 1)
    
    def _determine_communication_status(self) -> int:
        """Determine communication equipment status"""
        
        # Basic status based on equipment reliability
        if random.random() < self.equipment_state.vhf_quality:
            return 1  # Operational
        else:
            return 0  # Issues
    
    def _calculate_bridge_alarms(self) -> int:
        """Calculate number of active bridge alarms"""
        
        # Most of the time, no alarms
        if random.random() < 0.85:
            return 0
        elif random.random() < 0.95:
            return random.randint(1, 2)
        else:
            return random.randint(2, 3)
    
    def _get_current_watch_officer(self, timestamp: datetime) -> str:
        """Determine current watch officer based on time"""
        
        hour = timestamp.hour
        
        # Standard 4-hour watches
        if 0 <= hour < 4:
            return "OOW_1"
        elif 4 <= hour < 8:
            return "OOW_2"
        elif 8 <= hour < 12:
            return "OOW_3"
        elif 12 <= hour < 16:
            return "OOW_1"
        elif 16 <= hour < 20:
            return "OOW_2"
        else:  # 20-24
            return "OOW_3"
    
    # NEW: Utility methods for uncertainty analysis
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary of current uncertainty state"""
        
        if not self.uncertainty_enabled:
            return {'uncertainty_enabled': False}
        
        return {
            'uncertainty_enabled': True,
            'active_failures': {
                failure_type: {
                    'severity': data['severity'],
                    'stage': data['stage'],
                    'initiated_day': data['initiated_day'],
                    'affected_parameters': data['affected_parameters']
                }
                for failure_type, data in self.active_failures.items()
            },
            'human_factors': self.human_factors.copy(),
            'total_failure_events': len(self.failure_history),
            'current_risk_score': self._calculate_risk_score()
        }
    
    def get_failure_history(self) -> List[Dict]:
        """Get complete failure history for analysis"""
        
        return [
            {
                'day': event['day'],
                'event': event['event'],
                'failure_type': event['failure_type'],
                'severity': event['severity'],
                'vessel_id': event['vessel_id']
            }
            for event in self.failure_history
        ]

