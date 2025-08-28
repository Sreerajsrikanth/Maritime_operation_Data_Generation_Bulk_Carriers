"""
Enhanced Engine Room Data Generator with Uncertainty Injection - CORRECTED VERSION

This modifies the existing EnhancedEngineRoomGenerator to add realistic
uncertainty injection capabilities while maintaining all existing functionality.

KEY CORRECTIONS BASED ON EXTERNAL ANALYSIS:
1. Fix active_failures format (string instead of integer)
2. Add realistic weather condition variation 
3. Fix operational_area to be geographic regions
4. Improve maintenance_urgency variation based on risk scores
5. Increase engine load range to 95% during sea transit
6. Add realistic failure rates with MARCAT-based probabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import random
import math

import sys
import os


from core.vessel_profile import VesselProfile
from core.voyage_plan import VoyagePlan, VoyagePhase


class EnhancedEngineRoomGenerator:
    """
    Enhanced Engine Room Data Generator with Vessel Profile Integration and Uncertainty Injection
    
    This generator creates authentic engine room operational data with:
    1. Systematic variations based on vessel profile characteristics
    2. OPTIONAL uncertainty injection for training ML models on failure scenarios
    3. Maintains backward compatibility - can generate clean data when uncertainty_enabled=False
    """
    def __init__(self, voyage_plan: VoyagePlan, uncertainty_enabled: bool = False, failure_config: Optional[Dict] = None):
        """
        Initialize generator with optional uncertainty injection
        """
        self.voyage_plan = voyage_plan
        self.vessel_profile = voyage_plan.vessel_profile
        self.uncertainty_enabled = uncertainty_enabled
        self.failure_config = failure_config or {}
        
        # Base operational parameters (UNCHANGED)
        self.base_parameters = {
            "main_engine_power_kw": 12000,
            "auxiliary_engine_power_kw": 800,
            "fuel_consumption_base_mt_per_day": 24.0,
            "operating_temperature_base_c": 85.0,
            "cooling_water_flow_base_m3_per_hour": 450.0,
            "lubricating_oil_pressure_base_bar": 4.2,
            "fuel_oil_pressure_base_bar": 12.0
        }
        
        # Apply vessel profile adjustments (UNCHANGED)
        self._apply_vessel_profile_adjustments()
        
        # Calculate logging parameters (UNCHANGED)
        self._calculate_logging_parameters()
        
        # Initialize equipment degradation tracking (UNCHANGED)
        self._initialize_equipment_state()
        
        # NEW: Geographic operational areas (CORRECTED)
        self.operational_areas = [
            'north_atlantic', 'mediterranean', 'pacific', 
            'indian_ocean', 'baltic_sea', 'north_sea', 'caribbean'
        ]
        
        # NEW: Realistic weather distribution (CORRECTED)
        self.weather_weights = {
            'calm': 0.35,           # 35%
            'moderate': 0.30,       # 30% 
            'rough': 0.20,          # 20%
            'heavy_weather': 0.12,  # 12%
            'gale': 0.03           # 3%
        }
        
        # FIXED: Always initialize uncertainty attributes to prevent AttributeError
        self.base_failure_rates = {}
        self.active_failures = {}
        self.failure_history = []
        self.human_factors = {
            'fatigue_accumulation': 0.0,
            'experience_degradation': 0.0,
            'procedure_compliance': 0.8
        }
        
        self.active_electrical_failures = []
        self.electrical_failure_start = None
        self.current_main_engine_load = 0.6
        
        # FIXED: Only populate uncertainty data if enabled
        if self.uncertainty_enabled:
            self._populate_uncertainty_injection()

    def _populate_uncertainty_injection(self):
        """Populate uncertainty injection data only when enabled"""
        
        # MARCAT maritime failure rates (per day) - CORRECTED RATES
        self.base_failure_rates = {
            'propulsion': 6.96 / 365,   # 0.019 per day (MARCAT data)
            'electrical': 5.78 / 365,   # 0.016 per day (MARCAT data)
            'fuel_system': 4.2 / 365,   # 0.011 per day (subset of propulsion)
            'cooling': 3.8 / 365,       # 0.010 per day (subset of propulsion)
        }
        
        # Phase-specific risk multipliers (from project research)
        self.phase_risk_multipliers = {
            'loading': 2.0,          # Equipment intensive operations
            'unloading': 2.0,        # Equipment intensive operations  
            'departure': 1.3,        # Port operations
            'arrival': 1.3,          # Port operations
            'sea_transit': 1.0,      # Normal operations
            'port_operations': 1.8,  # General port operations
        }
        
        # Weather risk multipliers (CORRECTED)
        self.weather_risk_multipliers = {
            'calm': 0.8,            # Lower risk in calm conditions
            'moderate': 1.0,        # Baseline risk
            'rough': 1.5,           # Higher risk in rough weather
            'heavy_weather': 2.2,   # Much higher risk
            'gale': 3.0             # Extreme risk
        }
        
        # Vessel profile failure multipliers
        vessel_multipliers = {
            'Modern Major': {'propulsion': 0.7, 'electrical': 0.8, 'human': 0.6},
            'Aging Midtier': {'propulsion': 1.0, 'electrical': 1.2, 'human': 1.0},
            'Legacy Small': {'propulsion': 1.5, 'electrical': 1.8, 'human': 1.4}
        }
        
        self.vessel_failure_multipliers = vessel_multipliers.get(
            self.vessel_profile.profile_name, 
            vessel_multipliers['Aging Midtier']
        )
        
        print(f"Uncertainty injection populated for {self.vessel_profile.profile_name}")
    def _initialize_uncertainty_injection(self):
        """Initialize uncertainty injection system"""
        
        # MARCAT maritime failure rates (per day) - CORRECTED RATES
        self.base_failure_rates = {
            'propulsion': 6.96 / 365,   # 0.019 per day (MARCAT data)
            'electrical': 5.78 / 365,   # 0.016 per day (MARCAT data)
            'fuel_system': 4.2 / 365,   # 0.011 per day (subset of propulsion)
            'cooling': 3.8 / 365,       # 0.010 per day (subset of propulsion)
        }
        
        # Phase-specific risk multipliers (from project research)
        self.phase_risk_multipliers = {
            'loading': 2.0,          # Equipment intensive operations
            'unloading': 2.0,        # Equipment intensive operations  
            'departure': 1.3,        # Port operations
            'arrival': 1.3,          # Port operations
            'sea_transit': 1.0,      # Normal operations
            'port_operations': 1.8,  # General port operations
        }
        
        # Weather risk multipliers (CORRECTED)
        self.weather_risk_multipliers = {
            'calm': 0.8,            # Lower risk in calm conditions
            'moderate': 1.0,        # Baseline risk
            'rough': 1.5,           # Higher risk in rough weather
            'heavy_weather': 2.2,   # Much higher risk
            'gale': 3.0             # Extreme risk
        }
        
        # Vessel profile failure multipliers
        vessel_multipliers = {
            'Modern Major': {'propulsion': 0.7, 'electrical': 0.8, 'human': 0.6},
            'Aging Midtier': {'propulsion': 1.0, 'electrical': 1.2, 'human': 1.0},
            'Legacy Small': {'propulsion': 1.5, 'electrical': 1.8, 'human': 1.4}
        }
        
        self.vessel_failure_multipliers = vessel_multipliers.get(
            self.vessel_profile.profile_name, 
            vessel_multipliers['Aging Midtier']
        )
        
        # Track active failures and their progression
        self.active_failures = {}
        self.failure_history = []
        self.human_factors = {
            'fatigue_accumulation': 0.0,
            'experience_degradation': 0.0,
            'procedure_compliance': 0.8  # Base compliance
        }
        
        print(f"🔧 Uncertainty injection enabled for {self.vessel_profile.profile_name}")
        print(f"   Failure rate multipliers: {self.vessel_failure_multipliers}")
    
    # NEW: Weather condition generation (CORRECTED)
    def _get_realistic_weather_condition(self) -> str:
        """Generate realistic weather distribution instead of always 'calm'"""
        
        rand_val = random.random()
        cumulative = 0
        
        for weather, weight in self.weather_weights.items():
            cumulative += weight
            if rand_val <= cumulative:
                return weather
        
        return 'moderate'  # fallback
    
    # NEW: Geographic operational area (CORRECTED)
    def _get_operational_area(self) -> str:
        """Get geographic operational area based on voyage route"""
        # For now, select randomly. Later can integrate with voyage plan routing
        return random.choice(self.operational_areas)
    
    # NEW: Realistic maintenance urgency calculation (CORRECTED)
    def _get_maintenance_urgency(self, risk_score: float, active_failures: str) -> str:
        """Calculate maintenance urgency based on risk score and failures"""
        
        if active_failures:  # Any active failure
            if risk_score > 0.200:
                return 'immediate'
            elif risk_score > 0.150:
                return 'urgent'
            else:
                return 'scheduled'
        else:  # No active failures
            if risk_score > 0.150:
                return 'urgent'
            elif risk_score > 0.100:
                return 'scheduled'
            else:
                return 'routine'
    
    # NEW: Active failures as string format (CORRECTED)
    def _get_active_failures_string(self, risk_score: float) -> str:
        """Generate realistic active failures as strings, not integers"""
        
        # Don't generate new failures here if uncertainty injection handles it
        if self.uncertainty_enabled:
            active_failure_list = list(self.active_failures.keys())
            return ','.join(active_failure_list) if active_failure_list else ''
        
        # For non-uncertainty mode, use MARCAT-based probabilities
        failure_types = ['propulsion', 'electrical', 'fuel_system', 'cooling']
        
        for failure_type in failure_types:
            daily_rate = self.base_failure_rates.get(failure_type, 0.01)
            # Convert daily rate to per-record rate (assuming 24 records/day)
            record_rate = daily_rate / 24
            
            if random.random() < record_rate:
                return failure_type
        
        return ''  # No active failures (most common)
    
    # EXISTING METHODS (UNCHANGED) - keeping all original functionality
    def _apply_vessel_profile_adjustments(self):
        """Apply vessel profile adjustments to base parameters (UNCHANGED)"""
        
        # Adjust parameters based on vessel age
        age_factor = 1.0 + (self.vessel_profile.vessel_age - 10) * 0.02
        
        # Older vessels have slightly higher fuel consumption
        if self.vessel_profile.vessel_age > 15:
            self.base_parameters["fuel_consumption_base_mt_per_day"] *= 1.1
        elif self.vessel_profile.vessel_age > 10:
            self.base_parameters["fuel_consumption_base_mt_per_day"] *= 1.05
    
    def _calculate_logging_parameters(self):
        """Calculate logging parameters based on vessel profile (CORRECTED)"""
        
        # Base logging frequency (entries per day)
        base_frequency = 24.0  # Hourly logging baseline
        
        # Automation level affects logging frequency
        automation_multipliers = {
            "high": 1.5,    # More automated monitoring
            "medium": 1.0,  # Standard monitoring
            "low": 0.7      # More manual monitoring
        }
        
        automation_level = self.vessel_profile.automation_level.value
        self.logging_frequency = base_frequency * automation_multipliers.get(automation_level, 1.0)
        
        # 🔧 CORRECTED: Precision factor calculation
        # Modern vessels should have BETTER precision (lower measurement noise)
        # High automation should REDUCE noise, not increase it
        
        # Age-based precision: newer vessels have better instruments
        age_precision_factor = max(0.5, min(1.2, 0.4 + (25 - self.vessel_profile.vessel_age) * 0.02))
        
        # Automation precision: high automation REDUCES measurement noise
        automation_precision_multipliers = {
            "high": 0.8,    # High automation = 20% LESS noise  
            "medium": 1.0,  # Standard noise
            "low": 1.3      # Low automation = 30% MORE noise
        }
        
        automation_precision_factor = automation_precision_multipliers.get(automation_level, 1.0)
        
        # Calculate final precision factor
        raw_precision = age_precision_factor * automation_precision_factor
        
        # 🔧 CRITICAL: Cap precision factor to prevent massive anomalies
        self.precision_factor = max(0.3, min(1.2, raw_precision))
        
        # Human factors (unchanged)
        crew_stability_scores = {"high": 0.9, "medium": 0.75, "low": 0.6}
        self.base_crew_familiarity = crew_stability_scores.get(
            self.vessel_profile.crew_stability.value, 0.75
        )
    
    def _initialize_equipment_state(self):
        """Initialize equipment degradation state (UNCHANGED)"""
        
        # Equipment degradation based on vessel age
        base_degradation = min(0.3, (self.vessel_profile.vessel_age - 5) * 0.02)
        
        self.equipment_condition = {
            "main_engine": 1.0 - base_degradation,
            "auxiliary_engine": 1.0 - base_degradation * 0.8,
            "fuel_system": 1.0 - base_degradation * 1.2,
            "cooling_system": 1.0 - base_degradation,
            "electrical_system": max(0.95, 1.0 - base_degradation * 0.02),
            "lubrication_system": 1.0 - base_degradation * 1.1
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete engine room dataset with optional uncertainty injection
        """
        print(f"Generating engine room data for {self.vessel_profile.profile_name}")
        if self.uncertainty_enabled:
            print(f"Uncertainty injection: ENABLED")
        else:
            print(f"Clean operational data: ENABLED")
        
        voyage_duration = (self.voyage_plan.end_date - self.voyage_plan.start_date).days
        
        dataset_records = []
        current_time = self.voyage_plan.start_date
        
        # Generate data based on logging frequency
        time_interval = timedelta(hours=24/self.logging_frequency)
        
        while current_time <= self.voyage_plan.end_date:
            
            # Get current voyage phase
            current_phase = self.voyage_plan.get_current_phase(current_time)
            
            # Process uncertainty injection if enabled
            if self.uncertainty_enabled:
                self._process_uncertainty_step(current_time, current_phase)
            
            # Generate operational data
            record = self._generate_engine_record(current_time, current_phase)
            
            # Apply uncertainty effects if active
            if self.uncertainty_enabled:
                record = self._apply_uncertainty_effects(record, current_time)
            
            dataset_records.append(record)
            current_time += time_interval
        
        df = pd.DataFrame(dataset_records)
        
        # Add metadata based on uncertainty mode
        if self.uncertainty_enabled:
            self._add_uncertainty_metadata(df)
        else:
            self._add_clean_metadata(df)
        
        print(f"Generated {len(df)} engine room records")
        if self.uncertainty_enabled and self.failure_history:
            print(f"Injected {len(self.failure_history)} failure events")
        
        return df
    
    def _add_clean_metadata(self, df: pd.DataFrame):
        """Add clean operation metadata when uncertainty is disabled"""
        
        # All operations are normal
        df['risk_score'] = 0.0
        df['failure_progression_stage'] = 'normal'
        df['maintenance_urgency'] = 'routine'
        df['active_failures'] = ''
        df['cascade_active'] = False
        
        print(f"Clean metadata added: {len(df)} normal operation records")
        
    def _generate_engine_record(self, timestamp: datetime, phase_info) -> Dict[str, Any]:
        """Generate single engine record (CORRECTED with all fixes)"""
        
        # Handle phase_info - it could be VoyagePhase object or dict
        if isinstance(phase_info, dict):
            voyage_phase = phase_info.get('phase', VoyagePhase.SEA_TRANSIT)
            weather_condition = phase_info.get('weather_condition', 'calm')
        else:
            # phase_info is already a VoyagePhase object
            voyage_phase = phase_info
            weather_condition = self._get_realistic_weather_condition()  # CORRECTED: Not always 'calm'
        
        # Generate main engine data with corrected load ranges
        main_engine_data = self._generate_main_engine_data(timestamp, voyage_phase, weather_condition)
        # 🔍 DEBUG: Print main engine data
        print(f"DEBUG main engine generation:")
        print(f"  Main engine load: {main_engine_data.get('main_engine_load_percentage', 'NOT FOUND')}%")
        
        # Store engine load for electrical correlation
        if 'main_engine_load_percentage' in main_engine_data:
            self.current_main_engine_load = main_engine_data['main_engine_load_percentage'] / 100
            print(f"  Stored current_main_engine_load: {self.current_main_engine_load:.3f}")
        else:
            print(f"  ❌ ERROR: main_engine_load_percentage not found in main_engine_data!")
            print(f"  Available keys: {list(main_engine_data.keys())}")
        
        # Generate auxiliary systems data
        auxiliary_data = self._generate_auxiliary_engine_data(timestamp, voyage_phase)
        fuel_data = self._generate_fuel_system_data(timestamp, voyage_phase)
        cooling_data = self._generate_cooling_system_data(timestamp, voyage_phase)
        lubrication_data = self._generate_lubrication_system_data(timestamp, voyage_phase)
        electrical_data = self._generate_electrical_system_data(timestamp, voyage_phase)
        
        # Generate maintenance flags
        maintenance_data = self._generate_maintenance_flags(timestamp)
        
        # Generate human factors
        human_data = self._generate_human_factors_data(timestamp)
        
        # CORRECTED: Calculate risk score
        risk_score = self._calculate_risk_score() if self.uncertainty_enabled else 0.04 + random.random() * 0.191
        
        # CORRECTED: Get active failures as string format
        active_failures = self._get_active_failures_string(risk_score)
        
        # CORRECTED: Get realistic maintenance urgency
        maintenance_urgency = self._get_maintenance_urgency(risk_score, active_failures)
        
        # CORRECTED: Get geographic operational area
        operational_area = self._get_operational_area()
        
        # Combine all data
        record = {
            'timestamp': timestamp,
            'voyage_phase': voyage_phase.value if hasattr(voyage_phase, 'value') else str(voyage_phase),
            'weather_condition': weather_condition,  # CORRECTED: Now varied
            'log_type': self._determine_log_type(timestamp),
            **main_engine_data,
            **auxiliary_data,
            **fuel_data,
            **cooling_data,
            **lubrication_data,
            **electrical_data,
            **maintenance_data,
            **human_data,
            'active_failures': active_failures,      # CORRECTED: String format
            'cascade_active': bool(active_failures), # True if any failure
            'risk_score': risk_score,
            'failure_progression_stage': self._get_failure_stage(risk_score),
            'maintenance_urgency': maintenance_urgency,  # CORRECTED: Now varied
            'operational_area': operational_area,    # CORRECTED: Geographic regions
            'vessel_id': getattr(self.voyage_plan, 'vessel_id', 'unknown'),
            'voyage_id': getattr(self.voyage_plan, 'voyage_id', 'unknown'),
            'cargo_type': getattr(self.voyage_plan, 'cargo_type', 'unknown')
        }
        
        return record
    
    def _get_failure_stage(self, risk_score: float) -> str:
        """Calculate failure progression stage based on risk score"""
        if risk_score > 0.200:
            return 'critical'
        elif risk_score > 0.150:
            return 'severe'
        elif risk_score > 0.100:
            return 'moderate'
        else:
            return 'normal'
    
    # NEW: Uncertainty injection methods
    def _process_uncertainty_step(self, timestamp: datetime, phase_info):
        """Process uncertainty injection for current timestamp (FIXED)"""
        
        # 1. Check for new failure initiations (Temporal Injection)
        self._check_failure_initiation(timestamp, phase_info)
        
        # 2. Update human factors (Human Factors)
        self._update_human_factors(timestamp, phase_info)
        
        # 3. Progress existing failures (Cascade Failure progression)
        self._progress_active_failures(timestamp, phase_info)
    
    def _check_failure_initiation(self, timestamp: datetime, phase_info):
        """Check if new failures should be initiated (FIXED)"""
        
        # Handle phase_info - it could be VoyagePhase object or dict
        if isinstance(phase_info, dict):
            voyage_phase = phase_info.get('phase', VoyagePhase.SEA_TRANSIT)
            weather_condition = phase_info.get('weather_condition', 'calm')
        else:
            voyage_phase = phase_info
            weather_condition = self._get_realistic_weather_condition()
        
        # Calculate dynamic failure probability
        phase_mult = self.phase_risk_multipliers.get(
            voyage_phase.value if hasattr(voyage_phase, 'value') else str(voyage_phase), 
            1.0
        )
        weather_mult = self.weather_risk_multipliers.get(weather_condition, 1.0)
        
        # Check each failure type
        failure_types = ['fuel_system', 'cooling', 'electrical', 'propulsion']
        
        for failure_type in failure_types:
            if failure_type not in self.active_failures:
                
                # Calculate failure probability for this step
                base_prob = self.base_failure_rates.get(failure_type, 0.01)
                vessel_mult = self.vessel_failure_multipliers.get(failure_type.split('_')[0], 1.0)
                
                # Convert daily probability to per-record probability
                daily_prob = base_prob * phase_mult * weather_mult * vessel_mult
                record_prob = daily_prob / self.logging_frequency  # Records per day
                
                if random.random() < record_prob:
                    # Initiate new failure
                    self._initiate_failure(failure_type, timestamp)
    
    def _initiate_failure(self, failure_type: str, timestamp: datetime):
        """Initiate a new failure scenario"""
        
        self.active_failures[failure_type] = {
            'initiated_at': timestamp,
            'severity': 0.1,  # Start with minor issues
            'progression_rate': random.uniform(0.05, 0.15),  # How fast it gets worse
            'stage': 'early_warning',
            'affected_parameters': self._get_affected_parameters(failure_type)
        }
        
        # Log the failure initiation
        self.failure_history.append({
            'timestamp': timestamp,
            'event': 'failure_initiated',
            'failure_type': failure_type,
            'severity': 0.1,
            'vessel_id': getattr(self.voyage_plan, 'vessel_id', 'unknown')
        })
        
        print(f"⚠️  {timestamp}: {failure_type} failure initiated (severity: 0.1)")
    
    def _get_affected_parameters(self, failure_type: str) -> List[str]:
        """Get list of parameters affected by each failure type"""
        
        parameter_mapping = {
            'fuel_system': [
                'fuel_quality_percentage', 'fuel_oil_pressure_bar', 
                'main_engine_fuel_consumption_mt_per_day'
            ],
            'cooling': [
                'cooling_water_flow_m3_per_hour', 'cooling_water_temperature_c',
                'cooling_system_efficiency_percentage', 'main_engine_operating_temperature_c'
            ],
            'electrical': [
                'generator_power_kw', 'generator_voltage_v', 'generator_frequency_hz',
                'power_factor', 'electrical_load_percentage'
            ],
            'propulsion': [
                'main_engine_power_kw', 'main_engine_rpm', 
                'main_engine_operating_temperature_c'
            ]
        }
        
        return parameter_mapping.get(failure_type, [])
    
    def _update_human_factors(self, timestamp: datetime, phase_info):
        """Update human performance factors (FIXED)"""
        
        # Crew fatigue accumulation (realistic maritime patterns)
        hour = timestamp.hour
        
        # Watch change periods (every 4 hours: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
        if hour % 4 == 0:
            # Reset fatigue during watch changes
            self.human_factors['fatigue_accumulation'] *= 0.7
        else:
            # Accumulate fatigue (~5% per day)
            self.human_factors['fatigue_accumulation'] += 0.002
        
        # Experience degradation during high-stress periods
        # Handle phase_info - it could be VoyagePhase object or dict
        if isinstance(phase_info, dict):
            voyage_phase = phase_info.get('phase', VoyagePhase.SEA_TRANSIT)
            weather_condition = phase_info.get('weather_condition', 'calm')
        else:
            voyage_phase = phase_info
            weather_condition = self._get_realistic_weather_condition()
        
        if voyage_phase in [VoyagePhase.LOADING, VoyagePhase.UNLOADING] or weather_condition == 'heavy_weather':
            self.human_factors['experience_degradation'] += 0.001
        
        # Procedure compliance affected by active failures and fatigue
        active_failure_count = len(self.active_failures)
        fatigue_impact = self.human_factors['fatigue_accumulation']
        
        self.human_factors['procedure_compliance'] = max(
            0.3, 
            0.8 - (active_failure_count * 0.05) - (fatigue_impact * 0.5)
        )
    
    def _progress_active_failures(self, timestamp: datetime, phase_info):
        """Progress existing failures and check for cascade effects (FIXED)"""
        
        failures_to_remove = []
        
        for failure_type, failure_data in list(self.active_failures.items()):
            
            # Progress failure severity
            failure_data['severity'] += failure_data['progression_rate']
            
            # Update failure stage
            if failure_data['severity'] > 0.8:
                failure_data['stage'] = 'critical'
            elif failure_data['severity'] > 0.6:
                failure_data['stage'] = 'severe'
            elif failure_data['severity'] > 0.3:
                failure_data['stage'] = 'moderate'
            else:
                failure_data['stage'] = 'early_warning'
            
            # Check for cascade triggers
            self._check_cascade_triggers(failure_type, failure_data, timestamp)
            
            # Possibility of recovery during maintenance
            if hasattr(self, '_is_maintenance_active') and self._is_maintenance_active(timestamp):
                if failure_data['severity'] > 0.4:
                    failure_data['severity'] *= 0.6  # Maintenance reduces severity
                    
                    if failure_data['severity'] < 0.1:
                        failures_to_remove.append(failure_type)
                        print(f"✅ {timestamp}: {failure_type} failure resolved through maintenance")
        
        # Remove resolved failures
        for failure_type in failures_to_remove:
            del self.active_failures[failure_type]
    
    def _check_cascade_triggers(self, failure_type: str, failure_data: Dict, timestamp: datetime):
        """Check for cascade failure triggers"""
        
        # Cascade logic based on maritime failure patterns
        cascade_mapping = {
            'cooling': 'propulsion',  # Cooling failure → engine derating
            'fuel_system': 'electrical',  # Fuel issues → generator problems
            'electrical': 'cooling',  # Electrical failure → cooling pump issues
            'propulsion': 'electrical'  # Main engine failure → electrical overload
        }
        
        # Trigger cascade if failure is severe and cascade target not already active
        if (failure_data['severity'] > 0.6 and 
            failure_type in cascade_mapping and 
            cascade_mapping[failure_type] not in self.active_failures):
            
            cascade_target = cascade_mapping[failure_type]
            
            # Initiate cascade failure with higher initial severity
            self.active_failures[cascade_target] = {
                'initiated_at': timestamp,
                'severity': 0.3,  # Start higher due to cascade
                'progression_rate': random.uniform(0.08, 0.20),  # Faster progression
                'stage': 'moderate',
                'affected_parameters': self._get_affected_parameters(cascade_target),
                'cascade_origin': failure_type
            }
            
            # Log cascade event
            self.failure_history.append({
                'timestamp': timestamp,
                'event': 'cascade_triggered',
                'failure_type': cascade_target,
                'origin_failure': failure_type,
                'severity': 0.3,
                'vessel_id': getattr(self.voyage_plan, 'vessel_id', 'unknown')
            })
            
            print(f"🔗 {timestamp}: Cascade failure {cascade_target} triggered by {failure_type}")
    
    def _apply_uncertainty_effects(self, record: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """Apply uncertainty effects to the operational record"""

        if not self.uncertainty_enabled:
            return record
        
        # Start with clean record
        modified_record = record.copy()
        
        # Track active failures for metadata
        active_failure_list = list(self.active_failures.keys())
        modified_record['active_failures'] = ','.join(active_failure_list)
        modified_record['cascade_active'] = any('cascade_origin' in f for f in self.active_failures.values())
        
        # Apply effects for each active failure
        for failure_type, failure_data in self.active_failures.items():
            severity = failure_data['severity']
            
            # Apply parameter modifications based on failure type
            if failure_type == 'fuel_system':
                # Fuel system degradation
                if 'fuel_quality_percentage' in modified_record:
                    modified_record['fuel_quality_percentage'] *= (1 - severity * 0.3)  # Up to 30% degradation
                if 'fuel_oil_pressure_bar' in modified_record:
                    modified_record['fuel_oil_pressure_bar'] *= (1 - severity * 0.2)   # Up to 20% pressure drop
                if 'main_engine_fuel_consumption_mt_per_day' in modified_record:
                    modified_record['main_engine_fuel_consumption_mt_per_day'] *= (1 + severity * 0.25)  # Increased consumption
                    
            elif failure_type == 'cooling':
                # Cooling system issues
                if 'cooling_water_flow_m3_per_hour' in modified_record:
                    modified_record['cooling_water_flow_m3_per_hour'] *= (1 - severity * 0.4)  # Up to 40% flow reduction
                if 'cooling_water_temperature_c' in modified_record:
                    modified_record['cooling_water_temperature_c'] += severity * 15  # Up to 15°C temperature rise
                if 'main_engine_operating_temperature_c' in modified_record:
                    modified_record['main_engine_operating_temperature_c'] += severity * 25  # Up to 25°C overheat
                    
            elif failure_type == 'electrical':
                # Electrical system instability
                if 'generator_voltage_v' in modified_record:
                    modified_record['generator_voltage_v'] *= (1 + random.uniform(-severity*0.1, severity*0.1))  # Voltage fluctuation
                if 'generator_frequency_hz' in modified_record:
                    modified_record['generator_frequency_hz'] *= (1 + random.uniform(-severity*0.05, severity*0.05))  # Frequency drift
                if 'power_factor' in modified_record:
                    modified_record['power_factor'] *= (1 - severity * 0.3)  # Power factor degradation
                    
            elif failure_type == 'propulsion':
                # Main engine performance degradation
                if 'main_engine_power_kw' in modified_record:
                    modified_record['main_engine_power_kw'] *= (1 - severity * 0.2)  # Up to 20% power loss
                if 'main_engine_rpm' in modified_record:
                    modified_record['main_engine_rpm'] *= (1 - severity * 0.15)  # RPM reduction
        
        # Apply human factors effects
        fatigue_effect = self.human_factors['fatigue_accumulation']
        compliance_effect = self.human_factors['procedure_compliance']
        
        if 'crew_familiarity_factor' in modified_record:
            modified_record['crew_familiarity_factor'] = max(0.3, modified_record['crew_familiarity_factor'] - fatigue_effect)
        if 'procedure_compliance_factor' in modified_record:
            modified_record['procedure_compliance_factor'] = compliance_effect
        
        # Calculate composite risk score
        modified_record['risk_score'] = self._calculate_risk_score()
        
        return modified_record
    
    def _calculate_risk_score(self) -> float:
        """Calculate composite risk score based on all active factors"""
        
        # Base risk from active failures
        failure_risk = sum(f['severity'] * 0.3 for f in self.active_failures.values())
        
        # Human factors risk
        human_risk = (self.human_factors['fatigue_accumulation'] * 0.2 + 
                     (1 - self.human_factors['procedure_compliance']) * 0.2)
        
        # Cascade amplification
        cascade_count = sum(1 for f in self.active_failures.values() if 'cascade_origin' in f)
        cascade_risk = cascade_count * 0.15
        
        total_risk = min(1.0, failure_risk + human_risk + cascade_risk)
        return round(total_risk, 3)
    
    def _add_uncertainty_metadata(self, df: pd.DataFrame):
        """Add uncertainty tracking columns to dataframe"""
        
        # Initialize columns if they don't exist
        if 'active_failures' not in df.columns:
            df['active_failures'] = ''
        if 'cascade_active' not in df.columns:
            df['cascade_active'] = False
        if 'risk_score' not in df.columns:
            df['risk_score'] = 0.0
        
        # Add failure progression stages
        df['failure_progression_stage'] = 'normal'
        df['maintenance_urgency'] = 'routine'
        
        # Update based on risk scores
        df.loc[df['risk_score'] > 0.8, 'failure_progression_stage'] = 'critical'
        df.loc[df['risk_score'] > 0.8, 'maintenance_urgency'] = 'immediate'
        df.loc[(df['risk_score'] > 0.6) & (df['risk_score'] <= 0.8), 'failure_progression_stage'] = 'severe'
        df.loc[(df['risk_score'] > 0.6) & (df['risk_score'] <= 0.8), 'maintenance_urgency'] = 'urgent'
        df.loc[(df['risk_score'] > 0.3) & (df['risk_score'] <= 0.6), 'failure_progression_stage'] = 'moderate'
        df.loc[(df['risk_score'] > 0.3) & (df['risk_score'] <= 0.6), 'maintenance_urgency'] = 'scheduled'
    
    # CORRECTED: Main engine data generation with extended load range
    def _generate_main_engine_data(self, timestamp: datetime, voyage_phase: VoyagePhase, weather_condition: str) -> Dict[str, Any]:
        """Generate main engine operational data (CORRECTED with full load range)"""
        
        # Load percentage based on voyage phase - CORRECTED RANGES
        if voyage_phase in [VoyagePhase.LOADING, VoyagePhase.UNLOADING]:
            base_load = 0.25 + random.uniform(0, 0.20)  # 25-45% for port operations
        elif voyage_phase == VoyagePhase.SEA_TRANSIT:
            base_load = 0.60 + random.uniform(0, 0.35)  # 60-95% for sea transit (CORRECTED)
        else:
            base_load = 0.40 + random.uniform(0, 0.30)  # 40-70% for other phases
        
        # Weather adjustments - CORRECTED with realistic multipliers
        weather_multipliers = {
            'calm': 0.95,           # Slightly lower load in calm conditions
            'moderate': 1.0,        # Baseline load
            'rough': 1.15,          # Higher load in rough weather
            'heavy_weather': 1.25,  # Much higher load
            'gale': 1.40           # Extreme load requirements
        }
        
        weather_mult = weather_multipliers.get(weather_condition, 1.0)
        final_load = min(0.95, base_load * weather_mult)  # Cap at 95% (CORRECTED)
        
        # Equipment condition affects performance
        condition_factor = self.equipment_condition["main_engine"]
        actual_load = final_load * condition_factor
        operational_variation = random.uniform(-0.01, 0.01)  # ±1% variation
        load_fluctuation = random.uniform(-0.005, 0.005)       # ±0.5% measurement noise
        actual_load = actual_load + operational_variation + load_fluctuation
        actual_load = min(0.95, max(0.35, actual_load))  # Final bounds check   
        # Calculate main engine parameters
        max_power = self.base_parameters["main_engine_power_kw"]
        actual_power = max_power * actual_load
        
        # RPM correlates with load
        base_rpm = 85  # RPM at full load
        actual_rpm = base_rpm * actual_load * (0.85 + 0.15 * condition_factor)
        
        # Operating temperature based on load with thermal lag
        base_temp = self.base_parameters["operating_temperature_base_c"]
        load_temp_increase = actual_load * 35  # Up to 35°C increase at full load
        actual_temp = base_temp + load_temp_increase * (1.1 - condition_factor * 0.1)
        
        # Fuel consumption based on load
        base_consumption = self.base_parameters["fuel_consumption_base_mt_per_day"]
        actual_consumption = base_consumption * (0.3 + 0.7 * actual_load) * (1.05 - condition_factor * 0.05)
        
        # Add realistic variations
        actual_power += random.uniform(-0.02, 0.02) * actual_power * self.precision_factor
        actual_rpm += random.uniform(-0.015, 0.015) * actual_rpm * self.precision_factor
        actual_temp += random.uniform(-1.5, 1.5) * self.precision_factor
        actual_consumption += random.uniform(-0.01, 0.01) * actual_consumption * self.precision_factor
        
        return {
            'main_engine_power_kw': round(actual_power, 1),
            'main_engine_rpm': round(actual_rpm, 1),
            'main_engine_fuel_consumption_mt_per_day': round(actual_consumption, 3),
            'main_engine_operating_temperature_c': round(actual_temp, 1),
            'main_engine_load_percentage': round(actual_load * 100, 1)
        }
    
    def _generate_auxiliary_engine_data(self, timestamp: datetime, voyage_phase: VoyagePhase) -> Dict[str, Any]:
        """Generate auxiliary engine data (UNCHANGED)"""
        
        # Auxiliary engine load based on electrical demand
        if voyage_phase in [VoyagePhase.LOADING, VoyagePhase.UNLOADING]:
            aux_load = 0.75  # High load for cargo operations
        elif voyage_phase == VoyagePhase.SEA_TRANSIT:
            aux_load = 0.55  # Medium load for ship services
        else:
            aux_load = 0.65  # Standard load for port operations
        
        # Equipment condition affects performance
        condition_factor = self.equipment_condition["auxiliary_engine"]
        actual_load = aux_load * condition_factor
        
        # Calculate auxiliary engine parameters
        max_aux_power = self.base_parameters["auxiliary_engine_power_kw"]
        actual_aux_power = max_aux_power * actual_load
        
        # Auxiliary fuel consumption
        base_aux_consumption = 2.5  # mt/day at full load
        actual_aux_consumption = base_aux_consumption * (0.2 + 0.8 * actual_load)
        
        # Add variations
        actual_aux_power += random.uniform(-0.02, 0.02) * actual_aux_power * self.precision_factor
        actual_aux_consumption += random.uniform(-0.01, 0.01) * actual_aux_consumption * self.precision_factor
        
        return {
            'auxiliary_engine_power_kw': round(actual_aux_power, 1),
            'auxiliary_engine_fuel_consumption_mt_per_day': round(actual_aux_consumption, 3),
            'auxiliary_engine_load_percentage': round(actual_load * 100, 1)
        }
    
    def _generate_fuel_system_data(self, timestamp: datetime, voyage_phase: VoyagePhase) -> Dict[str, Any]:
        """Generate fuel system data (UNCHANGED)"""
        
        # Base fuel oil pressure
        base_pressure = self.base_parameters["fuel_oil_pressure_base_bar"]
        
        # Pressure adjustments based on engine load
        if voyage_phase == VoyagePhase.SEA_TRANSIT:
            pressure_multiplier = 1.0  # Full pressure during transit
        elif voyage_phase in [VoyagePhase.LOADING, VoyagePhase.UNLOADING]:
            pressure_multiplier = 0.6  # Lower pressure during port operations
        else:
            pressure_multiplier = 0.8  # Medium pressure for other phases
        
        # Equipment condition affects fuel system performance
        condition_factor = self.equipment_condition["fuel_system"]
        actual_pressure = base_pressure * pressure_multiplier * condition_factor
        
        # Fuel oil temperature (heated for viscosity control)
        fuel_temp = 85 + random.uniform(-5, 10)  # Target 85°C with variations
        
        # Fuel quality (degrades with poor handling/storage)
        base_quality = 98.0  # High quality fuel
        quality_degradation = (1 - condition_factor) * 5  # Up to 5% degradation
        fuel_quality = base_quality - quality_degradation
        
        # Add variations
        actual_pressure += random.uniform(-0.2, 0.2) * self.precision_factor
        fuel_temp += random.uniform(-2.0, 2.0) * self.precision_factor
        fuel_quality += random.uniform(-1.0, 1.0) * self.precision_factor
        
        return {
            'fuel_oil_pressure_bar': round(actual_pressure, 2),
            'fuel_oil_temperature_c': round(fuel_temp, 1),
            'fuel_quality_percentage': round(fuel_quality, 1)
        }
    
    def _generate_cooling_system_data(self, timestamp: datetime, voyage_phase: VoyagePhase) -> Dict[str, Any]:
        """Generate cooling system data (UNCHANGED)"""
        
        # Base cooling water flow
        base_flow = self.base_parameters["cooling_water_flow_base_m3_per_hour"]
        
        # Adjust based on engine load (approximate from voyage phase)
        if voyage_phase == VoyagePhase.SEA_TRANSIT:
            flow_multiplier = 1.0  # Full flow during transit
        elif voyage_phase in [VoyagePhase.LOADING, VoyagePhase.UNLOADING]:
            flow_multiplier = 0.6  # Reduced flow during port operations
        else:
            flow_multiplier = 0.8  # Medium flow for other phases
        
        # Equipment condition affects efficiency
        condition_factor = self.equipment_condition["cooling_system"]
        actual_flow = base_flow * flow_multiplier * condition_factor
        
        # Cooling water temperature (seawater intake + heat exchange)
        seawater_temp = 15 + random.uniform(-3, 8)  # Seasonal variation
        heat_exchange_increase = (1 - condition_factor) * 10  # Poor condition = higher temp
        cooling_water_temp = seawater_temp + heat_exchange_increase
        
        # System efficiency
        base_efficiency = 95.0  # 95% efficiency when new
        actual_efficiency = base_efficiency * condition_factor
        
        # Add variations
        actual_flow += random.uniform(-0.02, 0.02) * actual_flow * self.precision_factor
        cooling_water_temp += random.uniform(-1.0, 1.0) * self.precision_factor
        actual_efficiency += random.uniform(-1.0, 1.0) * self.precision_factor
        
        return {
            'cooling_water_flow_m3_per_hour': round(actual_flow, 1),
            'cooling_water_temperature_c': round(cooling_water_temp, 1),
            'cooling_system_efficiency_percentage': round(actual_efficiency, 1)
        }
    
    def _generate_lubrication_system_data(self, timestamp: datetime, voyage_phase: VoyagePhase) -> Dict[str, Any]:
        """Generate lubrication system data (UNCHANGED)"""
        
        # Base lubrication oil pressure
        base_pressure = self.base_parameters["lubricating_oil_pressure_base_bar"]
        
        # Pressure varies with engine load
        if voyage_phase == VoyagePhase.SEA_TRANSIT:
            pressure_multiplier = 1.0  # Full pressure during transit
        elif voyage_phase in [VoyagePhase.LOADING, VoyagePhase.UNLOADING]:
            pressure_multiplier = 0.7  # Lower pressure during port operations
        else:
            pressure_multiplier = 0.85  # Medium pressure for other phases
        
        # Equipment condition affects pressure
        condition_factor = self.equipment_condition["lubrication_system"]
        actual_pressure = base_pressure * pressure_multiplier * condition_factor
        
        # Oil temperature correlates with engine load and condition
        base_oil_temp = 45  # Base lubrication oil temperature
        load_temp_increase = (pressure_multiplier - 0.7) * 20  # Temperature increase with load
        condition_temp_increase = (1 - condition_factor) * 8  # Poor condition = higher temp
        oil_temperature = base_oil_temp + load_temp_increase + condition_temp_increase
        
        # Daily oil consumption (higher with poor condition)
        base_consumption = 0.8  # Liters per day baseline
        condition_consumption_factor = 1 + (1 - condition_factor) * 0.5
        daily_consumption = base_consumption * condition_consumption_factor
        
        # Add variations
        actual_pressure += random.uniform(-0.1, 0.1) * self.precision_factor
        oil_temperature += random.uniform(-2.0, 2.0) * self.precision_factor
        daily_consumption += random.uniform(-0.05, 0.05) * self.precision_factor
        
        return {
            'lubricating_oil_pressure_bar': round(actual_pressure, 2),
            'lubricating_oil_temperature_c': round(oil_temperature, 1),
            'lubricating_oil_consumption_liters_per_day': round(daily_consumption, 2)
        }
    

    def _generate_electrical_system_data(self, timestamp, voyage_phase):
        """Generate electrical system data - COMPLETELY CORRECTED"""
        import random
    
        # 🔍 DEBUG: Print what values we're actually using
        engine_load = getattr(self, 'current_main_engine_load', None)
        print(f"DEBUG electrical generation:")
        print(f"  Timestamp: {timestamp}")
        print(f"  Voyage phase: {voyage_phase}")
        print(f"  Engine load from attribute: {engine_load}")
        
        # FIXED: Get main engine load from the record being generated
        if hasattr(self, 'current_main_engine_load') and self.current_main_engine_load is not None:
            engine_load = self.current_main_engine_load
            print(f"  Using stored engine load: {engine_load:.3f}")
        else:
            # Calculate engine load based on voyage phase (matching main engine logic)
            if str(voyage_phase) in ['LOADING', 'UNLOADING']:
                engine_load = 0.35  # 25-45% average
            elif str(voyage_phase) == 'SEA_TRANSIT':
                engine_load = 0.75  # 60-95% average
            else:
                engine_load = 0.55  # 40-70% average
            print(f"  Using calculated engine load: {engine_load:.3f}")
        
        # CORRECTED: Electrical load should INCREASE with engine load
        base_electrical_load = 0.45  # Base electrical load
        # Positive correlation: more engine load = more electrical demand
        operational_electrical_load = engine_load * 0.25  # Up to 25% additional
        
        print(f"  Base electrical load: {base_electrical_load}")
        print(f"  Operational electrical load: {operational_electrical_load:.3f}")
        
        # Phase-specific electrical demands
        phase_electrical_multipliers = {
            'LOADING': 1.8,     # Cargo handling equipment, lighting
            'UNLOADING': 1.8,   # Cargo handling equipment  
            'SEA_TRANSIT': 1.0, # Normal navigation systems
            'DEPARTURE': 1.3,   # Port navigation systems
            'ARRIVAL': 1.3      # Port navigation systems
        }
        
        # Handle VoyagePhase enum properly
        phase_str = str(voyage_phase).split('.')[-1] if hasattr(voyage_phase, 'value') else str(voyage_phase)
        phase_multiplier = phase_electrical_multipliers.get(phase_str, 1.0)
        
        print(f"  Phase string: '{phase_str}'")
        print(f"  Phase multiplier: {phase_multiplier}")
        
        # FIXED: Proper positive correlation
        actual_load = (base_electrical_load + operational_electrical_load) * phase_multiplier
        
        print(f"  Actual load before variation: {actual_load:.3f}")
        
        # FIXED: Add continuous variability to prevent discrete values
        operational_variation = random.uniform(-0.05, 0.05)  # ±5% variation
        load_fluctuation = random.uniform(-0.02, 0.02)       # ±2% measurement noise
        actual_load = actual_load + operational_variation + load_fluctuation
        
        actual_load = min(0.95, max(0.35, actual_load))  # Final bounds check 35-95%
        
        print(f"  Final electrical load: {actual_load:.3f} ({actual_load*100:.1f}%)")
        print(f"  Expected correlation: POSITIVE (more engine load = more electrical load)")
        
        # Generator power output
        max_generator_power = self.base_parameters['auxiliary_engine_power_kw'] * 0.9
        generator_power_kw = max_generator_power * actual_load
        
        # IMPROVED voltage regulation with realistic operational patterns
        nominal_voltage = 440  # Standard marine voltage
        condition_factor = self.equipment_condition.get('electrical_system', 1.0)
        
        # Base voltage varies with load (realistic generator characteristic)
        load_voltage_effect = (actual_load - 0.5) * 10  # ±5V around nominal based on load
        base_voltage = nominal_voltage + load_voltage_effect * condition_factor
        
        # Normal regulation: ±1% under good conditions
        regulation_variation = random.uniform(-0.01, 0.01) * condition_factor
        voltage = base_voltage * (1 + regulation_variation)
        
        # IMPROVED failure effects with proper bounds
        if hasattr(self, 'active_electrical_failures') and self.active_electrical_failures:
            max_failure_severity = max(self.active_electrical_failures)
            # Voltage drops 2-15% during failures, but never below 300V (safety limit)
            voltage_drop = 0.02 + (max_failure_severity * 0.13)  # 2-15% drop
            voltage = max(300, voltage * (1 - voltage_drop))
        
        # Final regulation bounds (±10% for marine systems)
        voltage = max(396, min(484, voltage))
        
        # CORRECTED: Realistic frequency generation (50Hz system)
        nominal_frequency = 50.0
        # Tight frequency regulation: ±0.5% under normal conditions  
        normal_freq_variation = random.uniform(-0.005, 0.005)
        frequency = nominal_frequency * (1 + normal_freq_variation) * condition_factor
        
        # Apply failure effects to frequency
        if hasattr(self, 'active_electrical_failures') and self.active_electrical_failures:
            max_failure_severity = max(self.active_electrical_failures)
            # Frequency instability under electrical failures
            freq_variation = max_failure_severity * 0.02  # Up to ±2% variation
            frequency *= (1 + random.uniform(-freq_variation, freq_variation))
        
        frequency = max(48.5, min(51.5, frequency))  # Strict marine frequency limits
        
        # CORRECTED: Realistic power factor calculation with bounds checking
        if actual_load < 0.3:
            base_pf = 0.85 - (0.3 - actual_load) * 0.5  # Lower PF at very light loads
        elif actual_load > 0.8:
            base_pf = 0.90 + (actual_load - 0.8) * 0.25  # Better PF at higher loads
        else:
            base_pf = 0.88  # Good power factor in normal operating range
        
        # Equipment condition affects power factor
        power_factor = base_pf * condition_factor
        
        # FIXED: Apply failure effects with proper bounds
        if hasattr(self, 'active_electrical_failures') and self.active_electrical_failures:
            for failure_severity in self.active_electrical_failures:
                # FIXED: Calculate degradation with proper bounds
                pf_degradation = min(0.95, failure_severity * 0.3)  # Cap degradation at 95%
                power_factor = max(0.05, power_factor * (1 - pf_degradation))  # Minimum 0.05 PF
        
        # CRITICAL FIX: Enforce power factor bounds (0.0 ≤ pf ≤ 1.0)
        power_factor = max(0.0, min(1.0, power_factor))
        
        # Add measurement precision variations
        generator_power_kw += random.uniform(-0.02, 0.02) * generator_power_kw * self.precision_factor
        voltage += random.uniform(-2.0, 2.0) * self.precision_factor  
        frequency += random.uniform(-0.1, 0.1) * self.precision_factor
        power_factor += random.uniform(-0.01, 0.01) * self.precision_factor
        
        # Final bounds checking after measurement noise
        power_factor = max(0.0, min(1.0, power_factor))
        voltage = max(396, min(484, voltage))
        frequency = max(48.5, min(51.5, frequency))
        
        return {
            'generator_power_kw': round(generator_power_kw, 1),
            'generator_voltage_v': round(voltage, 1),
            'generator_frequency_hz': round(frequency, 2),
            'power_factor': round(power_factor, 3),
            'electrical_load_percentage': round(actual_load * 100, 1)
        }

   

    # Additional helper method for electrical failure tracking - FIXED ATTRIBUTE MAPPING
    def track_electrical_failures(self, timestamp, current_failures):
        """
        Track electrical system failures and their severity progression
        """
        self.active_electrical_failures = []

        if 'electrical' in str(current_failures).lower():
            # Electrical failures progress over time
            failure_age_hours = getattr(self, 'electrical_failure_start', 0)
            if failure_age_hours == 0:
                self.electrical_failure_start = timestamp

            # Severity increases over time if not addressed  
            hours_elapsed = (timestamp - self.electrical_failure_start).total_seconds() / 3600
            severity = min(0.8, 0.1 + hours_elapsed * 0.02)  # Starts at 10%, increases 2%/hour

            self.active_electrical_failures.append(severity)

        return self.active_electrical_failures
    
   
    def _generate_maintenance_flags(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate maintenance and inspection flags (UNCHANGED)"""
        
        maintenance_flags = {
            'maintenance_due': False,
            'inspection_due': False,
            'maintenance_type': None,
            'inspection_type': None
        }
        
        # Check for scheduled maintenance based on voyage plan
        # This is a simplified version - in reality would check actual maintenance schedules
        
        # Monthly maintenance check
        if timestamp.day == 1:
            maintenance_flags['maintenance_due'] = True
            maintenance_flags['maintenance_type'] = 'monthly_inspection'
        
        # Weekly inspection
        if timestamp.weekday() == 0:  # Monday
            maintenance_flags['inspection_due'] = True
            maintenance_flags['inspection_type'] = 'weekly_check'
        
        return maintenance_flags
    
    def _generate_human_factors_data(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate human factors data with corrected baseline values for risk assessment modeling"""
        
        # CORRECTED: Realistic baseline values for operational risk assessment
        vessel_profile_baselines = {
            'Modern Major': {
                'crew_familiarity': 0.85,      # Well-trained, stable crews
                'base_compliance': 0.92        # High regulatory compliance
            },
            'Aging Midtier': {
                'crew_familiarity': 0.80,      # Experienced but rotating crews
                'base_compliance': 0.88        # Good compliance standards
            },
            'Legacy Small': {
                'crew_familiarity': 0.75,      # Mixed experience levels
                'base_compliance': 0.85        # Acceptable compliance
            }
        }
        
        # Get baseline values for this vessel's profile
        profile_name = self.vessel_profile.profile_name
        baseline_familiarity = vessel_profile_baselines.get(profile_name, {}).get('crew_familiarity', 0.80)
        baseline_compliance = vessel_profile_baselines.get(profile_name, {}).get('base_compliance', 0.88)
        
        # Time-based variations (watch schedule effects)
        hour = timestamp.hour
        
        # Watch schedule effects (4-hour watches) - REDUCED impact for realistic operations
        if hour % 4 == 0:
            # Fresh watch - higher performance
            familiarity_modifier = 1.0
            compliance_modifier = 1.0
        elif hour % 4 == 3:
            # End of watch - slight performance decrease (REDUCED from original)
            familiarity_modifier = 0.95      # Was 0.9, now 0.95
            compliance_modifier = 0.96       # Was 0.85, now 0.96
        else:
            # Mid-watch - standard performance (IMPROVED from original)
            familiarity_modifier = 0.98      # Was 0.95, now 0.98
            compliance_modifier = 0.98       # Was 0.9, now 0.98
        
        # Apply modifiers to baseline values
        actual_familiarity = baseline_familiarity * familiarity_modifier
        actual_compliance = baseline_compliance * compliance_modifier
        
        # Ensure values stay within realistic bounds
        actual_familiarity = max(0.65, min(1.0, actual_familiarity))  # Floor at 0.65
        actual_compliance = max(0.75, min(1.0, actual_compliance))    # Floor at 0.75
        
        # Determine log entry quality based on improved human factors
        if actual_familiarity > 0.82 and actual_compliance > 0.90:
            log_quality = 'excellent'
        elif actual_familiarity > 0.75 and actual_compliance > 0.85:
            log_quality = 'good'
        elif actual_familiarity > 0.70 and actual_compliance > 0.80:
            log_quality = 'standard'
        else:
            log_quality = 'poor'
        
        return {
            'crew_familiarity_factor': round(actual_familiarity, 2),
            'procedure_compliance_factor': round(actual_compliance, 2),
            'log_entry_quality': log_quality
        }
    
    def _determine_log_type(self, timestamp: datetime) -> str:
        """Determine type of log entry based on timing (UNCHANGED)"""
        
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Watch changes (every 4 hours)
        if hour % 4 == 0 and minute == 0:
            return 'watch_change'
        # Hourly routine checks
        elif minute == 0:
            return 'hourly_check'
        # Half-hourly monitoring
        elif minute == 30:
            return 'routine_monitoring'
        # 15-minute checks during critical operations
        elif minute in [15, 45]:
            return 'frequent_monitoring'
        else:
            return 'standard_log'
    
    def _is_maintenance_active(self, timestamp: datetime) -> bool:
        """Check if maintenance is currently active (helper method for uncertainty)"""
        
        # Simplified maintenance detection
        # In real implementation, this would check the voyage plan maintenance schedule
        return timestamp.hour in [2, 3, 14, 15]  # Maintenance windows
    
    # NEW: Utility methods for external access
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
                    'initiated_at': data['initiated_at'].isoformat(),
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
                'timestamp': event['timestamp'].isoformat(),
                'event': event['event'],
                'failure_type': event['failure_type'],
                'severity': event['severity'],
                'vessel_id': event['vessel_id']
            }
            for event in self.failure_history
        ]
