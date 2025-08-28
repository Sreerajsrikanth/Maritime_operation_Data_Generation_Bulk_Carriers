"""
Enhanced Auxiliary Support Systems Generator with Uncertainty Injection
generators/auxiliary_safety.py

Industry-validated implementation for maritime vessel auxiliary systems
with optional uncertainty injection for proactive risk assessment training.
Maintains compatibility with existing fleet generation framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import existing core systems
from core.voyage_plan import VoyagePlan
from core.vessel_profile import VesselProfile, CompanyType, AutomationLevel, CrewStability


class FailureStage(Enum):
    """Failure progression stages for auxiliary systems"""
    NORMAL = "normal"
    EARLY_WARNING = "early_warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AuxiliaryEquipmentState:
    """Tracks auxiliary equipment condition and failure progression"""
    ballast_pump_efficiency: float = 0.90
    fire_detection_reliability: float = 0.95
    emergency_equipment_condition: float = 0.92
    bilge_pump_reliability: float = 0.88
    co2_system_pressure_ratio: float = 1.0
    
    # Failure tracking
    active_failures: Dict[str, Dict] = None
    days_since_last_inspection: int = 0
    
    def __post_init__(self):
        if self.active_failures is None:
            self.active_failures = {}


@dataclass
class VesselStabilityState:
    """Tracks vessel stability and ballast status"""
    cargo_loaded_mt: float
    ballast_water_mt: float
    total_displacement_mt: float
    stability_margin: float
    list_angle_deg: float
    

@dataclass 
class SafetySystemStatus:
    """Tracks safety equipment operational status"""
    fire_detection_status: str
    bilge_alarm_status: str
    emergency_equipment_status: str
    last_safety_drill: datetime
    safety_compliance_rating: float


class CrewTimingSimulator:
    """Simulates realistic crew timing variations for auxiliary & safety operations"""
    
    @staticmethod
    def generate_crew_timestamp(day: int, hour: int, 
                               task_duration_mins: Tuple[int, int] = (15, 35),
                               handover_delay: bool = False) -> datetime:
        """Generate realistic timestamp with crew timing variations"""
        
        # Base timestamp
        base_time = datetime(2024, 1, day, hour, 0)
        
        # Add task duration variation
        duration_variation = random.randint(*task_duration_mins)
        
        # Add handover delay if watch change
        handover_variation = random.randint(10, 25) if handover_delay else 0
        
        # Total delay
        total_delay = duration_variation + handover_variation
        
        return base_time + timedelta(minutes=total_delay)
    
    @staticmethod
    def is_handover_time(hour: int) -> bool:
        """Check if hour is a watch handover time (every 4 hours)"""
        return hour in [0, 4, 8, 12, 16, 20]


class EnhancedAuxiliarySafetySystemsGenerator:
    """
    Enhanced Auxiliary & Safety Systems Generator with Uncertainty Injection
    
    Generates synthetic data for pump room operations and safety systems
    with optional uncertainty injection for proactive risk assessment training.
    Industry-validated integration under Engine Department oversight.
    """
    
    def __init__(self, voyage_plan: VoyagePlan, uncertainty_enabled: bool = False):
        self.voyage_plan = voyage_plan
        self.vessel_profile = voyage_plan.vessel_profile
        self.timing_simulator = CrewTimingSimulator()
        self.uncertainty_enabled = uncertainty_enabled
        
        # Initialize equipment state
        self.equipment_state = AuxiliaryEquipmentState()
        
        # Initialize vessel stability state
        self.stability_state = VesselStabilityState(
            cargo_loaded_mt=0.0,  # Start empty
            ballast_water_mt=25000.0,  # Full ballast when empty
            total_displacement_mt=45000.0,  # Typical bulk carrier
            stability_margin=0.8,
            list_angle_deg=0.2
        )
        
        # Initialize safety system status
        self.safety_status = SafetySystemStatus(
            fire_detection_status='operational',
            bilge_alarm_status='operational', 
            emergency_equipment_status='operational',
            last_safety_drill=voyage_plan.start_date - timedelta(days=10),
            safety_compliance_rating=4.2
        )
        
        # Apply vessel profile adjustments
        self._apply_vessel_profile_adjustments()
        
        # Initialize pump operational parameters
        self.pump_specifications = {
            'ballast_pump_capacity_m3h': 800 * self.pump_efficiency_factor,
            'bilge_pump_capacity_m3h': 200 * self.pump_efficiency_factor,
            'fuel_transfer_capacity_m3h': 150 * self.pump_efficiency_factor,
            'max_operating_pressure_bar': 8.5,
            'power_consumption_kw': 185 * self.power_efficiency_factor
        }
        
        # Uncertainty injection components
        if self.uncertainty_enabled:
            self._initialize_uncertainty_components()
    
    def _apply_vessel_profile_adjustments(self):
        """Apply vessel profile variations to auxiliary and safety systems"""
        profile_name = self.vessel_profile.profile_name.lower()
        
        # Use actual vessel profile attributes
        if hasattr(self.vessel_profile, 'company_type'):
            if self.vessel_profile.company_type == CompanyType.MAJOR:
                self.safety_equipment_quality = 4.5
                self.inspection_frequency_multiplier = 1.0
                self.base_failure_multiplier = 0.7  # Lower failure rates
            elif self.vessel_profile.company_type == CompanyType.SMALL:
                self.safety_equipment_quality = 3.2
                self.inspection_frequency_multiplier = 1.2  # More frequent due to lower reliability
                self.base_failure_multiplier = 1.3  # Higher failure rates
            else:  # MIDTIER
                self.safety_equipment_quality = 3.8
                self.inspection_frequency_multiplier = 1.1
                self.base_failure_multiplier = 1.0  # Baseline
        else:
            # Fallback to profile name parsing
            if 'modern' in profile_name or 'major' in profile_name:
                self.safety_equipment_quality = 4.5
                self.inspection_frequency_multiplier = 1.0
                self.base_failure_multiplier = 0.7
            elif 'legacy' in profile_name or 'small' in profile_name:
                self.safety_equipment_quality = 3.2
                self.inspection_frequency_multiplier = 1.2
                self.base_failure_multiplier = 1.3
            else:
                self.safety_equipment_quality = 3.8
                self.inspection_frequency_multiplier = 1.1
                self.base_failure_multiplier = 1.0
        
        # Set automation and efficiency based on vessel profile
        if hasattr(self.vessel_profile, 'automation_level'):
            if self.vessel_profile.automation_level == AutomationLevel.HIGH:
                self.automation_level = 0.85
                self.pump_efficiency_factor = 1.2
                self.power_efficiency_factor = 0.9  # More efficient
            elif self.vessel_profile.automation_level == AutomationLevel.LOW:
                self.automation_level = 0.35
                self.pump_efficiency_factor = 0.8
                self.power_efficiency_factor = 1.3  # Less efficient
            else:  # MEDIUM
                self.automation_level = 0.65
                self.pump_efficiency_factor = 1.0
                self.power_efficiency_factor = 1.0
        else:
            # Fallback
            self.automation_level = 0.65
            self.pump_efficiency_factor = 1.0
            self.power_efficiency_factor = 1.0
    
    def _initialize_uncertainty_components(self):
        """Initialize uncertainty injection components (only if uncertainty_enabled=True)"""
        
        # Industry-validated failure rates (per day)
        self.base_failure_rates = {
            'ballast_pump_degradation': 0.018 * self.base_failure_multiplier,
            'fire_detection_failure': 0.012 * self.base_failure_multiplier,
            'emergency_equipment_failure': 0.008 * self.base_failure_multiplier,
            'bilge_pump_failure': 0.015 * self.base_failure_multiplier,
            'safety_inspection_quality_decline': 0.020 * self.base_failure_multiplier
        }
        
        # Risk accumulation factors
        self.risk_factors = {
            'equipment_condition': 0.35,
            'operational_stress': 0.25,
            'maintenance_overdue': 0.25,
            'human_factors': 0.15
        }
        
        # Human factors state
        self.human_factors = {
            'crew_fatigue_level': 0.1,
            'inspection_thoroughness': 0.9,
            'drill_performance_trend': 0.0,
            'procedure_compliance': 0.85
        }
        
        # Failure history for analytics
        self.failure_history = []
        self.uncertainty_metadata = {
            'total_failures_injected': 0,
            'failure_types_encountered': set(),
            'peak_risk_score': 0.0,
            'cascade_events': []
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete auxiliary & safety systems dataset for voyage
        Integrates pump room operations with safety system monitoring
        """
        all_data = []
        voyage_start = self.voyage_plan.start_date
        voyage_duration = (self.voyage_plan.end_date - self.voyage_plan.start_date).days
        
        print(f"âš¡ Generating Auxiliary & Safety Systems data for {voyage_duration}-day voyage...")
        if self.uncertainty_enabled:
            print(f"ðŸ”„ Uncertainty injection ENABLED - generating risk scenarios")
        
        for day in range(voyage_duration):
            current_date = voyage_start + timedelta(days=day)
            current_phase_enum = self.voyage_plan.get_current_phase(current_date)
            current_phase = {'phase': current_phase_enum.value}
            
            # Apply uncertainty effects if enabled
            if self.uncertainty_enabled:
                self._apply_daily_uncertainties(day, current_phase)
            
            # Update vessel stability based on cargo operations
            self._update_vessel_stability(day, current_phase)
            
            # Generate ballast pump operations (event-driven by cargo loading)
            if current_phase['phase'] in ['loading', 'unloading']:
                # High activity during cargo operations
                for pump_hour in [2, 6, 10, 14, 18, 22]:  # Every 4 hours
                    ballast_data = self._generate_ballast_operation_data(
                        current_date, pump_hour, current_phase, day
                    )
                    all_data.append(ballast_data)
            else:
                # Minimal activity during sea transit
                for pump_hour in [8, 20]:  # Twice daily monitoring
                    ballast_data = self._generate_ballast_monitoring_data(
                        current_date, pump_hour, current_phase, day
                    )
                    all_data.append(ballast_data)
            
            # Generate safety system monitoring (every 6 hours)
            for safety_hour in [6, 12, 18, 24]:
                safety_data = self._generate_safety_monitoring_data(
                    current_date, safety_hour % 24, current_phase, day
                )
                all_data.append(safety_data)
            
            # Generate weekly safety inspections (SOLAS compliance)
            if day % 7 == 0:
                inspection_data = self._generate_safety_inspection_data(
                    current_date, 9, current_phase, day  # 09:00 AM inspection
                )
                all_data.append(inspection_data)
            
            # Generate monthly emergency drill (FIXED: Add realistic schedule variability)
            base_drill_interval = 30
            # Add operational variability: Â±5 days, avoid loading/unloading if possible
            schedule_variance = random.randint(-5, 5)
            actual_drill_interval = base_drill_interval + schedule_variance
            
            # Check if we're in a busy operational phase and try to reschedule
            if (current_phase['phase'] in ['loading', 'unloading'] and 
                abs(schedule_variance) < 3):  # Small variance, try to avoid busy periods
                schedule_variance += 3 if schedule_variance >= 0 else -3
                actual_drill_interval = base_drill_interval + schedule_variance
            
            if day % actual_drill_interval == 15:  # Mid-voyage drill with variance
                drill_data = self._generate_emergency_drill_data(
                    current_date, 14, current_phase, day  # 14:00 PM drill
                )
                all_data.append(drill_data)
        
        # Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(all_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"âœ… Auxiliary & Safety Systems dataset generated: {len(df)} records")
        if self.uncertainty_enabled:
            print(f"ðŸŽ¯ Uncertainty summary: {self.uncertainty_metadata['total_failures_injected']} failure scenarios injected")
        
        return df
    
    def _apply_daily_uncertainties(self, day: int, current_phase: Dict):
        """Apply daily uncertainty progression and new failure scenarios"""
        
        # Progress existing failures
        self._progress_active_failures(day, current_phase)
        
        # Check for new failure initiations
        self._check_new_failure_initiations(day, current_phase)
        
        # Update human factors
        self._update_human_factors(day, current_phase)
        
        # Update equipment degradation
        self._update_equipment_degradation(day)
        
        # Update inspection schedule drift
        self.equipment_state.days_since_last_inspection += 1
    
    def _progress_active_failures(self, day: int, current_phase: Dict):
        """Progress existing failures and check for cascade effects"""
        
        failures_to_remove = []
        
        for failure_type, failure_data in self.equipment_state.active_failures.items():
            
            # Progress failure severity
            failure_data['severity'] += failure_data['progression_rate']
            
            # Update failure stage
            if failure_data['severity'] > 0.8:
                failure_data['stage'] = FailureStage.CRITICAL.value
            elif failure_data['severity'] > 0.6:
                failure_data['stage'] = FailureStage.DEGRADED.value
            elif failure_data['severity'] > 0.3:
                failure_data['stage'] = FailureStage.EARLY_WARNING.value
            else:
                failure_data['stage'] = FailureStage.NORMAL.value
            
            # Check for cascade triggers
            self._check_cascade_triggers(failure_type, failure_data, day)
            
            # Possibility of recovery during port (maintenance opportunity)
            if current_phase['phase'] in ['loading', 'unloading']:
                if random.random() < 0.1:  # 10% chance of maintenance during port
                    failure_data['severity'] *= 0.6  # Maintenance reduces severity
                    
                    if failure_data['severity'] < 0.1:
                        failures_to_remove.append(failure_type)
                        print(f"âœ… Day {day}: {failure_type} failure resolved through port maintenance")
        
        # Remove resolved failures
        for failure_type in failures_to_remove:
            del self.equipment_state.active_failures[failure_type]
    
    def _check_new_failure_initiations(self, day: int, current_phase: Dict):
        """Check for new failure scenario initiations"""
        
        # Phase-specific risk multipliers
        phase_multiplier = self._get_phase_risk_multiplier(current_phase['phase'])
        
        for failure_type, base_rate in self.base_failure_rates.items():
            
            # Skip if failure already active
            if failure_type in self.equipment_state.active_failures:
                continue
            
            # Calculate adjusted failure probability
            adjusted_rate = base_rate * phase_multiplier
            
            # Check for failure initiation
            if random.random() < adjusted_rate:
                self._initiate_failure_scenario(failure_type, day)
    
    def _initiate_failure_scenario(self, failure_type: str, day: int):
        """Initiate a new failure scenario"""
        
        failure_data = {
            'initiated_day': day,
            'severity': 0.1,  # Start with low severity
            'stage': FailureStage.EARLY_WARNING.value,
            'progression_rate': random.uniform(0.05, 0.15),  # Daily progression
            'root_cause': self._determine_failure_root_cause(failure_type)
        }
        
        self.equipment_state.active_failures[failure_type] = failure_data
        
        # Update metadata
        self.uncertainty_metadata['total_failures_injected'] += 1
        self.uncertainty_metadata['failure_types_encountered'].add(failure_type)
        
        # Log failure event
        self.failure_history.append({
            'day': day,
            'failure_type': failure_type,
            'action': 'initiated',
            'severity': failure_data['severity']
        })
        
        print(f"âš ï¸ Day {day}: {failure_type} initiated - {failure_data['root_cause']}")
    
    def _determine_failure_root_cause(self, failure_type: str) -> str:
        """Determine realistic root cause for failure type"""
        
        root_causes = {
            'ballast_pump_degradation': random.choice([
                'bearing_wear', 'impeller_erosion', 'seal_degradation', 'vibration_misalignment'
            ]),
            'fire_detection_failure': random.choice([
                'detector_contamination', 'wiring_corrosion', 'calibration_drift', 'sensor_aging'
            ]),
            'emergency_equipment_failure': random.choice([
                'seal_deterioration', 'battery_degradation', 'mechanical_wear', 'corrosion_damage'
            ]),
            'bilge_pump_failure': random.choice([
                'clogged_strainer', 'motor_overheating', 'valve_sticking', 'float_switch_failure'
            ]),
            'safety_inspection_quality_decline': random.choice([
                'crew_fatigue', 'time_pressure', 'insufficient_training', 'equipment_familiarity_loss'
            ])
        }
        
        return root_causes.get(failure_type, 'unknown_cause')
    
    def _check_cascade_triggers(self, failure_type: str, failure_data: Dict, day: int):
        """Check for cascade failure triggers"""
        
        # Only trigger cascades for severe failures
        if failure_data['severity'] < 0.7:
            return
        
        # Cascade logic based on auxiliary system dependencies
        cascade_mapping = {
            'ballast_pump_degradation': 'bilge_pump_failure',  # Shared hydraulic/electrical systems
            'fire_detection_failure': 'emergency_equipment_failure',  # Fire system dependencies
            'bilge_pump_failure': 'ballast_pump_degradation',  # Pump room flooding risk
        }
        
        cascade_target = cascade_mapping.get(failure_type)
        
        if (cascade_target and 
            cascade_target not in self.equipment_state.active_failures and
            random.random() < 0.3):  # 30% chance of cascade
            
            self._initiate_failure_scenario(cascade_target, day)
            
            # Log cascade event
            self.uncertainty_metadata['cascade_events'].append({
                'day': day,
                'trigger': failure_type,
                'cascade_target': cascade_target
            })
            
            print(f"ðŸ”— Day {day}: Cascade failure - {failure_type} â†’ {cascade_target}")
    
    def _get_phase_risk_multiplier(self, phase: str) -> float:
        """Get risk multiplier based on operational phase"""
        
        multipliers = {
            'loading': 2.5,      # High ballast/safety system stress
            'unloading': 2.0,    # Moderate ballast system stress
            'sea_transit': 1.0,  # Baseline risk
            'port_approach': 1.5,# Increased activity
            'port_departure': 1.3 # Moderate activity
        }
        
        return multipliers.get(phase, 1.0)
    
    def _update_human_factors(self, day: int, current_phase: Dict):
        """Update human factors that affect auxiliary system operations"""
        
        # Crew fatigue accumulation
        fatigue_increase = random.uniform(0.01, 0.03)  # 1-3% daily increase
        if current_phase['phase'] in ['loading', 'unloading']:
            fatigue_increase *= 1.5  # Higher fatigue during port operations
        
        self.human_factors['crew_fatigue_level'] = min(0.8, 
            self.human_factors['crew_fatigue_level'] + fatigue_increase)
        
        # Port recovery effect
        if current_phase['phase'] in ['loading', 'unloading'] and random.random() < 0.3:
            self.human_factors['crew_fatigue_level'] *= 0.9  # Some recovery during port
        
        # Inspection thoroughness affected by fatigue and time pressure
        active_failure_count = len(self.equipment_state.active_failures)
        self.human_factors['inspection_thoroughness'] = max(0.5,
            0.9 - (self.human_factors['crew_fatigue_level'] * 0.3) - (active_failure_count * 0.05))
        
        # Procedure compliance affected by fatigue and active failures
        self.human_factors['procedure_compliance'] = max(0.4,
            0.85 - (self.human_factors['crew_fatigue_level'] * 0.4) - (active_failure_count * 0.03))
    
    def _update_equipment_degradation(self, day: int):
        """Update equipment condition based on active failures"""
        
        # Ballast pump efficiency
        if 'ballast_pump_degradation' in self.equipment_state.active_failures:
            severity = self.equipment_state.active_failures['ballast_pump_degradation']['severity']
            self.equipment_state.ballast_pump_efficiency = max(0.6, 0.90 - (severity * 0.3))
        else:
            # Gradual recovery when no active failures
            self.equipment_state.ballast_pump_efficiency = min(0.90,
                self.equipment_state.ballast_pump_efficiency + 0.005)
        
        # Fire detection reliability
        if 'fire_detection_failure' in self.equipment_state.active_failures:
            severity = self.equipment_state.active_failures['fire_detection_failure']['severity']
            self.equipment_state.fire_detection_reliability = max(0.7, 0.95 - (severity * 0.25))
        else:
            self.equipment_state.fire_detection_reliability = min(0.95,
                self.equipment_state.fire_detection_reliability + 0.003)
        
        # Emergency equipment condition
        if 'emergency_equipment_failure' in self.equipment_state.active_failures:
            severity = self.equipment_state.active_failures['emergency_equipment_failure']['severity']
            self.equipment_state.emergency_equipment_condition = max(0.6, 0.92 - (severity * 0.32))
        else:
            self.equipment_state.emergency_equipment_condition = min(0.92,
                self.equipment_state.emergency_equipment_condition + 0.004)
        
        # Bilge pump reliability
        if 'bilge_pump_failure' in self.equipment_state.active_failures:
            severity = self.equipment_state.active_failures['bilge_pump_failure']['severity']
            self.equipment_state.bilge_pump_reliability = max(0.5, 0.88 - (severity * 0.38))
        else:
            self.equipment_state.bilge_pump_reliability = min(0.88,
                self.equipment_state.bilge_pump_reliability + 0.006)
    
    def _calculate_risk_score(self, day: int, current_phase: Dict) -> float:
        """Calculate composite risk score for auxiliary systems"""
        
        if not self.uncertainty_enabled:
            return 0.0  # No risk calculation for clean data
        
        # Equipment condition risk (35% weight)
        equipment_risk = (
            (1 - self.equipment_state.ballast_pump_efficiency) * 0.4 +
            (1 - self.equipment_state.fire_detection_reliability) * 0.3 +
            (1 - self.equipment_state.emergency_equipment_condition) * 0.2 +
            (1 - self.equipment_state.bilge_pump_reliability) * 0.1
        )
        
        # Operational stress risk (25% weight)
        phase_stress = self._get_phase_risk_multiplier(current_phase['phase']) / 2.5  # Normalize
        active_failure_stress = min(1.0, len(self.equipment_state.active_failures) * 0.25)
        operational_risk = (phase_stress + active_failure_stress) / 2
        
        # Maintenance overdue risk (25% weight)
        days_overdue = max(0, self.equipment_state.days_since_last_inspection - 7)
        maintenance_risk = min(1.0, days_overdue / 14)  # Max risk after 3 weeks
        
        # Human factors risk (15% weight)
        human_risk = (
            self.human_factors['crew_fatigue_level'] * 0.4 +
            (1 - self.human_factors['inspection_thoroughness']) * 0.3 +
            (1 - self.human_factors['procedure_compliance']) * 0.3
        )
        
        # Composite risk score
        risk_score = (
            equipment_risk * self.risk_factors['equipment_condition'] +
            operational_risk * self.risk_factors['operational_stress'] +
            maintenance_risk * self.risk_factors['maintenance_overdue'] +
            human_risk * self.risk_factors['human_factors']
        )
        
        # Track peak risk
        self.uncertainty_metadata['peak_risk_score'] = max(
            self.uncertainty_metadata['peak_risk_score'], risk_score)
        
        return min(1.0, risk_score)
    
    def _update_vessel_stability(self, day: int, current_phase: Dict):
        """Update vessel stability state based on cargo operations"""
        if current_phase['phase'] == 'loading':
            # Cargo increases, ballast decreases
            loading_rate = 1200  # MT per hour typical bulk loading
            hours_loaded = min(day * 24, 60)  # Assume 60 hours max loading
            self.stability_state.cargo_loaded_mt = hours_loaded * loading_rate
            
            # Ballast adjustment for stability (inverse relationship)
            max_ballast = 25000
            current_ballast_ratio = 1 - (self.stability_state.cargo_loaded_mt / 75000)  # Max cargo 75,000 MT
            self.stability_state.ballast_water_mt = max_ballast * max(0.1, current_ballast_ratio)
            
        elif current_phase['phase'] == 'unloading':
            # Cargo decreases, ballast increases
            remaining_cargo_ratio = max(0, 1 - (day - 25) * 0.15)  # Assume unloading starts day 25
            self.stability_state.cargo_loaded_mt = 75000 * remaining_cargo_ratio
            
            # Increase ballast as cargo decreases
            ballast_ratio = 1 - remaining_cargo_ratio
            self.stability_state.ballast_water_mt = 25000 * (0.1 + 0.9 * ballast_ratio)
        
        # Update stability margin based on loading condition and equipment failures
        base_stability = 0.6 if self.stability_state.cargo_loaded_mt > 0 else 0.8
        
        # Adjust for equipment failures if uncertainty enabled
        if self.uncertainty_enabled and 'ballast_pump_degradation' in self.equipment_state.active_failures:
            failure_severity = self.equipment_state.active_failures['ballast_pump_degradation']['severity']
            stability_reduction = failure_severity * 0.2  # Up to 20% reduction
            base_stability *= (1 - stability_reduction)
        
        self.stability_state.stability_margin = base_stability + random.uniform(-0.1, 0.1)
    
    def _generate_ballast_operation_data(self, date: datetime, hour: int, phase: Dict, day: int) -> Dict:
        """Generate active ballast pump operation data during cargo operations"""
        actual_time = self.timing_simulator.generate_crew_timestamp(
            day + 1, hour, (15, 35), self.timing_simulator.is_handover_time(hour)
        )
        
        # High pump activity during cargo operations
        pump_running = random.choice([True, True, True, False])  # 75% running
        
        if pump_running:
            base_flow_rate = self.pump_specifications['ballast_pump_capacity_m3h'] * random.uniform(0.6, 0.95)
            # Apply equipment degradation if uncertainty enabled
            if self.uncertainty_enabled:
                flow_rate = base_flow_rate * self.equipment_state.ballast_pump_efficiency
            else:
                flow_rate = base_flow_rate
            
            power_consumption = self.pump_specifications['power_consumption_kw'] * (flow_rate / self.pump_specifications['ballast_pump_capacity_m3h'])
        else:
            flow_rate = 0
            power_consumption = 0
        
        # Generate base pressure with uncertainty variations and improved pump physics
        base_pressure = 6.2 if pump_running else 0
        if self.uncertainty_enabled and pump_running:
            # Apply pressure variations based on equipment condition
            pressure_factor = self.equipment_state.ballast_pump_efficiency
            # IMPROVED: Strengthen flow rate/pressure correlation (pump curve physics)
            flow_ratio = flow_rate / self.pump_specifications['ballast_pump_capacity_m3h']
            # Realistic pump curve: pressure increases with resistance at higher flows
            pump_curve_factor = 1.0 + (flow_ratio * 0.3)  # 30% pressure increase at max flow
            base_pressure *= pressure_factor * pump_curve_factor
        elif pump_running:
            # Clean data also follows pump physics
            flow_ratio = flow_rate / self.pump_specifications['ballast_pump_capacity_m3h']
            pump_curve_factor = 1.0 + (flow_ratio * 0.3)
            base_pressure *= pump_curve_factor
        
        pressure = base_pressure + random.uniform(-0.8, 1.2) if pump_running else 0
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(day, phase) if self.uncertainty_enabled else 0.0
        
        # Determine active failures
        active_failures = list(self.equipment_state.active_failures.keys()) if self.uncertainty_enabled else []
        
        # Build base record
        record = {
            'timestamp': actual_time,
            'operational_phase': phase['phase'],
            'vessel_region': 'auxiliary_safety',
            'log_type': 'ballast_operation',
            'pump_status': 'running' if pump_running else 'standby',
            'ballast_flow_rate_m3h': round(flow_rate, 1),
            'ballast_pressure_bar': round(pressure, 1),
            'total_ballast_water_mt': round(self.stability_state.ballast_water_mt, 0),
            'cargo_loaded_mt': round(self.stability_state.cargo_loaded_mt, 0),
            'vessel_list_angle_deg': round(0.5 + random.uniform(-0.3, 0.8), 1),
            'stability_margin': round(self.stability_state.stability_margin, 2),
            'pump_power_consumption_kw': round(power_consumption, 1),
            'pump_running_hours': round(random.uniform(2, 8), 1),
            'bilge_level_mm': random.randint(50, 200),
            'bilge_pump_status': 'auto' if random.random() > 0.1 else 'manual',
            'tank_ullage_monitoring': 'operational',
            'vessel_profile': self.vessel_profile.profile_name
        }
        
        # Add uncertainty-specific fields if enabled
        if self.uncertainty_enabled:
            record.update({
                'risk_score': round(risk_score, 3),
                'active_failures': ','.join(active_failures) if active_failures else 'none',
                'equipment_efficiency': round(self.equipment_state.ballast_pump_efficiency, 2),
                'failure_stage': self._get_dominant_failure_stage(),
                'maintenance_urgency': self._calculate_maintenance_urgency()
            })
        
        return record
    
    def _generate_ballast_monitoring_data(self, date: datetime, hour: int, phase: Dict, day: int) -> Dict:
        """Generate routine ballast monitoring during sea transit"""
        actual_time = self.timing_simulator.generate_crew_timestamp(
            day + 1, hour, (20, 40), False
        )
        
        # Muster and deployment times with realistic variability
        base_muster_time = 4.0
        base_deployment_time = 8.0
        
        if self.uncertainty_enabled:
            # Times affected by crew condition and equipment status
            fatigue_delay = self.human_factors['crew_fatigue_level'] * 2.0  # Up to 2 min delay
            equipment_delay = (1 - self.equipment_state.emergency_equipment_condition) * 3.0  # Up to 3 min delay
            
            # Operational phase affects response time
            phase_delay = 1.0 if phase['phase'] in ['loading', 'unloading'] else 0.0
            
            muster_time = base_muster_time + fatigue_delay + phase_delay + random.uniform(-0.5, 1.0)
            deployment_time = base_deployment_time + equipment_delay + phase_delay + random.uniform(-1.0, 2.0)
        else:
            # Clean data - good performance with normal variation
            muster_time = base_muster_time + random.uniform(-1, 1.5)
            deployment_time = base_deployment_time + random.uniform(-2, 3)
        
        # Ensure realistic minimums
        muster_time = max(2.0, muster_time)
        deployment_time = max(5.0, deployment_time)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(day, phase) if self.uncertainty_enabled else 0.0
        
        # Determine active failures
        active_failures = list(self.equipment_state.active_failures.keys()) if self.uncertainty_enabled else []
        
        record = {
            'timestamp': actual_time,
            'operational_phase': phase['phase'],
            'vessel_region': 'auxiliary_safety',
            'log_type': 'ballast_monitoring',
            'pump_status': 'standby',
            'ballast_flow_rate_m3h': 0,
            'ballast_pressure_bar': 0,
            'total_ballast_water_mt': round(self.stability_state.ballast_water_mt, 0),
            'cargo_loaded_mt': round(self.stability_state.cargo_loaded_mt, 0),
            'vessel_list_angle_deg': round(random.uniform(-0.5, 0.5), 1),
            'stability_margin': round(self.stability_state.stability_margin, 2),
            'pump_power_consumption_kw': 5,  # Standby power
            'pump_running_hours': 0,
            'bilge_level_mm': random.randint(30, 120),
            'bilge_pump_status': 'auto',
            'tank_ullage_monitoring': 'operational',
            'fuel_transfer_status': 'standby',
            'sea_water_system_pressure_bar': 4.5 + random.uniform(-0.5, 0.8),
            'vessel_profile': self.vessel_profile.profile_name
        }
        
        # Add uncertainty-specific fields if enabled
        if self.uncertainty_enabled:
            record.update({
                'risk_score': round(risk_score, 3),
                'active_failures': ','.join(active_failures) if active_failures else 'none',
                'equipment_efficiency': round(self.equipment_state.ballast_pump_efficiency, 2),
                'failure_stage': self._get_dominant_failure_stage(),
                'maintenance_urgency': self._calculate_maintenance_urgency()
            })
        
        return record
    
    def _generate_safety_monitoring_data(self, date: datetime, hour: int, phase: Dict, day: int) -> Dict:
        """Generate safety system monitoring data"""
        actual_time = self.timing_simulator.generate_crew_timestamp(
            day + 1, hour, (15, 30), False
        )
        
        # Safety monitoring intensity increases during cargo operations
        monitoring_intensity = 'high' if phase['phase'] in ['loading', 'unloading'] else 'routine'
        
        # Fire detection system status with uncertainty variations
        fire_system_status = self._get_safety_equipment_status('fire_detection')
        
        # Bilge and flooding monitoring with uncertainty variations
        base_bilge_level = random.randint(30, 150)
        if self.uncertainty_enabled and 'bilge_pump_failure' in self.equipment_state.active_failures:
            severity = self.equipment_state.active_failures['bilge_pump_failure']['severity']
            base_bilge_level = int(base_bilge_level * (1 + severity * 2))  # Higher levels during failure
        
        bilge_level = min(500, base_bilge_level)  # Cap at emergency level
        bilge_alarm_status = 'normal' if bilge_level < 150 else ('attention_required' if bilge_level < 300 else 'high_level_alarm')
        
        # Fire detection zones with uncertainty variations
        base_zones = random.randint(14, 16)
        if self.uncertainty_enabled:
            zones_active = int(base_zones * self.equipment_state.fire_detection_reliability)
        else:
            zones_active = base_zones
        
        # Sprinkler system pressure with uncertainty variations
        base_sprinkler_pressure = 12.5 + random.uniform(-1.0, 1.5)
        if self.uncertainty_enabled and 'fire_detection_failure' in self.equipment_state.active_failures:
            severity = self.equipment_state.active_failures['fire_detection_failure']['severity']
            base_sprinkler_pressure *= (1 - severity * 0.3)  # Pressure reduction during failure
        
        # CO2 system pressure with uncertainty variations (FIXED: Realistic maritime range)
        base_co2_pressure = 57.5 + random.uniform(-2.5, 2.5)  # Realistic 55-60 bar range
        if self.uncertainty_enabled:
            co2_pressure = base_co2_pressure * self.equipment_state.co2_system_pressure_ratio
        else:
            co2_pressure = base_co2_pressure
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(day, phase) if self.uncertainty_enabled else 0.0
        
        # Determine active failures
        active_failures = list(self.equipment_state.active_failures.keys()) if self.uncertainty_enabled else []
        
        record = {
            'timestamp': actual_time,
            'operational_phase': phase['phase'],
            'vessel_region': 'auxiliary_safety',
            'log_type': 'safety_monitoring',
            'monitoring_intensity': monitoring_intensity,
            'fire_detection_status': fire_system_status,
            'smoke_detector_zones_active': zones_active,
            'fire_alarm_panel_status': 'normal' if zones_active >= 14 else 'degraded',
            'sprinkler_system_pressure_bar': round(base_sprinkler_pressure, 1),
            'bilge_level_engine_room_mm': bilge_level,
            'bilge_level_cargo_holds_mm': random.randint(20, 80),
            'bilge_alarm_status': bilge_alarm_status,
            'emergency_lighting_status': 'operational',
            'pa_system_status': 'operational',
            'emergency_shutdown_systems': 'armed',
            'co2_system_pressure_bar': round(co2_pressure, 1),
            'foam_system_readiness': 'standby',
            'safety_compliance_rating': round(self.safety_status.safety_compliance_rating, 1),
            'vessel_profile': self.vessel_profile.profile_name
        }
        
        # Add uncertainty-specific fields if enabled
        if self.uncertainty_enabled:
            record.update({
                'risk_score': round(risk_score, 3),
                'active_failures': ','.join(active_failures) if active_failures else 'none',
                'fire_detection_reliability': round(self.equipment_state.fire_detection_reliability, 2),
                'emergency_equipment_condition': round(self.equipment_state.emergency_equipment_condition, 2),
                'failure_stage': self._get_dominant_failure_stage(),
                'maintenance_urgency': self._calculate_maintenance_urgency()
            })
        
        return record
    
    def _generate_safety_inspection_data(self, date: datetime, hour: int, phase: Dict, day: int) -> Dict:
        """Generate weekly SOLAS-compliant safety inspection data"""
        actual_time = self.timing_simulator.generate_crew_timestamp(
            day + 1, hour, (45, 90), False  # Inspections take longer
        )
        
        # Reset inspection counter
        self.equipment_state.days_since_last_inspection = 0
        
        # Weekly inspection improves safety compliance rating (IMPROVED: More realistic patterns)
        base_compliance_improvement = random.uniform(0.1, 0.3)
        if self.uncertainty_enabled:
            # Reduce improvement if human factors are poor
            thoroughness_factor = self.human_factors['inspection_thoroughness']
            # Add fatigue impact on inspection quality
            fatigue_factor = 1 - (self.human_factors['crew_fatigue_level'] * 0.5)
            # Operational phase affects inspection quality
            phase_factor = 0.8 if phase['phase'] in ['loading', 'unloading'] else 1.0  # Busier = less thorough
            
            compliance_improvement = base_compliance_improvement * thoroughness_factor * fatigue_factor * phase_factor
        else:
            compliance_improvement = base_compliance_improvement
        
        # Prevent unrealistic compliance ratings
        self.safety_status.safety_compliance_rating = min(5.0, max(3.0,
            self.safety_status.safety_compliance_rating + compliance_improvement))
        
        # Inspection findings with uncertainty variations (IMPROVED: More realistic patterns)
        findings = []
        base_finding_probability = 0.15  # 15% chance of minor findings
        
        if self.uncertainty_enabled:
            # Increase finding probability based on multiple factors
            adjusted_probability = base_finding_probability
            
            # Active failures increase finding probability
            adjusted_probability += len(self.equipment_state.active_failures) * 0.12
            
            # Poor inspection thoroughness increases missed issues (paradoxically more findings later)
            thoroughness_factor = 2.2 - self.human_factors['inspection_thoroughness']
            adjusted_probability *= thoroughness_factor
            
            # Operational phase affects finding rate
            if phase['phase'] in ['loading', 'unloading']:
                adjusted_probability *= 1.3  # Higher finding rate during busy operations
            
            # Equipment condition affects findings
            avg_equipment_condition = (
                self.equipment_state.ballast_pump_efficiency +
                self.equipment_state.fire_detection_reliability +
                self.equipment_state.emergency_equipment_condition +
                self.equipment_state.bilge_pump_reliability
            ) / 4
            adjusted_probability *= (2 - avg_equipment_condition)  # Poor equipment = more findings
            
            # Cap probability at reasonable maximum
            adjusted_probability = min(0.6, adjusted_probability)
            
            if random.random() < adjusted_probability:
                # Multiple findings possible for severe conditions
                num_findings = 1
                if adjusted_probability > 0.3:
                    num_findings += random.choice([0, 0, 1])  # 33% chance of additional finding
                
                for _ in range(num_findings):
                    findings.append(random.choice([
                        'minor_maintenance_required', 'calibration_due', 'documentation_update',
                        'equipment_wear_noted', 'seal_replacement_needed', 'filter_cleaning_required',
                        'valve_adjustment_needed', 'alarm_testing_overdue'
                    ]))
        else:
            # Clean data - fewer findings, more predictable
            if random.random() < base_finding_probability:
                findings.append(random.choice([
                    'minor_maintenance_required', 'calibration_due', 'documentation_update'
                ]))
        
        # Remove duplicate findings
        findings = list(set(findings))
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(day, phase) if self.uncertainty_enabled else 0.0
        
        # Determine active failures
        active_failures = list(self.equipment_state.active_failures.keys()) if self.uncertainty_enabled else []
        
        record = {
            'timestamp': actual_time,
            'operational_phase': phase['phase'],
            'vessel_region': 'auxiliary_safety',
            'log_type': 'safety_inspection',
            'inspection_type': 'weekly_solas_compliance',
            'inspector': 'chief_engineer',
            'fire_extinguishers_checked': random.randint(25, 35),
            'fire_extinguisher_status': 'compliant' if not findings else 'minor_issues',
            'lifeboat_inspection_status': 'satisfactory',
            'emergency_equipment_count': random.randint(95, 105),  # Expected ~100 items
            'immersion_suits_checked': random.randint(22, 28),  # Crew + spares
            'safety_signs_condition': 'good',
            'emergency_escape_routes': 'clear',
            'safety_equipment_deficiencies': len(findings),
            'inspection_findings': ', '.join(findings) if findings else 'no_deficiencies',
            'compliance_rating': round(self.safety_status.safety_compliance_rating, 1),
            'next_inspection_due': (actual_time + timedelta(days=7)).strftime('%Y-%m-%d'),
            'inspection_duration_minutes': 90 + random.randint(-15, 30),
            'vessel_profile': self.vessel_profile.profile_name
        }
        
        # Add uncertainty-specific fields if enabled
        if self.uncertainty_enabled:
            record.update({
                'risk_score': round(risk_score, 3),
                'active_failures': ','.join(active_failures) if active_failures else 'none',
                'inspection_thoroughness': round(self.human_factors['inspection_thoroughness'], 2),
                'crew_fatigue_level': round(self.human_factors['crew_fatigue_level'], 2),
                'failure_stage': self._get_dominant_failure_stage(),
                'maintenance_urgency': self._calculate_maintenance_urgency()
            })
        
        return record
    
    def _generate_emergency_drill_data(self, date: datetime, hour: int, phase: Dict, day: int) -> Dict:
        """Generate monthly emergency drill data"""
        actual_time = self.timing_simulator.generate_crew_timestamp(
            day + 1, hour, (30, 60), False  # Drills take longer
        )
        
        # Update last drill date
        self.safety_status.last_safety_drill = actual_time
        
        # Drill type varies
        drill_types = ['fire_drill', 'abandon_ship_drill', 'man_overboard_drill', 'collision_drill']
        drill_type = random.choice(drill_types)
        
        # Crew participation
        total_crew = 20 + random.randint(-2, 3)
        crew_participation = random.randint(18, total_crew)
        
        # Drill performance rating with uncertainty variations (IMPROVED: More realistic factors)
        base_performance = self.safety_equipment_quality
        if self.uncertainty_enabled:
            # Performance affected by multiple realistic factors
            equipment_factor = self.equipment_state.emergency_equipment_condition * 0.25
            compliance_factor = self.human_factors['procedure_compliance'] * 0.25
            fatigue_factor = (1 - self.human_factors['crew_fatigue_level']) * 0.20
            experience_factor = 0.30  # Base crew experience - could be vessel profile dependent
            
            # Operational phase affects performance
            phase_stress = 1.0
            if current_phase['phase'] in ['loading', 'unloading']:
                phase_stress = 0.95  # Slightly reduced performance when crew is busy
            
            # Active failures impact drill performance
            failure_impact = max(0.85, 1 - (len(self.equipment_state.active_failures) * 0.05))
            
            performance_modifier = (equipment_factor + compliance_factor + fatigue_factor + experience_factor)
            drill_performance = base_performance * performance_modifier * phase_stress * failure_impact
            
            # Add some random variation but keep within realistic bounds
            drill_performance *= random.uniform(0.90, 1.10)
            drill_performance = max(2.0, min(5.0, drill_performance))  # Clamp to realistic range
        else:
            # Clean data - good but not perfect performance
            drill_performance = base_performance * random.uniform(0.85, 1.15)
            drill_performance = max(3.0, min(5.0, drill_performance))  # Higher baseline for clean data
        
        # Update drill performance trend
        if self.uncertainty_enabled:
            self.human_factors['drill_performance_trend'] = drill_performance - 4.0  # Track relative to baseline
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(day, phase) if self.uncertainty_enabled else 0.0
        
        # Determine active failures
        active_failures = list(self.equipment_state.active_failures.keys()) if self.uncertainty_enabled else []
        
        record = {
            'timestamp': actual_time,
            'operational_phase': phase['phase'],
            'vessel_region': 'auxiliary_safety',
            'log_type': 'emergency_drill',
            'drill_type': drill_type,
            'drill_duration_minutes': 25 + random.randint(-8, 15),
            'crew_participation_count': crew_participation,
            'total_crew_count': total_crew,
            'muster_time_minutes': round(muster_time, 1),
            'equipment_deployment_time_minutes': round(deployment_time, 1),
            'communication_effectiveness': 'satisfactory' if drill_performance > 3.5 else 'needs_improvement',
            'drill_performance_rating': round(drill_performance, 1),
            'lessons_learned': 'routine_drill_completed',
            'improvement_actions': 'none_required' if drill_performance > 4.0 else 'additional_training',
            'regulatory_compliance': 'solas_compliant',
            'next_drill_scheduled': (actual_time + timedelta(days=30 + random.randint(-3, 3))).strftime('%Y-%m-%d'),  # Variable scheduling
            'officer_in_charge': 'chief_officer',
            'vessel_profile': self.vessel_profile.profile_name
        }
        
        # Add uncertainty-specific fields if enabled
        if self.uncertainty_enabled:
            record.update({
                'risk_score': round(risk_score, 3),
                'active_failures': ','.join(active_failures) if active_failures else 'none',
                'crew_fatigue_impact': round(self.human_factors['crew_fatigue_level'], 2),
                'equipment_readiness': round(self.equipment_state.emergency_equipment_condition, 2),
                'failure_stage': self._get_dominant_failure_stage(),
                'maintenance_urgency': self._calculate_maintenance_urgency()
            })
        
        return record
    
    def _get_safety_equipment_status(self, equipment_type: str) -> str:
        """Determine safety equipment status based on vessel profile and uncertainty"""
        
        if not self.uncertainty_enabled:
            # Clean data - mostly operational
            return 'operational' if random.random() > 0.05 else 'minor_maintenance_due'
        
        # Equipment reliability varies by vessel profile and active failures
        base_reliability = self.safety_equipment_quality / 5.0  # Normalize to 0-1
        
        # Adjust for active failures
        if equipment_type == 'fire_detection' and 'fire_detection_failure' in self.equipment_state.active_failures:
            severity = self.equipment_state.active_failures['fire_detection_failure']['severity']
            failure_probability = 0.05 + (severity * 0.3)  # Up to 35% failure probability
        else:
            failure_probability = 0.05 * (1 / base_reliability)  # Lower failure rate for higher quality equipment
        
        if random.random() < failure_probability:
            return random.choice(['maintenance_required', 'reduced_capability', 'offline'])
        else:
            return 'operational'
    
    def _get_dominant_failure_stage(self) -> str:
        """Get the most severe failure stage currently active"""
        
        if not self.uncertainty_enabled or not self.equipment_state.active_failures:
            return FailureStage.NORMAL.value
        
        # Find the most severe failure stage
        severity_order = [
            FailureStage.NORMAL.value,
            FailureStage.EARLY_WARNING.value,
            FailureStage.DEGRADED.value,
            FailureStage.CRITICAL.value,
            FailureStage.EMERGENCY.value
        ]
        
        most_severe_stage = FailureStage.NORMAL.value
        for failure_data in self.equipment_state.active_failures.values():
            current_stage = failure_data['stage']
            if severity_order.index(current_stage) > severity_order.index(most_severe_stage):
                most_severe_stage = current_stage
        
        return most_severe_stage
    
    def _calculate_maintenance_urgency(self) -> str:
        """Calculate maintenance urgency based on current conditions"""
        
        if not self.uncertainty_enabled:
            return 'routine'
        
        # Calculate urgency based on active failures and overdue maintenance
        urgency_score = 0.0
        
        # Active failure contribution
        for failure_data in self.equipment_state.active_failures.values():
            urgency_score += failure_data['severity'] * 0.3
        
        # Overdue maintenance contribution
        days_overdue = max(0, self.equipment_state.days_since_last_inspection - 7)
        urgency_score += min(0.4, days_overdue / 14)
        
        # Equipment condition contribution
        avg_equipment_condition = (
            self.equipment_state.ballast_pump_efficiency +
            self.equipment_state.fire_detection_reliability +
            self.equipment_state.emergency_equipment_condition +
            self.equipment_state.bilge_pump_reliability
        ) / 4
        urgency_score += (1 - avg_equipment_condition) * 0.3
        
        # Determine urgency level
        if urgency_score > 0.7:
            return 'immediate'
        elif urgency_score > 0.4:
            return 'high'
        elif urgency_score > 0.2:
            return 'moderate'
        else:
            return 'routine'
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary of uncertainty injection for analysis"""
        
        if not self.uncertainty_enabled:
            return {'uncertainty_enabled': False}
        
        return {
            'uncertainty_enabled': True,
            'total_failures_injected': self.uncertainty_metadata['total_failures_injected'],
            'failure_types_encountered': list(self.uncertainty_metadata['failure_types_encountered']),
            'cascade_events': len(self.uncertainty_metadata['cascade_events']),
            'peak_risk_score': self.uncertainty_metadata['peak_risk_score'],
            'final_equipment_state': {
                'ballast_pump_efficiency': self.equipment_state.ballast_pump_efficiency,
                'fire_detection_reliability': self.equipment_state.fire_detection_reliability,
                'emergency_equipment_condition': self.equipment_state.emergency_equipment_condition,
                'bilge_pump_reliability': self.equipment_state.bilge_pump_reliability
            },
            'final_human_factors': self.human_factors.copy(),
            'active_failures_at_end': list(self.equipment_state.active_failures.keys())
        }
    
    def get_failure_history(self) -> List[Dict]:
        """Get detailed failure history for analysis"""
        
        if not self.uncertainty_enabled:
            return []
        
        return self.failure_history.copy()


# Example usage and testing
if __name__ == "__main__":
    from datetime import datetime
    
    # Mock objects for testing (replace with actual imports)
    class MockVoyagePlan:
        def __init__(self):
            self.start_date = datetime(2024, 1, 1)
            self.end_date = datetime(2024, 1, 31)
            self.vessel_profile = MockVesselProfile()
        
        def get_current_phase(self, date):
            day = (date - self.start_date).days
            if day < 3:
                return type('Phase', (), {'value': 'loading'})()
            elif day < 25:
                return type('Phase', (), {'value': 'sea_transit'})()
            else:
                return type('Phase', (), {'value': 'unloading'})()
    
    class MockVesselProfile:
        def __init__(self):
            self.profile_name = "Modern_Major_Bulk_Carrier"
            self.company_type = CompanyType.MAJOR
            self.automation_level = AutomationLevel.HIGH
    
    # Test the generator - Clean Data
    print("ðŸ§ª Testing Enhanced Auxiliary Support Generator - Clean Data")
    voyage_plan = MockVoyagePlan()
    generator = EnhancedAuxiliarySafetySystemsGenerator(voyage_plan, uncertainty_enabled=False)
    clean_dataset = generator.generate_dataset()
    
    print(f"\nðŸ“Š Clean Dataset Generated:")
    print(f"Total records: {len(clean_dataset)}")
    print(f"Date range: {clean_dataset['timestamp'].min()} to {clean_dataset['timestamp'].max()}")
    print(f"Log types: {clean_dataset['log_type'].value_counts().to_dict()}")
    print(f"Operational phases: {clean_dataset['operational_phase'].value_counts().to_dict()}")
    
    # Test the generator - With Uncertainty
    print("\nðŸ§ª Testing Enhanced Auxiliary Support Generator - With Uncertainty")
    uncertainty_generator = EnhancedAuxiliarySafetySystemsGenerator(voyage_plan, uncertainty_enabled=True)
    uncertain_dataset = uncertainty_generator.generate_dataset()
    
    print(f"\nðŸ“Š Uncertain Dataset Generated:")
    print(f"Total records: {len(uncertain_dataset)}")
    print(f"Log types: {uncertain_dataset['log_type'].value_counts().to_dict()}")
    
    # Get uncertainty summary
    summary = uncertainty_generator.get_uncertainty_summary()
    print(f"\nðŸŽ¯ Uncertainty Summary:")
    print(f"Failures injected: {summary['total_failures_injected']}")
    print(f"Failure types: {summary['failure_types_encountered']}")
    print(f"Peak risk score: {summary['peak_risk_score']:.3f}")
    print(f"Cascade events: {summary['cascade_events']}")
    
    # Sample records
    print(f"\nðŸ“ Sample Ballast Operation Record (Clean):")
    ballast_sample = clean_dataset[clean_dataset['log_type'] == 'ballast_operation'].iloc[0] if len(clean_dataset[clean_dataset['log_type'] == 'ballast_operation']) > 0 else None
    if ballast_sample is not None:
        print(ballast_sample.to_dict())
    
    print(f"\nðŸ“ Sample Safety Monitoring Record (Uncertain):")
    safety_sample = uncertain_dataset[uncertain_dataset['log_type'] == 'safety_monitoring'].iloc[0] if len(uncertain_dataset[uncertain_dataset['log_type'] == 'safety_monitoring']) > 0 else None
    if safety_sample is not None:
        print(safety_sample.to_dict())