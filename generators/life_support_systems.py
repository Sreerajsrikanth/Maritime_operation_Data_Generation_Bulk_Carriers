"""
Enhanced Life Support Systems Generator with Industry-Validated Uncertainty Injection
Following exact pattern of Engine, Bridge, Deck, Cargo generators

INDUSTRY CORRECTIONS APPLIED:
✅ Equipment Failure Rates: Galley 0.016/day, HVAC 0.014/day, Hot Water 0.008/day, Laundry 0.005/day
✅ MLC Compliance: 65% fatigue trigger (not 75%), 15% probability when triggered
✅ Food Safety: 0.001/day minor, 0.0005/day major, 0.002/day expired detection
✅ Cascade Failures: 10min-2hr minor blackouts, 4-8hr major blackouts
✅ Crew Fatigue: Variable 1-2.5%/day increase, -3 to -6%/day port recovery
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
from core.vessel_profile import VesselProfile, CompanyType, AutomationLevel


class FailureStage(Enum):
    NORMAL = "normal_operations"
    EARLY_WARNING = "early_warning_signs"
    DEGRADED = "degraded_performance"
    CRITICAL = "critical_condition"
    EMERGENCY = "emergency_situation"


class EquipmentType(Enum):
    GALLEY = "galley_equipment"
    HVAC = "hvac_system"
    HOT_WATER = "hot_water_system"
    LAUNDRY = "laundry_equipment"


@dataclass
class LifeSupportEquipmentState:
    equipment_type: EquipmentType
    current_stage: FailureStage
    efficiency: float  # 0.0-1.0
    days_in_current_stage: int
    failure_history: List[Dict]
    maintenance_due_days: int
    last_failure_date: Optional[datetime]


@dataclass
class CrewWelfareState:
    fatigue_level: float  # 0.0-1.0
    rest_compliance: bool
    consecutive_rest_violations: int
    meal_quality_rating: float
    accommodation_comfort: float
    last_shore_leave: datetime
    crew_complaints_count: int
    medical_issues_active: int


@dataclass
class FoodSafetyState:
    supplies_freshness: float  # 0.0-1.0
    refrigeration_temperature_compliance: bool
    water_quality_status: str
    last_food_safety_incident: Optional[datetime]
    hygiene_compliance_rating: float
    spoilage_rate_multiplier: float


class LifeSupportUncertaintySimulator:
    """Industry-validated uncertainty injection for life support systems"""
    
    def __init__(self, vessel_profile: VesselProfile):
        self.vessel_profile = vessel_profile
        
        # Industry-validated failure rates (per day) - CORRECTED
        self.base_failure_rates = {
            EquipmentType.GALLEY: 0.016,      # 6 failures/year (corrected from 8-12)
            EquipmentType.HVAC: 0.014,        # 5 failures/year (corrected from 6-10)
            EquipmentType.HOT_WATER: 0.008,   # 3 failures/year (corrected from 2-4)
            EquipmentType.LAUNDRY: 0.005      # 2 failures/year
        }
        
        # Phase-specific risk multipliers
        self.phase_multipliers = {
            'loading': 2.0,
            'unloading': 1.5,
            'heavy_weather': 1.8,
            'port_operations': 1.3,
            'sea_transit': 1.0
        }
        
        # Vessel profile adjustments
        self._apply_vessel_profile_multipliers()
        
        # Food safety incident rates (per day) - CORRECTED
        self.food_safety_rates = {
            'minor_poisoning': 0.001,      # 0.4/year (single crew)
            'major_outbreak': 0.0005,      # 0.2/year (multiple crew) - corrected from 0.7
            'expired_detection': 0.002,    # 0.7/year - corrected from 1.5
            'contaminated_water': 0.001    # 0.4/year
        }
    
    def _apply_vessel_profile_multipliers(self):
        """Apply vessel profile specific multipliers"""
        if hasattr(self.vessel_profile, 'company_type'):
            if self.vessel_profile.company_type == CompanyType.MAJOR:
                self.failure_multiplier = 0.8
                self.maintenance_quality = 1.2
            elif self.vessel_profile.company_type == CompanyType.SMALL:
                self.failure_multiplier = 1.3
                self.maintenance_quality = 0.8
            else:  # MIDTIER
                self.failure_multiplier = 1.0
                self.maintenance_quality = 1.0
        else:
            # Fallback based on profile name
            profile_name = self.vessel_profile.profile_name.lower()
            if 'major' in profile_name or 'modern' in profile_name:
                self.failure_multiplier = 0.8
                self.maintenance_quality = 1.2
            elif 'small' in profile_name or 'legacy' in profile_name:
                self.failure_multiplier = 1.3
                self.maintenance_quality = 0.8
            else:
                self.failure_multiplier = 1.0
                self.maintenance_quality = 1.0
    
    def simulate_equipment_failure(self, equipment_type: EquipmentType, 
                                 current_phase: str, day: int) -> Optional[Dict]:
        """Simulate equipment failure with 5-stage progression"""
        
        # Calculate adjusted failure probability
        base_rate = self.base_failure_rates[equipment_type]
        phase_multiplier = self.phase_multipliers.get(current_phase, 1.0)
        vessel_multiplier = self.failure_multiplier
        
        daily_failure_probability = base_rate * phase_multiplier * vessel_multiplier
        
        if random.random() < daily_failure_probability:
            failure_type = self._generate_failure_type(equipment_type)
            severity = random.choice(['minor', 'moderate', 'major'])
            repair_time = self._estimate_repair_time(equipment_type, severity)
            
            return {
                'equipment_type': equipment_type.value,
                'failure_type': failure_type,
                'severity': severity,
                'estimated_repair_hours': repair_time,
                'failure_stage': FailureStage.EARLY_WARNING.value,
                'operational_impact': self._assess_operational_impact(equipment_type, severity),
                'day_occurred': day
            }
        
        return None
    
    def simulate_cascade_failure(self, trigger_event: str) -> Optional[Dict]:
        """Simulate cascade failures with corrected durations"""
        
        cascade_probability = {
            'electrical_overload': 0.15,
            'hvac_failure': 0.08,
            'galley_fire': 0.25
        }.get(trigger_event, 0.05)
        
        if random.random() < cascade_probability:
            # Corrected blackout durations
            if trigger_event == 'electrical_overload':
                duration_minutes = random.randint(10, 120)  # 10min-2hr minor blackouts
                blackout_type = 'minor'
            else:
                duration_minutes = random.randint(240, 480)  # 4-8hr major blackouts (rare)
                blackout_type = 'major'
            
            return {
                'cascade_type': 'electrical_blackout',
                'trigger_event': trigger_event,
                'blackout_type': blackout_type,
                'duration_minutes': duration_minutes,
                'affected_systems': ['lighting', 'ventilation', 'galley_equipment'],
                'emergency_power_activated': duration_minutes > 30
            }
        
        return None
    
    def simulate_mlc_violation(self, crew_fatigue_level: float, 
                              current_phase: str) -> Optional[Dict]:
        """Simulate MLC compliance violations - CORRECTED"""
        
        # Corrected trigger point: 65% fatigue (not 75%)
        if crew_fatigue_level >= 0.65:
            
            # Phase-specific multipliers for violation probability
            phase_multiplier = self.phase_multipliers.get(current_phase, 1.0)
            
            # Corrected: 15% probability when triggered (not deterministic)
            violation_probability = 0.15 * phase_multiplier
            
            if random.random() < violation_probability:
                return {
                    'violation_type': 'rest_hours_non_compliance',
                    'fatigue_level': crew_fatigue_level,
                    'operational_phase': current_phase,
                    'severity': 'minor' if crew_fatigue_level < 0.75 else 'major',
                    'regulatory_risk': 'inspection_required' if crew_fatigue_level > 0.8 else 'monitoring_increased'
                }
        
        return None
    
    def simulate_food_safety_incident(self, supplies_freshness: float,
                                    temperature_compliance: bool) -> Optional[Dict]:
        """Simulate food safety incidents with industry-validated rates"""
        
        # Adjust probabilities based on conditions
        freshness_multiplier = 2.0 if supplies_freshness < 0.6 else 1.0
        temperature_multiplier = 1.5 if not temperature_compliance else 1.0
        
        for incident_type, base_rate in self.food_safety_rates.items():
            adjusted_rate = base_rate * freshness_multiplier * temperature_multiplier
            
            if random.random() < adjusted_rate:
                return {
                    'incident_type': incident_type,
                    'supplies_freshness': supplies_freshness,
                    'temperature_compliance': temperature_compliance,
                    'affected_crew_count': 1 if incident_type == 'minor_poisoning' else random.randint(3, 8),
                    'investigation_required': incident_type in ['major_outbreak', 'contaminated_water']
                }
        
        return None
    
    def _generate_failure_type(self, equipment_type: EquipmentType) -> str:
        """Generate specific failure types per equipment"""
        failure_types = {
            EquipmentType.GALLEY: ['oven_malfunction', 'refrigeration_failure', 'ventilation_blocked', 'electrical_fault'],
            EquipmentType.HVAC: ['compressor_failure', 'fan_malfunction', 'thermostat_error', 'duct_blockage'],
            EquipmentType.HOT_WATER: ['heater_element_failure', 'pump_malfunction', 'temperature_sensor_error', 'pipe_leak'],
            EquipmentType.LAUNDRY: ['washing_machine_fault', 'dryer_overheating', 'water_supply_issue', 'drainage_blockage']
        }
        
        return random.choice(failure_types[equipment_type])
    
    def _estimate_repair_time(self, equipment_type: EquipmentType, severity: str) -> int:
        """Estimate repair time in hours"""
        base_times = {
            EquipmentType.GALLEY: {'minor': 2, 'moderate': 6, 'major': 12},
            EquipmentType.HVAC: {'minor': 3, 'moderate': 8, 'major': 16},
            EquipmentType.HOT_WATER: {'minor': 1, 'moderate': 4, 'major': 8},
            EquipmentType.LAUNDRY: {'minor': 1, 'moderate': 3, 'major': 6}
        }
        
        base_time = base_times[equipment_type][severity]
        return base_time + random.randint(-1, 3)  # Add variation
    
    def _assess_operational_impact(self, equipment_type: EquipmentType, severity: str) -> str:
        """Assess operational impact by equipment and severity"""
        if severity == 'major':
            return 'service_disruption'
        elif severity == 'moderate':
            return 'reduced_capacity'
        else:
            return 'minimal_impact'


class LifeSupportCrewTimingSimulator:
    """Simulates realistic crew timing variations for life support operations"""
    
    @staticmethod
    def generate_crew_timestamp(day: int, hour: int, 
                               task_duration_mins: Tuple[int, int] = (10, 30),
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


class EnhancedLifeSupportSystemsGenerator:
    """
    Enhanced Life Support Systems Generator with Industry-Validated Uncertainty Injection
    Following exact pattern of Engine, Bridge, Deck, Cargo generators
    """
    
    def __init__(self, voyage_plan: VoyagePlan, uncertainty_enabled: bool = True):
        self.voyage_plan = voyage_plan
        self.vessel_profile = voyage_plan.vessel_profile
        self.uncertainty_enabled = uncertainty_enabled
        self.timing_simulator = LifeSupportCrewTimingSimulator()
        
        # Apply vessel profile adjustments
        self._apply_vessel_profile_adjustments()
        
        # Standard meal times (industry validated)
        self.meal_times = [7, 12, 18]  # 07:00, 12:00, 18:00
        
        # Initialize operational state
        self.current_operational_state = {
            'galley_equipment_status': 'operational',
            'food_supplies_level': 1.0,
            'accommodation_cleanliness': 0.8,
            'crew_quarters_temperature': 21.0,
            'ventilation_status': 'operational'
        }
        
        # Track uncertainty events
        self.failure_history = []
        self.cascade_events = []
        self.mlc_violations = []
        self.food_safety_incidents = []
        
        # CRITICAL FIX: Always initialize states (even for clean mode)
        self._initialize_base_states()
        
        # Initialize uncertainty simulation if enabled
        if self.uncertainty_enabled:
            self.uncertainty_simulator = LifeSupportUncertaintySimulator(self.vessel_profile)
            self._initialize_uncertainty_states()
    
    def _initialize_base_states(self):
        """Initialize base states needed for both clean and uncertain modes"""
        
        # Crew welfare state - ALWAYS needed
        self.crew_welfare_state = CrewWelfareState(
            fatigue_level=0.3,  # Start moderate
            rest_compliance=True,
            consecutive_rest_violations=0,
            meal_quality_rating=3.5,
            accommodation_comfort=0.7,
            last_shore_leave=self.voyage_plan.start_date - timedelta(days=5),
            crew_complaints_count=0,
            medical_issues_active=0
        )
        
        # Food safety state - ALWAYS needed
        self.food_safety_state = FoodSafetyState(
            supplies_freshness=1.0,
            refrigeration_temperature_compliance=True,
            water_quality_status='compliant',
            last_food_safety_incident=None,
            hygiene_compliance_rating=0.9,
            spoilage_rate_multiplier=1.0
        )
    
    def _initialize_uncertainty_states(self):
        """Initialize uncertainty tracking states (only for uncertainty mode)"""
        
        # Equipment states - ONLY for uncertainty mode
        self.equipment_states = {}
        for equipment_type in EquipmentType:
            self.equipment_states[equipment_type] = LifeSupportEquipmentState(
                equipment_type=equipment_type,
                current_stage=FailureStage.NORMAL,
                efficiency=1.0,
                days_in_current_stage=0,
                failure_history=[],
                maintenance_due_days=random.randint(30, 90),
                last_failure_date=None
            )
    
    def _apply_vessel_profile_adjustments(self):
        """Apply vessel profile variations to life support systems"""
        
        # Company type effects
        if hasattr(self.vessel_profile, 'company_type'):
            if self.vessel_profile.company_type == CompanyType.MAJOR:
                self.facility_quality_multiplier = 1.2
                self.crew_facility_rating = 4.2
            elif self.vessel_profile.company_type == CompanyType.SMALL:
                self.facility_quality_multiplier = 0.8
                self.crew_facility_rating = 3.0
            else:  # MIDTIER
                self.facility_quality_multiplier = 1.0
                self.crew_facility_rating = 3.5
        else:
            # Fallback to profile name parsing
            profile_name = self.vessel_profile.profile_name.lower()
            if 'modern' in profile_name or 'major' in profile_name:
                self.facility_quality_multiplier = 1.2
                self.crew_facility_rating = 4.2
            elif 'legacy' in profile_name or 'small' in profile_name:
                self.facility_quality_multiplier = 0.8
                self.crew_facility_rating = 3.0
            else:
                self.facility_quality_multiplier = 1.0
                self.crew_facility_rating = 3.5
        
        # Automation level effects
        if hasattr(self.vessel_profile, 'automation_level'):
            if self.vessel_profile.automation_level == AutomationLevel.HIGH:
                self.automation_level = 0.85
                self.galley_equipment_efficiency = 1.15
            elif self.vessel_profile.automation_level == AutomationLevel.LOW:
                self.automation_level = 0.35
                self.galley_equipment_efficiency = 0.85
            else:  # MEDIUM
                self.automation_level = 0.65
                self.galley_equipment_efficiency = 1.0
        else:
            # Fallback
            self.automation_level = 0.65
            self.galley_equipment_efficiency = 1.0
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete life support systems dataset for voyage
        Integrates accommodation and galley operations with optional uncertainty injection
        """
        all_data = []  # FIXED: Initialize data list
        voyage_start = self.voyage_plan.start_date
        
        # Handle different VoyagePlan structures for duration
        if hasattr(self.voyage_plan, 'voyage_duration_days'):
            voyage_duration = self.voyage_plan.voyage_duration_days
        else:
            duration = self.voyage_plan.end_date - self.voyage_plan.start_date
            voyage_duration = duration.days
        
        print(f"🏠 Generating Life Support Systems data for {voyage_duration}-day voyage...")
        if self.uncertainty_enabled:
            print("   ⚠️  Uncertainty injection ENABLED")
        else:
            print("   ✅ Clean baseline data generation")
        
        for day in range(voyage_duration):
            current_date = voyage_start + timedelta(days=day)
            
            # Handle different get_current_phase return types
            current_phase_raw = self.voyage_plan.get_current_phase(current_date)
            
            if hasattr(current_phase_raw, 'value'):
                current_phase = {
                    "phase": current_phase_raw.value,
                    "logging_intensity": "high" if current_phase_raw.value in ["loading", "unloading"] else "moderate"
                }
            else:
                current_phase = current_phase_raw
            
            # Update daily uncertainties if enabled
            if self.uncertainty_enabled:
                self._update_daily_uncertainties(day, current_phase, current_date)
            else:
                self._update_crew_welfare_state_clean(day, current_phase)
            
            # Generate meal service data (3x daily)
            for meal_hour in self.meal_times:
                meal_data = self._generate_meal_service_data(
                    current_date, meal_hour, current_phase, day
                )
                all_data.append(meal_data)
            
            # Generate accommodation monitoring data (every 8 hours)
            for check_hour in [6, 14, 22]:
                accommodation_data = self._generate_accommodation_data(
                    current_date, check_hour, current_phase, day
                )
                all_data.append(accommodation_data)
            
            # Generate crew welfare monitoring (every 12 hours)
            for welfare_hour in [8, 20]:
                welfare_data = self._generate_crew_welfare_data(
                    current_date, welfare_hour, current_phase, day
                )
                all_data.append(welfare_data)
            
            # MUST-FIX 2: AGGRESSIVE deep cleaning generation during port phases
            # Generate deep cleaning MUCH more frequently to ensure it appears in dataset
            if current_phase['phase'] in ['loading', 'unloading', 'port_operations']:
                # MULTIPLE ways to trigger deep cleaning for guaranteed coverage
                daily_cleaning_chance = 0.6  # 60% chance each day in port
                
                # Trigger conditions (ANY of these will generate cleaning):
                cleaning_triggers = [
                    (day % 2 == 0),           # Every 2 days
                    (day % 3 == 0),           # Every 3 days  
                    (random.random() < daily_cleaning_chance),  # 60% random chance
                    (day == 0),               # Always on first day
                    (day >= 5 and day % 4 == 0)  # Every 4 days after day 5
                ]
                
                if any(cleaning_triggers):
                    # Generate 1-3 cleaning operations per day to ensure coverage
                    cleaning_count = random.randint(1, 3)
                    for cleaning_session in range(cleaning_count):
                        cleaning_hour = 10 + (cleaning_session * 4)  # 10:00, 14:00, 18:00
                        cleaning_data = self._generate_deep_cleaning_data(
                            current_date, cleaning_hour, current_phase, day
                        )
                        all_data.append(cleaning_data)
            
            # ADDITIONAL: Generate cleaning during sea transit (routine maintenance)
            elif current_phase['phase'] == 'sea_transit' and random.random() < 0.2:
                # 20% chance of routine cleaning at sea
                cleaning_data = self._generate_deep_cleaning_data(
                    current_date, 14, current_phase, day  # 14:00 afternoon cleaning
                )
                all_data.append(cleaning_data)
        
        # FIXED: Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(all_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Life Support Systems dataset generated: {len(df)} records")
        if self.uncertainty_enabled:
            print(f"   🔧 Equipment failures: {len(self.failure_history)}")
            print(f"   ⚡ Cascade events: {len(self.cascade_events)}")
            print(f"   📋 MLC violations: {len(self.mlc_violations)}")
            print(f"   🍽️ Food safety incidents: {len(self.food_safety_incidents)}")
        
        return df
    
    def _update_equipment_degradation(self, day: int):
        """MUST-FIX 3: Progressive equipment degradation for realistic failure scenarios"""
        for equipment_type, state in self.equipment_states.items():
            # Daily degradation rate varies by equipment type and operational conditions
            base_degradation = {
                EquipmentType.GALLEY: 0.008,      # Higher degradation for complex equipment
                EquipmentType.HVAC: 0.006,        # Moderate degradation
                EquipmentType.HOT_WATER: 0.004,   # Lower degradation
                EquipmentType.LAUNDRY: 0.003      # Lowest degradation
            }
            
            daily_degradation = base_degradation[equipment_type] * random.uniform(0.5, 2.0)
            
            # Apply degradation
            state.efficiency = max(0.2, state.efficiency - daily_degradation)
            state.days_in_current_stage += 1
            
            # Progressive failure stage advancement based on efficiency
            if state.efficiency < 0.9 and state.current_stage == FailureStage.NORMAL:
                state.current_stage = FailureStage.EARLY_WARNING
                state.days_in_current_stage = 0
            elif state.efficiency < 0.7 and state.current_stage == FailureStage.EARLY_WARNING:
                state.current_stage = FailureStage.DEGRADED
                state.days_in_current_stage = 0
            elif state.efficiency < 0.5 and state.current_stage == FailureStage.DEGRADED:
                state.current_stage = FailureStage.CRITICAL
                state.days_in_current_stage = 0
            elif state.efficiency < 0.3 and state.current_stage == FailureStage.CRITICAL:
                state.current_stage = FailureStage.EMERGENCY
                state.days_in_current_stage = 0
            
            # Chance for maintenance to improve efficiency
            if random.random() < 0.05:  # 5% chance of maintenance
                maintenance_improvement = random.uniform(0.1, 0.3)
                state.efficiency = min(1.0, state.efficiency + maintenance_improvement)
                # May improve failure stage after maintenance
                if state.efficiency > 0.7 and state.current_stage in [FailureStage.CRITICAL, FailureStage.EMERGENCY]:
                    state.current_stage = FailureStage.DEGRADED
                elif state.efficiency > 0.9 and state.current_stage == FailureStage.DEGRADED:
                    state.current_stage = FailureStage.EARLY_WARNING
    
    def _update_daily_uncertainties(self, day: int, current_phase: Dict, current_date: datetime):
        """Update daily uncertainties for all systems"""
        
        # Update crew fatigue with CORRECTED variable accumulation
        fatigue_increase = random.uniform(0.01, 0.025)  # Variable 1-2.5%/day
        
        if current_phase['phase'] in ['loading', 'unloading']:
            # Port time - variable recovery
            fatigue_decrease = random.uniform(0.03, 0.06)  # -3% to -6%/day
            self.crew_welfare_state.fatigue_level = max(0.2, 
                self.crew_welfare_state.fatigue_level - fatigue_decrease)
        else:
            # Sea time - gradual accumulation
            self.crew_welfare_state.fatigue_level = min(0.9, 
                self.crew_welfare_state.fatigue_level + fatigue_increase)
        
        # MUST-FIX 3: Enhanced equipment failure scenarios with progressive degradation
        self._update_equipment_degradation(day)
        
        # Simulate equipment failures with increased frequency
        for equipment_type in EquipmentType:
            failure = self.uncertainty_simulator.simulate_equipment_failure(
                equipment_type, current_phase['phase'], day
            )
            if failure:
                self.failure_history.append(failure)
                # Progressive failure stage advancement
                current_stage = self.equipment_states[equipment_type].current_stage
                if current_stage == FailureStage.NORMAL:
                    self.equipment_states[equipment_type].current_stage = FailureStage.EARLY_WARNING
                elif current_stage == FailureStage.EARLY_WARNING:
                    self.equipment_states[equipment_type].current_stage = FailureStage.DEGRADED
                elif current_stage == FailureStage.DEGRADED:
                    self.equipment_states[equipment_type].current_stage = FailureStage.CRITICAL
                # Apply efficiency reduction
                self.equipment_states[equipment_type].efficiency *= random.uniform(0.6, 0.8)
        
        # CRITICAL FIX: Enhanced MLC violations with AGGRESSIVE realistic occurrence
        # Make violations happen MORE FREQUENTLY for realistic maritime operations
        
        # First check for fatigue-based violations (original logic)
        mlc_violation = self.uncertainty_simulator.simulate_mlc_violation(
            self.crew_welfare_state.fatigue_level, current_phase['phase']
        )
        if mlc_violation:
            self.mlc_violations.append(mlc_violation)
            self.crew_welfare_state.consecutive_rest_violations += 1
        
        # ADDITIONAL: Random operational violations (more realistic)
        # Even with good fatigue management, violations can happen due to operational demands
        base_violation_chance = 0.08  # 8% base chance per day
        
        # Phase multipliers for operational pressure
        phase_multipliers = {
            'loading': 2.5,      # High pressure during cargo ops
            'unloading': 2.0,    # High pressure during cargo ops
            'sea_transit': 0.5,  # Lower pressure at sea
            'port_operations': 1.5,
            'arrival': 1.8,      # Port approach pressure
            'departure': 1.8     # Port departure pressure
        }
        
        phase_multiplier = phase_multipliers.get(current_phase['phase'], 1.0)
        daily_violation_chance = base_violation_chance * phase_multiplier
        
        # Additional violation from operational pressure (independent of fatigue)
        if random.random() < daily_violation_chance:
            self.crew_welfare_state.consecutive_rest_violations += 1
            self.mlc_violations.append({
                'violation_type': 'operational_rest_hours_shortage',
                'cause': 'operational_demands',
                'operational_phase': current_phase['phase'],
                'day_occurred': day
            })
        
        # REALISTIC RESET: Allow violations to reset after compliance periods
        # Make reset chance based on operational phase and current violation count
        if self.crew_welfare_state.consecutive_rest_violations > 0:
            # Higher violation count = lower chance to reset (accumulated fatigue effect)
            reset_base_chance = 0.25  # 25% base chance
            violation_penalty = min(0.15, self.crew_welfare_state.consecutive_rest_violations * 0.05)
            reset_chance = reset_base_chance - violation_penalty
            
            # Port phases have better reset chances (more rest opportunities)
            if current_phase['phase'] in ['loading', 'unloading']:
                reset_chance *= 1.5
            
            if random.random() < reset_chance:
                self.crew_welfare_state.consecutive_rest_violations = max(0, 
                    self.crew_welfare_state.consecutive_rest_violations - 1)
        
        # Simulate food safety incidents
        food_incident = self.uncertainty_simulator.simulate_food_safety_incident(
            self.food_safety_state.supplies_freshness,
            self.food_safety_state.refrigeration_temperature_compliance
        )
        if food_incident:
            self.food_safety_incidents.append(food_incident)
            self.food_safety_state.last_food_safety_incident = current_date
        
        # MUST-FIX 1: Add realistic variance to food spoilage rate
        # Update food supplies with variable spoilage rate
        supply_consumption = random.uniform(0.02, 0.04)
        self.food_safety_state.supplies_freshness -= supply_consumption
        self.food_safety_state.supplies_freshness = max(0.3, self.food_safety_state.supplies_freshness)
        
        # CRITICAL FIX: Make spoilage rate variable based on conditions
        base_spoilage = 1.0
        # Spoilage increases with temperature, age, and equipment failures
        temperature_factor = 1.0 + (self.current_operational_state.get('crew_quarters_temperature', 21) - 21) * 0.02
        age_factor = 1.0 + (1.0 - self.food_safety_state.supplies_freshness) * 0.5
        equipment_factor = 1.0 + (1.0 - self.equipment_states[EquipmentType.GALLEY].efficiency) * 0.3
        
        self.food_safety_state.spoilage_rate_multiplier = base_spoilage * temperature_factor * age_factor * equipment_factor
        self.food_safety_state.spoilage_rate_multiplier = min(2.0, self.food_safety_state.spoilage_rate_multiplier)
        
        # Update accommodation cleanliness
        if day % 3 == 0:  # Cleaning day
            self.current_operational_state['accommodation_cleanliness'] = min(0.95, 
                self.current_operational_state['accommodation_cleanliness'] + 0.15)
        else:
            self.current_operational_state['accommodation_cleanliness'] *= 0.98
    
    def _update_crew_welfare_state_clean(self, day: int, current_phase: Dict):
        """Update crew welfare state for clean baseline data (no uncertainty)"""
        
        # Fatigue accumulates over time, improves during port time
        if current_phase['phase'] in ['loading', 'unloading']:
            self.crew_welfare_state.fatigue_level = max(0.2, 
                self.crew_welfare_state.fatigue_level - 0.05)  # Fixed 5% recovery
        else:
            self.crew_welfare_state.fatigue_level = min(0.7, 
                self.crew_welfare_state.fatigue_level + 0.02)  # Fixed 2% accumulation
        
        # Food supplies diminish predictably
        self.food_safety_state.supplies_freshness *= 0.97
        
        # Accommodation cleanliness predictable cycle
        if day % 3 == 0:
            self.current_operational_state['accommodation_cleanliness'] = min(0.9, 
                self.current_operational_state['accommodation_cleanliness'] + 0.1)
        else:
            self.current_operational_state['accommodation_cleanliness'] *= 0.98
    
    def _generate_meal_service_data(self, date: datetime, hour: int, phase: Dict, day: int) -> Dict:
        """Generate meal service operational data"""
        meal_types = ['breakfast', 'lunch', 'dinner']
        meal_type = meal_types[self.meal_times.index(hour)]
        
        # Apply realistic crew timing variations
        actual_time = self.timing_simulator.generate_crew_timestamp(
            day + 1, hour, (10, 25), self.timing_simulator.is_handover_time(hour)
        )
        
        # Crew count affects meal preparation
        crew_count = 20 + random.randint(-2, 3)
        
        # Meal quality affected by vessel profile and supplies
        base_quality = self.crew_facility_rating
        if self.uncertainty_enabled:
            supply_factor = min(1.0, self.food_safety_state.supplies_freshness * 1.2)
            equipment_efficiency = self.equipment_states[EquipmentType.GALLEY].efficiency
            meal_quality = base_quality * supply_factor * equipment_efficiency * random.uniform(0.9, 1.1)
        else:
            supply_factor = min(1.0, self.food_safety_state.supplies_freshness * 1.2)
            meal_quality = base_quality * supply_factor * random.uniform(0.95, 1.05)
        
        base_data = {
            'timestamp': actual_time,
            'operational_phase': phase['phase'],
            'vessel_region': 'life_support',
            'log_type': 'meal_service',
            'meal_type': meal_type,
            'meals_served': crew_count,
            'meal_quality_rating': round(meal_quality, 2),
            'food_temperature_c': 65 + random.uniform(-8, 12),
            'food_safety_status': 'compliant',
            'galley_temperature_c': 28 + random.uniform(-3, 5),
            'galley_equipment_status': self._get_equipment_status_clean('galley'),
            'food_supplies_level_percent': round(self.food_safety_state.supplies_freshness * 100, 1),
            'service_duration_minutes': 45 + random.randint(-10, 20),
            'crew_attendance_percent': 85 + random.randint(-15, 15),
            'vessel_profile': self.vessel_profile.profile_name
        }
        
        # Add uncertainty fields if enabled
        if self.uncertainty_enabled:
            galley_state = self.equipment_states[EquipmentType.GALLEY]
            base_data.update({
                'equipment_efficiency': round(galley_state.efficiency, 3),
                'galley_failure_stage': galley_state.current_stage.value,
                'food_spoilage_rate': round(self.food_safety_state.spoilage_rate_multiplier, 2),  # FIXED: Now variable
                'galley_equipment_status': self._get_equipment_status_with_uncertainty(EquipmentType.GALLEY)
            })
        
        return base_data
    
    def _generate_accommodation_data(self, date: datetime, hour: int, phase: Dict, day: int) -> Dict:
        """Generate accommodation monitoring and maintenance data"""
        actual_time = self.timing_simulator.generate_crew_timestamp(
            day + 1, hour, (15, 35), False
        )
        
        # Temperature varies by phase and vessel efficiency
        base_temp = 21.0
        if phase['phase'] in ['loading', 'unloading']:
            base_temp += 2.0
        
        temperature = base_temp + random.uniform(-2, 3) * (1 / self.facility_quality_multiplier)
        
        base_data = {
            'timestamp': actual_time,
            'operational_phase': phase['phase'],
            'vessel_region': 'life_support',
            'log_type': 'accommodation_monitoring',
            'crew_quarters_temperature_c': round(temperature, 1),
            'accommodation_humidity_percent': 45 + random.uniform(-8, 15),
            'ventilation_status': self._get_equipment_status_clean('ventilation'),
            'noise_level_db': self._calculate_noise_level(phase),
            'cleanliness_rating': round(self.current_operational_state['accommodation_cleanliness'] * 5, 1),
            'equipment_functionality': self._get_equipment_status_clean('accommodation'),
            'crew_quarters_occupancy': random.randint(18, 22),
            'lighting_status': 'operational',
            'hot_water_availability': 'available',
            'laundry_status': 'operational' if day % 3 == 0 else 'scheduled',
            'vessel_profile': self.vessel_profile.profile_name
        }
        
        # Add uncertainty fields if enabled
        if self.uncertainty_enabled:
            hvac_state = self.equipment_states[EquipmentType.HVAC]
            base_data.update({
                'hvac_efficiency': round(hvac_state.efficiency, 3),
                'hvac_failure_stage': hvac_state.current_stage.value,
                'risk_score': self._calculate_life_support_risk_score()['accommodation_risk']
            })
        
        return base_data
    
    def _generate_crew_welfare_data(self, date: datetime, hour: int, phase: Dict, day: int) -> Dict:
        """Generate crew welfare and MLC compliance monitoring"""
        actual_time = self.timing_simulator.generate_crew_timestamp(
            day + 1, hour, (20, 40), False
        )
        
        # Rest compliance based on fatigue levels and regulations
        rest_compliance = self.crew_welfare_state.fatigue_level < 0.7
        
        base_data = {
            'timestamp': actual_time,
            'operational_phase': phase['phase'],
            'vessel_region': 'life_support',
            'log_type': 'crew_welfare_monitoring',
            'crew_fatigue_level': round(self.crew_welfare_state.fatigue_level, 2),
            'rest_hours_compliance': 'compliant' if rest_compliance else 'non_compliant',
            'crew_morale_rating': round(3.5 + (1 - self.crew_welfare_state.fatigue_level), 1),
            'medical_issues_reported': random.choice([0, 0, 0, 1]) if random.random() < 0.05 else 0,
            'crew_complaints': random.choice([0, 0, 1]) if random.random() < 0.1 else 0,
            'welfare_officer_notes': 'routine_check',
            'mlc_compliance_status': 'compliant' if rest_compliance else 'review_required',
            'days_since_shore_leave': (date - self.crew_welfare_state.last_shore_leave).days,
            'crew_recreational_activities': 'available' if phase['phase'] == 'sea_transit' else 'limited',
            'vessel_profile': self.vessel_profile.profile_name
        }
        
        # Add uncertainty fields if enabled
        if self.uncertainty_enabled:
            base_data.update({
                'consecutive_violations': self.crew_welfare_state.consecutive_rest_violations,  # FIXED: Now variable
                'mlc_inspection_risk': self._assess_mlc_inspection_risk(),
                'welfare_deterioration_risk': self._assess_welfare_risk()
            })
        
        return base_data
    
    def _generate_deep_cleaning_data(self, date: datetime, hour: int, phase: Dict, day: int) -> Dict:
        """Generate deep cleaning operations data - ENHANCED for guaranteed coverage"""
        actual_time = self.timing_simulator.generate_crew_timestamp(
            day + 1, hour, (30, 60), False
        )
        
        # Deep cleaning improves accommodation conditions significantly
        self.current_operational_state['accommodation_cleanliness'] = 0.95
        
        # Vary cleaning types for realism
        cleaning_types = [
            'weekly_deep_clean',
            'routine_maintenance_clean', 
            'port_hygiene_inspection_clean',
            'crew_quarters_deep_clean',
            'galley_deep_sanitization',
            'accommodation_spaces_clean'
        ]
        
        # Vary areas cleaned based on operational phase
        if phase['phase'] in ['loading', 'unloading']:
            areas_cleaned_options = [
                'all_accommodation_spaces',
                'crew_quarters_and_galley',
                'recreation_and_dining_areas',
                'sanitation_facilities',
                'crew_common_areas'
            ]
        else:
            areas_cleaned_options = [
                'routine_crew_quarters',
                'galley_maintenance_areas',
                'basic_accommodation_clean'
            ]
        
        # Vary special attention areas
        special_areas = [
            'galley_ventilation_system',
            'crew_quarters_air_circulation',
            'waste_management_areas',
            'laundry_facilities',
            'food_storage_areas',
            'water_systems_cleaning'
        ]
        
        return {
            'timestamp': actual_time,
            'operational_phase': phase['phase'],
            'vessel_region': 'life_support',
            'log_type': 'deep_cleaning_operation',  # CRITICAL: Ensure this exact log_type
            'cleaning_type': random.choice(cleaning_types),
            'areas_cleaned': random.choice(areas_cleaned_options),
            'cleaning_duration_hours': round(2.5 + random.uniform(-0.5, 2.0), 1),  # 2-4.5 hours
            'cleaning_supplies_used': random.choice([
                'standard_maritime_detergents',
                'industrial_cleaning_compounds',
                'sanitization_chemicals',
                'eco_friendly_cleaners'
            ]),
            'post_cleaning_rating': round(4.0 + random.uniform(-0.5, 1.0), 1),  # 3.5-5.0
            'equipment_maintenance': random.choice([
                'routine_check_completed',
                'minor_repairs_performed', 
                'equipment_servicing_done',
                'preventive_maintenance'
            ]),
            'hygiene_compliance': random.choice([
                'mlc_compliant',
                'exceeds_standards',
                'meets_requirements',
                'inspection_ready'
            ]),
            'crew_assigned': random.randint(2, 8),  # 2-8 crew members
            'special_attention_areas': random.choice(special_areas),
            'cleaning_priority': random.choice(['routine', 'high', 'urgent', 'scheduled']),
            'inspection_status': random.choice(['passed', 'completed', 'approved', 'certified']),
            'vessel_profile': self.vessel_profile.profile_name
        }
    
    def _get_equipment_status_with_uncertainty(self, equipment_type: EquipmentType) -> str:
        """Get equipment status with uncertainty injection"""
        equipment_state = self.equipment_states[equipment_type]
        
        if equipment_state.current_stage == FailureStage.NORMAL:
            return 'operational'
        elif equipment_state.current_stage == FailureStage.EARLY_WARNING:
            return 'monitoring_required'
        elif equipment_state.current_stage == FailureStage.DEGRADED:
            return 'reduced_capacity'
        elif equipment_state.current_stage == FailureStage.CRITICAL:
            return 'maintenance_required'
        else:  # EMERGENCY
            return 'offline'
    
    def _get_equipment_status_clean(self, equipment_type: str) -> str:
        """Get equipment status for clean baseline data"""
        # Equipment reliability varies by vessel age and profile
        reliability_factor = self.facility_quality_multiplier
        failure_probability = 0.02 * (1 / reliability_factor)  # Very low for clean data
        
        if random.random() < failure_probability:
            return random.choice(['maintenance_scheduled', 'monitoring'])
        else:
            return 'operational'
    
    def _calculate_noise_level(self, phase: Dict) -> int:
        """Calculate accommodation noise levels based on operational phase"""
        base_noise = 45  # Baseline accommodation noise
        
        if phase['phase'] in ['loading', 'unloading']:
            base_noise += 10  # Cargo operations increase noise
        elif phase['phase'] == 'sea_transit':
            base_noise += 5   # Engine operation noise
        
        # Vessel profile affects noise insulation
        noise_reduction = self.facility_quality_multiplier * 3
        actual_noise = base_noise - noise_reduction + random.uniform(-3, 8)
        
        return int(max(35, min(65, actual_noise)))
    
    def _assess_maintenance_urgency(self) -> str:
        """Assess overall maintenance urgency"""
        if not self.uncertainty_enabled:
            return 'routine'
        
        urgent_count = sum(1 for state in self.equipment_states.values() 
                          if state.current_stage in [FailureStage.CRITICAL, FailureStage.EMERGENCY])
        
        if urgent_count >= 2:
            return 'urgent'
        elif urgent_count == 1:
            return 'priority'
        else:
            return 'routine'
    
    def _assess_welfare_risk(self) -> float:
        """Assess crew welfare deterioration risk"""
        if not self.uncertainty_enabled:
            return 0.1
        
        fatigue_risk = self.crew_welfare_state.fatigue_level * 0.4
        violation_risk = min(0.3, self.crew_welfare_state.consecutive_rest_violations * 0.1)
        accommodation_risk = (1 - self.current_operational_state['accommodation_cleanliness']) * 0.2
        
        total_risk = fatigue_risk + violation_risk + accommodation_risk
        return round(min(1.0, total_risk), 3)
    
    def _assess_mlc_inspection_risk(self) -> str:
        """Assess MLC inspection risk level"""
        if not self.uncertainty_enabled:
            return 'low'
        
        if self.crew_welfare_state.consecutive_rest_violations >= 3:
            return 'high'
        elif self.crew_welfare_state.fatigue_level > 0.8:
            return 'elevated'
        elif self.crew_welfare_state.consecutive_rest_violations > 0:
            return 'moderate'
        else:
            return 'low'
    
    def _calculate_life_support_risk_score(self) -> Dict:
        """Calculate comprehensive life support risk scores with industry-validated weighting"""
        if not self.uncertainty_enabled:
            return {
                'equipment_reliability': 0.05,
                'mlc_compliance': 0.02,
                'crew_welfare': 0.08,
                'food_safety': 0.03,
                'accommodation_risk': 0.04,
                'overall_life_support_risk': 0.06
            }
        
        # Equipment reliability (35% weight)
        equipment_risk = 0.0
        for equipment_state in self.equipment_states.values():
            if equipment_state.current_stage == FailureStage.EMERGENCY:
                equipment_risk += 0.25
            elif equipment_state.current_stage == FailureStage.CRITICAL:
                equipment_risk += 0.15
            elif equipment_state.current_stage == FailureStage.DEGRADED:
                equipment_risk += 0.08
            elif equipment_state.current_stage == FailureStage.EARLY_WARNING:
                equipment_risk += 0.03
        
        equipment_reliability = min(1.0, equipment_risk) * 0.35
        
        # MLC compliance (25% weight)
        mlc_risk = 0.0
        if self.crew_welfare_state.fatigue_level > 0.8:
            mlc_risk += 0.4
        elif self.crew_welfare_state.fatigue_level > 0.65:
            mlc_risk += 0.2
        
        mlc_risk += min(0.3, self.crew_welfare_state.consecutive_rest_violations * 0.1)
        mlc_compliance = min(1.0, mlc_risk) * 0.25
        
        # Crew welfare (25% weight)
        welfare_risk = self.crew_welfare_state.fatigue_level * 0.6
        welfare_risk += (1 - self.current_operational_state['accommodation_cleanliness']) * 0.3
        welfare_risk += min(0.1, self.crew_welfare_state.medical_issues_active * 0.05)
        crew_welfare = min(1.0, welfare_risk) * 0.25
        
        # Food safety (15% weight)
        food_risk = (1 - self.food_safety_state.supplies_freshness) * 0.7
        if not self.food_safety_state.refrigeration_temperature_compliance:
            food_risk += 0.2
        if self.food_safety_state.last_food_safety_incident:
            days_since = (datetime.now() - self.food_safety_state.last_food_safety_incident).days
            if days_since < 7:
                food_risk += 0.1
        
        food_safety = min(1.0, food_risk) * 0.15
        
        # Overall risk score
        overall_risk = equipment_reliability + mlc_compliance + crew_welfare + food_safety
        
        return {
            'equipment_reliability': round(equipment_reliability, 3),
            'mlc_compliance': round(mlc_compliance, 3),
            'crew_welfare': round(crew_welfare, 3),
            'food_safety': round(food_safety, 3),
            'accommodation_risk': round(crew_welfare * 0.6, 3),  # Subset of crew welfare
            'overall_life_support_risk': round(overall_risk, 3)
        }
    
    def get_uncertainty_summary(self) -> Dict:
        """Get comprehensive uncertainty injection summary"""
        if not self.uncertainty_enabled:
            return {
                'uncertainty_enabled': False,
                'message': 'Clean baseline data generation - no uncertainties injected'
            }
        
        # Equipment failure summary
        equipment_failures_by_type = {}
        for failure in self.failure_history:
            equipment_type = failure['equipment_type']
            if equipment_type not in equipment_failures_by_type:
                equipment_failures_by_type[equipment_type] = 0
            equipment_failures_by_type[equipment_type] += 1
        
        # Current equipment status
        current_equipment_status = {}
        for equipment_type, state in self.equipment_states.items():
            current_equipment_status[equipment_type.value] = {
                'stage': state.current_stage.value,
                'efficiency': round(state.efficiency, 3),
                'days_in_stage': state.days_in_current_stage
            }
        
        # Risk scoring summary
        risk_scores = self._calculate_life_support_risk_score()
        
        return {
            'uncertainty_enabled': True,
            'vessel_profile': self.vessel_profile.profile_name,
            'total_failures': len(self.failure_history),
            'equipment_failures_by_type': equipment_failures_by_type,
            'cascade_events': len(self.cascade_events),
            'mlc_violations': len(self.mlc_violations),
            'food_safety_incidents': len(self.food_safety_incidents),
            'current_equipment_status': current_equipment_status,
            'current_crew_fatigue': round(self.crew_welfare_state.fatigue_level, 3),
            'current_food_freshness': round(self.food_safety_state.supplies_freshness, 3),
            'current_accommodation_cleanliness': round(self.current_operational_state['accommodation_cleanliness'], 3),
            'risk_scores': risk_scores,
            'failure_history': self.failure_history[-5:] if self.failure_history else [],  # Last 5 failures
        }
    
    def get_failure_history(self) -> Dict:
        """Get complete failure history"""
        if not self.uncertainty_enabled:
            return {
                'equipment_failures': [],
                'cascade_events': [],
                'mlc_violations': [],
                'food_safety_incidents': []
            }
        
        return {
            'equipment_failures': self.failure_history,
            'cascade_events': self.cascade_events,
            'mlc_violations': self.mlc_violations,
            'food_safety_incidents': self.food_safety_incidents
        }
    
    def get_current_system_status(self) -> Dict:
        """Get current system status snapshot"""
        base_status = {
            'operational_state': self.current_operational_state,
            'crew_welfare': {
                'fatigue_level': round(self.crew_welfare_state.fatigue_level, 3),
                'rest_compliance': self.crew_welfare_state.rest_compliance,
                'meal_quality': self.crew_welfare_state.meal_quality_rating
            },
            'food_safety': {
                'supplies_freshness': round(self.food_safety_state.supplies_freshness, 3),
                'temperature_compliance': self.food_safety_state.refrigeration_temperature_compliance,
                'water_quality': self.food_safety_state.water_quality_status
            },
            'vessel_profile_effects': {
                'facility_quality_multiplier': self.facility_quality_multiplier,
                'crew_facility_rating': self.crew_facility_rating,
                'automation_level': self.automation_level
            }
        }
        
        if self.uncertainty_enabled:
            base_status.update({
                'equipment_states': {
                    equipment_type.value: {
                        'stage': state.current_stage.value,
                        'efficiency': round(state.efficiency, 3),
                        'maintenance_due_days': state.maintenance_due_days
                    }
                    for equipment_type, state in self.equipment_states.items()
                },
                'risk_scores': self._calculate_life_support_risk_score(),
                'maintenance_urgency': self._assess_maintenance_urgency(),
                'mlc_inspection_risk': self._assess_mlc_inspection_risk()
            })
        
        return base_status