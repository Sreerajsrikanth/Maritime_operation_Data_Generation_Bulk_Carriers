"""
Voyage Plan with Vessel Profile Integration

This module integrates vessel profiles with voyage planning to create
systematic variations in vessel operations while maintaining authentic
maritime operational patterns.

The voyage plan remains the foundation for all data generation, but now
vessel profiles modify operational parameters to create realistic
fleet-wide variations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import random

# Import vessel profile system
from .vessel_profile import VesselProfile, VesselProfileManager


class VoyagePhase(Enum):
    """Voyage operational phases"""
    LOADING = "loading"
    DEPARTURE = "departure"
    SEA_TRANSIT = "sea_transit"
    ARRIVAL = "arrival"
    UNLOADING = "unloading"
    PORT_OPERATIONS = "port_operations"


class WeatherCondition(Enum):
    """Weather conditions affecting vessel operations"""
    CALM = "calm"           # Sea State 0-2
    MODERATE = "moderate"   # Sea State 3-4
    ROUGH = "rough"         # Sea State 5-6
    SEVERE = "severe"       # Sea State 7+


@dataclass
class RouteSegment:
    """Individual segment of a voyage route"""
    start_port: str
    end_port: str
    start_time: datetime
    end_time: datetime
    distance_nm: float
    planned_speed_kts: float
    weather_condition: WeatherCondition
    phase: VoyagePhase
    
    def duration_hours(self) -> float:
        """Calculate segment duration in hours"""
        return (self.end_time - self.start_time).total_seconds() / 3600


@dataclass
class CargoOperation:
    """Cargo loading/unloading operations"""
    operation_type: str  # "loading" or "unloading"
    cargo_type: str
    quantity_tonnes: float
    start_time: datetime
    end_time: datetime
    port: str
    
    def duration_hours(self) -> float:
        """Calculate operation duration in hours"""
        return (self.end_time - self.start_time).total_seconds() / 3600


@dataclass
class VoyagePlan:
    """
    Complete voyage plan with vessel profile integration
    
    This maintains the existing voyage-plan-driven approach while
    incorporating vessel profile variations for realistic fleet operations.
    """
    
    # Core voyage information
    voyage_id: str
    vessel_profile: VesselProfile
    start_date: datetime
    end_date: datetime
    
    # Route information
    origin_port: str
    destination_port: str
    route_segments: List[RouteSegment] = field(default_factory=list)
    
    # Cargo operations
    cargo_operations: List[CargoOperation] = field(default_factory=list)
    
    # Operational parameters (modified by vessel profile)
    crew_watch_schedule: Dict[str, List[datetime]] = field(default_factory=dict)
    maintenance_schedule: Dict[str, List[datetime]] = field(default_factory=dict)
    inspection_schedule: Dict[str, List[datetime]] = field(default_factory=dict)
    
    # Profile-adjusted operational parameters
    adjusted_logging_frequencies: Dict[str, float] = field(default_factory=dict)
    adjusted_failure_rates: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize voyage plan with vessel profile adjustments"""
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")
        
        # Calculate profile-adjusted operational parameters
        self._calculate_adjusted_parameters()
        
        # Generate schedules based on vessel profile
        self._generate_crew_schedules()
        self._generate_maintenance_schedules()
        self._generate_inspection_schedules()
    
    def _calculate_adjusted_parameters(self):
        """Calculate operational parameters adjusted for vessel profile"""
        
        # Base logging frequencies (entries per day)
        base_logging_frequencies = {
            "engine_room": 8.0,     # ~8 entries/day (watch + rounds)
            "bridge": 12.0,         # ~12 entries/day (watch + navigation)
            "cargo": 3.0,           # ~3 entries/day (active during port ops)
            "safety": 2.0,          # ~2 entries/day (safety rounds)
            "deck": 4.0,            # ~4 entries/day (deck operations)
            "accommodation": 3.0,   # ~3 entries/day (crew activities)
            "pump_room": 2.0        # ~2 entries/day (pump operations)
        }
        
        # Apply vessel profile multipliers
        for region, base_freq in base_logging_frequencies.items():
            self.adjusted_logging_frequencies[region] = \
                self.vessel_profile.get_logging_frequency(region, base_freq)
        
        # Base failure rates (per day, from MARCAT data)
        base_failure_rates = {
            "propulsion": 0.019,     # 6.96/year → 0.019/day
            "electrical": 0.016,     # 5.78/year → 0.016/day
            "navigation": 0.010,     # 3.59/year → 0.010/day
            "fire_explosion": 0.003, # 0.95/year → 0.003/day
            "structural": 0.005,     # Estimated structural issues
            "machinery": 0.012,      # General machinery failures
            "human_error": 0.008     # Human factor incidents
        }
        
        # Apply vessel profile multipliers
        for failure_type, base_rate in base_failure_rates.items():
            self.adjusted_failure_rates[failure_type] = \
                self.vessel_profile.get_failure_rate(failure_type, base_rate)
    
    def _generate_crew_schedules(self):
        """Generate crew watch schedules based on vessel profile"""
        
        # Standard 4-hour watch schedule
        watch_times = [0, 4, 8, 12, 16, 20]  # 24-hour format
        
        current_date = self.start_date
        while current_date <= self.end_date:
            daily_watches = []
            
            for watch_hour in watch_times:
                watch_time = current_date.replace(hour=watch_hour, minute=0, second=0, microsecond=0)
                
                # Add human factors variation based on vessel profile
                crew_familiarity = self.vessel_profile.get_human_factor("crew_familiarity")
                delay_variation = random.uniform(0, 30 * (1 - crew_familiarity))  # 0-30 min delay
                
                actual_watch_time = watch_time + timedelta(minutes=delay_variation)
                daily_watches.append(actual_watch_time)
            
            self.crew_watch_schedule[current_date.strftime("%Y-%m-%d")] = daily_watches
            current_date += timedelta(days=1)
    
    def _generate_maintenance_schedules(self):
        """Generate maintenance schedules based on vessel profile"""
        
        # Get maintenance intervals from vessel profile
        engine_interval = self.vessel_profile.get_maintenance_interval("engine_major")
        electrical_interval = self.vessel_profile.get_maintenance_interval("electrical_check")
        safety_interval = self.vessel_profile.get_maintenance_interval("safety_inspection")
        
        # Schedule maintenance events
        self.maintenance_schedule["engine_major"] = self._schedule_recurring_events(
            "engine_major", engine_interval
        )
        self.maintenance_schedule["electrical_check"] = self._schedule_recurring_events(
            "electrical_check", electrical_interval
        )
        self.maintenance_schedule["safety_inspection"] = self._schedule_recurring_events(
            "safety_inspection", safety_interval
        )
    
    def _generate_inspection_schedules(self):
        """Generate inspection schedules based on vessel profile"""
        
        # Weekly safety inspections (required by SOLAS)
        self.inspection_schedule["weekly_safety"] = self._schedule_recurring_events(
            "weekly_safety", 7
        )
        
        # Monthly safety drills
        self.inspection_schedule["monthly_drill"] = self._schedule_recurring_events(
            "monthly_drill", 30
        )
        
        # Hull inspections based on vessel profile
        hull_interval = self.vessel_profile.get_maintenance_interval("hull_inspection")
        self.inspection_schedule["hull_inspection"] = self._schedule_recurring_events(
            "hull_inspection", hull_interval
        )
    
    def _schedule_recurring_events(self, event_type: str, interval_days: int) -> List[datetime]:
        """Schedule recurring events at specified intervals"""
        events = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Add some variation based on vessel profile
            procedure_compliance = self.vessel_profile.get_human_factor("procedure_compliance")
            delay_variation = random.uniform(0, 24 * (1 - procedure_compliance))  # 0-24 hour delay
            
            event_time = current_date + timedelta(hours=delay_variation)
            if event_time <= self.end_date:
                events.append(event_time)
            
            current_date += timedelta(days=interval_days)
        
        return events
    
    def get_current_phase(self, timestamp: datetime) -> VoyagePhase:
        """Determine voyage phase at given timestamp"""
        
        # Check if timestamp is during cargo operations
        for cargo_op in self.cargo_operations:
            if cargo_op.start_time <= timestamp <= cargo_op.end_time:
                return VoyagePhase.LOADING if cargo_op.operation_type == "loading" else VoyagePhase.UNLOADING
        
        # Check route segments
        for segment in self.route_segments:
            if segment.start_time <= timestamp <= segment.end_time:
                return segment.phase
        
        # Default to sea transit
        return VoyagePhase.SEA_TRANSIT
    
    def get_weather_condition(self, timestamp: datetime) -> WeatherCondition:
        """Get weather condition at given timestamp"""
        
        # Find applicable route segment
        for segment in self.route_segments:
            if segment.start_time <= timestamp <= segment.end_time:
                return segment.weather_condition
        
        # Default to moderate conditions
        return WeatherCondition.MODERATE
    
    def get_risk_multiplier(self, timestamp: datetime) -> float:
        """Calculate risk multiplier based on current conditions"""
        
        phase = self.get_current_phase(timestamp)
        weather = self.get_weather_condition(timestamp)
        
        # Base multiplier
        multiplier = 1.0
        
        # Phase-specific risk multipliers
        phase_multipliers = {
            VoyagePhase.LOADING: 2.0,        # High risk during cargo operations
            VoyagePhase.UNLOADING: 2.0,      # High risk during cargo operations
            VoyagePhase.DEPARTURE: 1.5,      # Moderate risk during port operations
            VoyagePhase.ARRIVAL: 1.5,        # Moderate risk during port operations
            VoyagePhase.SEA_TRANSIT: 1.0,    # Baseline risk
            VoyagePhase.PORT_OPERATIONS: 1.8  # High risk during port operations
        }
        
        # Weather-specific risk multipliers
        weather_multipliers = {
            WeatherCondition.CALM: 1.0,       # Baseline risk
            WeatherCondition.MODERATE: 1.2,   # Slightly elevated risk
            WeatherCondition.ROUGH: 1.5,      # Elevated risk
            WeatherCondition.SEVERE: 2.0      # High risk
        }
        
        multiplier *= phase_multipliers.get(phase, 1.0)
        multiplier *= weather_multipliers.get(weather, 1.0)
        
        return multiplier
    
    def get_logging_frequency(self, region: str, timestamp: datetime) -> float:
        """Get adjusted logging frequency for region at timestamp"""
        
        base_frequency = self.adjusted_logging_frequencies.get(region, 1.0)
        
        # Apply phase-specific adjustments
        phase = self.get_current_phase(timestamp)
        
        # Cargo operations have higher logging frequency
        if phase in [VoyagePhase.LOADING, VoyagePhase.UNLOADING]:
            if region == "cargo":
                base_frequency *= 3.0  # Much higher during cargo operations
            elif region == "engine_room":
                base_frequency *= 1.5  # Higher engine monitoring during cargo ops
        
        # Port operations have different patterns
        elif phase in [VoyagePhase.DEPARTURE, VoyagePhase.ARRIVAL, VoyagePhase.PORT_OPERATIONS]:
            if region == "bridge":
                base_frequency *= 1.8  # Higher bridge activity in port
            elif region == "deck":
                base_frequency *= 2.0  # Higher deck activity in port
        
        return base_frequency
    
    def get_failure_rate(self, failure_type: str, timestamp: datetime) -> float:
        """Get adjusted failure rate for failure type at timestamp"""
        
        base_rate = self.adjusted_failure_rates.get(failure_type, 0.001)
        risk_multiplier = self.get_risk_multiplier(timestamp)
        
        return base_rate * risk_multiplier
    
    def is_maintenance_scheduled(self, timestamp: datetime, maintenance_type: str) -> bool:
        """Check if maintenance is scheduled at given timestamp"""
        
        scheduled_times = self.maintenance_schedule.get(maintenance_type, [])
        
        # Check if timestamp is within 2 hours of scheduled maintenance
        for scheduled_time in scheduled_times:
            if abs((timestamp - scheduled_time).total_seconds()) <= 7200:  # 2 hours
                return True
        
        return False
    
    def is_inspection_scheduled(self, timestamp: datetime, inspection_type: str) -> bool:
        """Check if inspection is scheduled at given timestamp"""
        
        scheduled_times = self.inspection_schedule.get(inspection_type, [])
        
        # Check if timestamp is within 4 hours of scheduled inspection
        for scheduled_time in scheduled_times:
            if abs((timestamp - scheduled_time).total_seconds()) <= 14400:  # 4 hours
                return True
        
        return False
    
    def get_crew_watch_times(self, date: datetime) -> List[datetime]:
        """Get crew watch times for specific date"""
        
        date_str = date.strftime("%Y-%m-%d")
        return self.crew_watch_schedule.get(date_str, [])
    
    def get_voyage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive voyage statistics"""
        
        duration = self.end_date - self.start_date
        
        # Calculate total distance
        total_distance = sum(segment.distance_nm for segment in self.route_segments)
        
        # Calculate cargo statistics
        total_cargo_loaded = sum(
            op.quantity_tonnes for op in self.cargo_operations 
            if op.operation_type == "loading"
        )
        
        total_cargo_unloaded = sum(
            op.quantity_tonnes for op in self.cargo_operations 
            if op.operation_type == "unloading"
        )
        
        # Calculate phase durations
        phase_durations = {}
        for phase in VoyagePhase:
            phase_duration = sum(
                segment.duration_hours() for segment in self.route_segments 
                if segment.phase == phase
            )
            if phase_duration > 0:
                phase_durations[phase.value] = phase_duration
        
        return {
            "voyage_id": self.voyage_id,
            "vessel_profile": {
                "vessel_id": self.vessel_profile.vessel_id,
                "profile_name": self.vessel_profile.profile_name,
                "vessel_age": self.vessel_profile.vessel_age,
                "company_type": self.vessel_profile.company_type.value,
                "automation_level": self.vessel_profile.automation_level.value
            },
            "duration": {
                "total_days": duration.days,
                "total_hours": duration.total_seconds() / 3600
            },
            "route": {
                "origin": self.origin_port,
                "destination": self.destination_port,
                "total_distance_nm": total_distance,
                "segments": len(self.route_segments)
            },
            "cargo": {
                "total_loaded_tonnes": total_cargo_loaded,
                "total_unloaded_tonnes": total_cargo_unloaded,
                "operations": len(self.cargo_operations)
            },
            "phase_durations_hours": phase_durations,
            "scheduled_events": {
                "maintenance": sum(len(events) for events in self.maintenance_schedule.values()),
                "inspections": sum(len(events) for events in self.inspection_schedule.values())
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert voyage plan to dictionary for serialization"""
        return {
            "voyage_id": self.voyage_id,
            "vessel_profile": self.vessel_profile.to_dict(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "origin_port": self.origin_port,
            "destination_port": self.destination_port,
            "route_segments": [
                {
                    "start_port": seg.start_port,
                    "end_port": seg.end_port,
                    "start_time": seg.start_time.isoformat(),
                    "end_time": seg.end_time.isoformat(),
                    "distance_nm": seg.distance_nm,
                    "planned_speed_kts": seg.planned_speed_kts,
                    "weather_condition": seg.weather_condition.value,
                    "phase": seg.phase.value
                }
                for seg in self.route_segments
            ],
            "cargo_operations": [
                {
                    "operation_type": op.operation_type,
                    "cargo_type": op.cargo_type,
                    "quantity_tonnes": op.quantity_tonnes,
                    "start_time": op.start_time.isoformat(),
                    "end_time": op.end_time.isoformat(),
                    "port": op.port
                }
                for op in self.cargo_operations
            ],
            "adjusted_logging_frequencies": self.adjusted_logging_frequencies,
            "adjusted_failure_rates": self.adjusted_failure_rates
        }


class VoyagePlanGenerator:
    """
    Generates realistic voyage plans with vessel profile integration
    
    This creates authentic maritime voyage plans that incorporate
    vessel-specific variations while maintaining operational realism.
    """
    
    def __init__(self, vessel_profile_manager: VesselProfileManager):
        self.vessel_profile_manager = vessel_profile_manager
        
        # Common routes for bulk carriers
        self.common_routes = [
            {
                "origin": "Port of Newcastle (Australia)",
                "destination": "Port of Qingdao (China)",
                "distance_nm": 3200,
                "typical_duration_days": 14,
                "cargo_type": "Iron Ore"
            },
            {
                "origin": "Port of Hamburg (Germany)",
                "destination": "Port of New York (USA)",
                "distance_nm": 3500,
                "typical_duration_days": 16,
                "cargo_type": "Steel Products"
            },
            {
                "origin": "Port of Santos (Brazil)",
                "destination": "Port of Rotterdam (Netherlands)",
                "distance_nm": 4800,
                "typical_duration_days": 21,
                "cargo_type": "Soybeans"
            },
            {
                "origin": "Port of Vancouver (Canada)",
                "destination": "Port of Yokohama (Japan)",
                "distance_nm": 4200,
                "typical_duration_days": 18,
                "cargo_type": "Grain"
            },
            {
                "origin": "Port of Durban (South Africa)",
                "destination": "Port of Mumbai (India)",
                "distance_nm": 2800,
                "typical_duration_days": 12,
                "cargo_type": "Coal"
            }
        ]
    
    def generate_voyage_plan(self, 
                           vessel_id: str,
                           route_index: Optional[int] = None,
                           start_date: Optional[datetime] = None) -> VoyagePlan:
        """
        Generate a complete voyage plan for a specific vessel
        
        Args:
            vessel_id: ID of the vessel
            route_index: Optional specific route index
            start_date: Optional specific start date
            
        Returns:
            Complete voyage plan with vessel profile integration
        """
        
        # Get vessel profile
        vessel_profile = self.vessel_profile_manager.get_profile(vessel_id)
        if not vessel_profile:
            raise ValueError(f"Vessel profile not found for {vessel_id}")
        
        # Select route
        if route_index is None:
            route_index = random.randint(0, len(self.common_routes) - 1)
        
        route = self.common_routes[route_index]
        
        # Set voyage dates
        if start_date is None:
            start_date = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        
        # Adjust duration based on vessel profile
        base_duration = route["typical_duration_days"]
        
        # Older vessels may be slower
        if vessel_profile.vessel_age > 15:
            duration_adjustment = 1.1  # 10% longer
        elif vessel_profile.vessel_age > 10:
            duration_adjustment = 1.05  # 5% longer
        else:
            duration_adjustment = 1.0
        
        # Weather and route adjustments
        weather_adjustment = random.uniform(0.95, 1.15)  # ±15% for weather
        
        adjusted_duration = base_duration * duration_adjustment * weather_adjustment
        end_date = start_date + timedelta(days=adjusted_duration)
        
        # Generate voyage ID
        voyage_id = f"{vessel_id}_V{random.randint(1000, 9999)}"
        
        # Create voyage plan
        voyage_plan = VoyagePlan(
            voyage_id=voyage_id,
            vessel_profile=vessel_profile,
            start_date=start_date,
            end_date=end_date,
            origin_port=route["origin"],
            destination_port=route["destination"]
        )
        
        # Generate route segments
        voyage_plan.route_segments = self._generate_route_segments(
            route, start_date, end_date, vessel_profile
        )
        
        # Generate cargo operations
        voyage_plan.cargo_operations = self._generate_cargo_operations(
            route, start_date, end_date
        )
        
        return voyage_plan
    
    def _generate_route_segments(self, 
                               route: Dict[str, Any], 
                               start_date: datetime, 
                               end_date: datetime,
                               vessel_profile: VesselProfile) -> List[RouteSegment]:
        """Generate route segments for the voyage"""
        
        segments = []
        current_time = start_date
        
        # Loading phase (2-8 hours depending on cargo and vessel profile)
        loading_duration = random.uniform(4, 12)  # Hours
        
        # Adjust based on vessel profile
        if vessel_profile.automation_level.value == "high":
            loading_duration *= 0.8  # Faster with automation
        elif vessel_profile.automation_level.value == "low":
            loading_duration *= 1.2  # Slower with manual operations
        
        loading_end = current_time + timedelta(hours=loading_duration)
        
        # Departure phase (1-3 hours)
        departure_duration = random.uniform(1, 3)
        departure_end = loading_end + timedelta(hours=departure_duration)
        
        # Sea transit phase (majority of voyage)
        sea_transit_start = departure_end
        
        # Calculate arrival time (leave time for arrival and unloading)
        unloading_duration = random.uniform(6, 14)  # Hours
        arrival_duration = random.uniform(2, 4)  # Hours
        
        sea_transit_end = end_date - timedelta(hours=arrival_duration + unloading_duration)
        
        # Arrival phase
        arrival_end = sea_transit_end + timedelta(hours=arrival_duration)
        
        # Generate weather conditions
        weather_conditions = self._generate_weather_sequence(
            (sea_transit_end - sea_transit_start).days
        )
        
        # Create segments
        segments.append(RouteSegment(
            start_port=route["origin"],
            end_port=route["origin"],
            start_time=current_time,
            end_time=loading_end,
            distance_nm=0,
            planned_speed_kts=0,
            weather_condition=WeatherCondition.CALM,
            phase=VoyagePhase.LOADING
        ))
        
        segments.append(RouteSegment(
            start_port=route["origin"],
            end_port=route["origin"],
            start_time=loading_end,
            end_time=departure_end,
            distance_nm=0,
            planned_speed_kts=5,
            weather_condition=WeatherCondition.CALM,
            phase=VoyagePhase.DEPARTURE
        ))
        
        segments.append(RouteSegment(
            start_port=route["origin"],
            end_port=route["destination"],
            start_time=sea_transit_start,
            end_time=sea_transit_end,
            distance_nm=route["distance_nm"],
            planned_speed_kts=route["distance_nm"] / ((sea_transit_end - sea_transit_start).total_seconds() / 3600),
            weather_condition=random.choice(list(weather_conditions)),
            phase=VoyagePhase.SEA_TRANSIT
        ))
        
        segments.append(RouteSegment(
            start_port=route["destination"],
            end_port=route["destination"],
            start_time=sea_transit_end,
            end_time=arrival_end,
            distance_nm=0,
            planned_speed_kts=8,
            weather_condition=WeatherCondition.CALM,
            phase=VoyagePhase.ARRIVAL
        ))
        
        segments.append(RouteSegment(
            start_port=route["destination"],
            end_port=route["destination"],
            start_time=arrival_end,
            end_time=end_date,
            distance_nm=0,
            planned_speed_kts=0,
            weather_condition=WeatherCondition.CALM,
            phase=VoyagePhase.UNLOADING
        ))
        
        return segments
    
    def _generate_cargo_operations(self, 
                                 route: Dict[str, Any], 
                                 start_date: datetime, 
                                 end_date: datetime) -> List[CargoOperation]:
        """Generate cargo operations for the voyage"""
        
        operations = []
        
        # Cargo quantity (realistic for bulk carriers)
        cargo_quantity = random.uniform(25000, 75000)  # Tonnes
        
        # Loading operation
        loading_start = start_date + timedelta(hours=random.uniform(0.5, 2))
        loading_duration = random.uniform(6, 16)  # Hours
        loading_end = loading_start + timedelta(hours=loading_duration)
        
        operations.append(CargoOperation(
            operation_type="loading",
            cargo_type=route["cargo_type"],
            quantity_tonnes=cargo_quantity,
            start_time=loading_start,
            end_time=loading_end,
            port=route["origin"]
        ))
        
        # Unloading operation
        unloading_duration = random.uniform(8, 18)  # Hours
        unloading_start = end_date - timedelta(hours=unloading_duration)
        
        operations.append(CargoOperation(
            operation_type="unloading",
            cargo_type=route["cargo_type"],
            quantity_tonnes=cargo_quantity,
            start_time=unloading_start,
            end_time=end_date,
            port=route["destination"]
        ))
        
        return operations
    
    def _generate_weather_sequence(self, duration_days: int) -> List[WeatherCondition]:
        """Generate realistic weather sequence for voyage"""
        
        weather_sequence = []
        
        # Start with moderate conditions
        current_weather = WeatherCondition.MODERATE
        
        for day in range(duration_days):
            # Weather tends to persist but can change
            if random.random() < 0.2:  # 20% chance of weather change
                current_weather = random.choice(list(WeatherCondition))
            
            weather_sequence.append(current_weather)
        
        return weather_sequence
    
    def generate_fleet_voyages(self, 
                             voyages_per_vessel: int = 3,
                             start_date_base: Optional[datetime] = None) -> Dict[str, List[VoyagePlan]]:
        """
        Generate multiple voyages for the entire fleet
        
        Args:
            voyages_per_vessel: Number of voyages per vessel
            start_date_base: Base date for voyage generation
            
        Returns:
            Dictionary of voyage plans keyed by vessel_id
        """
        
        if start_date_base is None:
            start_date_base = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        
        fleet_voyages = {}
        
        for vessel_id in self.vessel_profile_manager.profiles.keys():
            vessel_voyages = []
            current_date = start_date_base
            
            for voyage_num in range(voyages_per_vessel):
                # Generate voyage
                voyage = self.generate_voyage_plan(
                    vessel_id=vessel_id,
                    start_date=current_date
                )
                
                vessel_voyages.append(voyage)
                
                # Next voyage starts after current voyage ends + port time
                port_time = random.uniform(24, 72)  # 1-3 days in port
                current_date = voyage.end_date + timedelta(hours=port_time)
            
            fleet_voyages[vessel_id] = vessel_voyages
        
        return fleet_voyages


# # Example usage and testing
# if __name__ == "__main__":
#     # Create vessel profile manager and fleet
#     profile_manager = VesselProfileManager()
#     fleet_profiles = profile_manager.create_fleet_profiles(9)  # 3 of each type
    
#     # Create voyage plan generator
#     voyage_generator = VoyagePlanGenerator(profile_manager)
    
#     # Generate example voyage
#     vessel_id = "MM_01"
#     voyage_plan = voyage_generator.generate_voyage_plan(vessel_id)
    
#     # Display voyage statistics
#     stats = voyage_plan.get_voyage_statistics()
#     print(f"Voyage: {stats['voyage_id']}")
#     print(f"Vessel: {stats['vessel_profile']['vessel_id']} ({stats['vessel_profile']['profile_name']})")
#     print(f"Route: {stats['route']['origin']} → {stats['route']['destination']}")
#     print(f"Duration: {stats['duration']['total_days']} days")
#     print(f"Distance: {stats['route']['total_distance_nm']} nm")
#     print(f"Cargo: {stats['cargo']['total_loaded_tonnes']:.0f} tonnes")
    
#     # Show adjusted parameters
#     print(f"\nAdjusted Parameters:")
#     print(f"Engine room logging frequency: {voyage_plan.adjusted_logging_frequencies['engine_room']:.1f} entries/day")
#     print(f"Propulsion failure rate: {voyage_plan.adjusted_failure_rates['propulsion']:.4f} failures/day")
    
#     # Generate fleet voyages
#     print(f"\nGenerating fleet voyages (3 voyages per vessel)...")
#     fleet_voyages = voyage_generator.generate_fleet_voyages(voyages_per_vessel=3)
    
#     print(f"Generated {sum(len(voyages) for voyages in fleet_voyages.values())} total voyages")
#     print(f"for {len(fleet_voyages)} vessels")