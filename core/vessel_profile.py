"""
Vessel Profile System for Maritime Risk Modeling

This module implements the vessel profile system that creates systematic variations
in vessel operations based on maritime industry characteristics:
- Vessel age affects maintenance and automation
- Company type affects procedures and standards
- Automation level affects logging frequency
- Crew stability affects familiarity and efficiency

Based on research showing "partial but not absolute standardization" in maritime operations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
import random


class CompanyType(Enum):
    """Maritime company categories based on operational standards"""
    MAJOR = "major"      # Large international shipping companies
    MIDTIER = "midtier"  # Regional shipping companies
    SMALL = "small"      # Small operators, tramp shipping


class AutomationLevel(Enum):
    """Vessel automation levels affecting operational procedures"""
    HIGH = "high"        # Modern integrated bridge systems
    MEDIUM = "medium"    # Partial automation with manual backup
    LOW = "low"         # Traditional manual operations


class CrewStability(Enum):
    """Crew stability levels affecting operational efficiency"""
    STABLE = "stable"    # Long-term crew with vessel familiarity
    ROTATING = "rotating"  # Regular crew rotation schedules
    MIXED = "mixed"      # High turnover, mixed experience levels


@dataclass
class VesselProfile:
    """
    Defines operational characteristics for a specific vessel
    
    This creates systematic variations in vessel operations without breaking
    the fundamental voyage-plan-driven data generation approach.
    """
    
    # Core vessel characteristics
    vessel_id: str
    vessel_age: int  # Years since build
    company_type: CompanyType
    automation_level: AutomationLevel
    crew_stability: CrewStability
    
    # Operational parameters
    profile_name: str
    description: str
    
    # Failure rate multipliers (applied to base MARCAT rates)
    failure_multipliers: Dict[str, float]
    
    # Logging frequency multipliers
    logging_multipliers: Dict[str, float]
    
    # Maintenance schedule variations
    maintenance_intervals: Dict[str, int]  # Days between maintenance
    
    # Human factors variations
    human_factors: Dict[str, float]
    
    def __post_init__(self):
        """Validate profile consistency"""
        if self.vessel_age < 0 or self.vessel_age > 30:
            raise ValueError(f"Vessel age {self.vessel_age} outside realistic range (0-30 years)")
        
        # Validate multipliers are positive
        for category, multiplier in self.failure_multipliers.items():
            if multiplier <= 0:
                raise ValueError(f"Failure multiplier for {category} must be positive")
    
    @classmethod
    def create_modern_major(cls, vessel_id: str) -> 'VesselProfile':
        """
        Modern vessel operated by major shipping company
        
        Characteristics:
        - Age: 3-7 years (newer fleet)
        - High automation reduces human error
        - Stable crew with good training
        - Lower failure rates due to modern equipment
        """
        return cls(
            vessel_id=vessel_id,
            vessel_age=random.randint(3, 7),
            company_type=CompanyType.MAJOR,
            automation_level=AutomationLevel.HIGH,
            crew_stability=CrewStability.STABLE,
            profile_name="Modern Major",
            description="New vessel, major company, high automation, stable crew",
            
            # Lower failure rates due to modern equipment and procedures
            failure_multipliers={
                "propulsion": 0.7,      # Modern engines more reliable
                "electrical": 0.8,      # Better electrical systems
                "navigation": 0.6,      # Advanced navigation systems
                "human_error": 0.6,     # Better training, stable crew
                "fire_explosion": 0.5,  # Modern fire suppression
                "structural": 0.8,      # Newer hull and structure
                "machinery": 0.7        # Modern machinery
            },
            
            # Higher logging frequency due to automation
            logging_multipliers={
                "engine_room": 1.2,     # More automated monitoring
                "bridge": 1.3,          # Advanced navigation logging
                "cargo": 1.1,           # Better cargo monitoring
                "safety": 1.2,          # Comprehensive safety systems
                "deck": 1.0,            # Standard deck operations
                "accommodation": 1.0,   # Standard crew activities
                "pump_room": 1.1        # Better monitoring systems
            },
            
            # More frequent maintenance (preventive approach)
            maintenance_intervals={
                "engine_major": 180,    # Days between major engine maintenance
                "electrical_check": 30, # Electrical system checks
                "safety_inspection": 7, # Safety equipment checks
                "hull_inspection": 90,  # Hull and structure checks
                "cargo_system": 14      # Cargo handling system checks
            },
            
            # Human factors (positive values = better performance)
            human_factors={
                "crew_familiarity": 0.9,    # High familiarity with vessel
                "training_level": 0.9,      # Excellent training programs
                "fatigue_resistance": 0.8,  # Better rest management
                "communication": 0.9,       # Clear communication protocols
                "procedure_compliance": 0.9 # High compliance with procedures
            }
        )
    
    @classmethod
    def create_aging_midtier(cls, vessel_id: str) -> 'VesselProfile':
        """
        Aging vessel operated by mid-tier shipping company
        
        Characteristics:
        - Age: 10-15 years (middle-aged fleet)
        - Medium automation with some legacy systems
        - Rotating crew with standard training
        - Baseline failure rates
        """
        return cls(
            vessel_id=vessel_id,
            vessel_age=random.randint(10, 15),
            company_type=CompanyType.MIDTIER,
            automation_level=AutomationLevel.MEDIUM,
            crew_stability=CrewStability.ROTATING,
            profile_name="Aging Midtier",
            description="Aging vessel, midtier company, medium automation, rotating crew",
            
            # Baseline failure rates (reference values)
            failure_multipliers={
                "propulsion": 1.0,      # Baseline propulsion reliability
                "electrical": 1.2,      # Aging electrical systems
                "navigation": 1.0,      # Standard navigation equipment
                "human_error": 1.0,     # Baseline human error rates
                "fire_explosion": 1.0,  # Standard fire safety
                "structural": 1.1,      # Some structural aging
                "machinery": 1.1        # Aging machinery
            },
            
            # Standard logging frequency
            logging_multipliers={
                "engine_room": 1.0,     # Standard engine room logging
                "bridge": 1.0,          # Standard bridge procedures
                "cargo": 1.0,           # Standard cargo operations
                "safety": 1.0,          # Standard safety procedures
                "deck": 1.0,            # Standard deck operations
                "accommodation": 1.0,   # Standard crew activities
                "pump_room": 1.0        # Standard pump room operations
            },
            
            # Standard maintenance intervals
            maintenance_intervals={
                "engine_major": 210,    # Standard major maintenance
                "electrical_check": 45, # Standard electrical checks
                "safety_inspection": 7, # Required safety checks
                "hull_inspection": 120, # Standard hull inspection
                "cargo_system": 21      # Standard cargo system checks
            },
            
            # Standard human factors
            human_factors={
                "crew_familiarity": 0.7,    # Moderate familiarity
                "training_level": 0.7,      # Standard training programs
                "fatigue_resistance": 0.7,  # Standard fatigue management
                "communication": 0.7,       # Standard communication
                "procedure_compliance": 0.7 # Standard procedure compliance
            }
        )
    
    @classmethod
    def create_legacy_small(cls, vessel_id: str) -> 'VesselProfile':
        """
        Legacy vessel operated by small shipping company
        
        Characteristics:
        - Age: 16-25 years (older fleet)
        - Low automation, mostly manual operations
        - Mixed crew with varying experience
        - Higher failure rates due to aging equipment
        """
        return cls(
            vessel_id=vessel_id,
            vessel_age=random.randint(16, 25),
            company_type=CompanyType.SMALL,
            automation_level=AutomationLevel.LOW,
            crew_stability=CrewStability.MIXED,
            profile_name="Legacy Small",
            description="Older vessel, small company, low automation, mixed crew",
            
            # Higher failure rates due to aging equipment
            failure_multipliers={
                "propulsion": 1.5,      # Aging propulsion systems
                "electrical": 1.8,      # Old electrical systems
                "navigation": 1.4,      # Older navigation equipment
                "human_error": 1.4,     # Less training, crew turnover
                "fire_explosion": 1.6,  # Older fire safety systems
                "structural": 1.3,      # Structural aging effects
                "machinery": 1.5        # Aging machinery
            },
            
            # Lower logging frequency due to manual operations
            logging_multipliers={
                "engine_room": 0.8,     # Less automated monitoring
                "bridge": 0.9,          # Traditional navigation logging
                "cargo": 0.9,           # Manual cargo operations
                "safety": 0.8,          # Basic safety procedures
                "deck": 0.9,            # Manual deck operations
                "accommodation": 1.0,   # Standard crew activities
                "pump_room": 0.8        # Manual pump room operations
            },
            
            # Extended maintenance intervals (cost-driven)
            maintenance_intervals={
                "engine_major": 270,    # Extended major maintenance
                "electrical_check": 60, # Less frequent electrical checks
                "safety_inspection": 7, # Minimum required safety checks
                "hull_inspection": 150, # Extended hull inspection
                "cargo_system": 28      # Extended cargo system checks
            },
            
            # Challenging human factors
            human_factors={
                "crew_familiarity": 0.5,    # Low familiarity due to turnover
                "training_level": 0.5,      # Limited training programs
                "fatigue_resistance": 0.6,  # Challenging fatigue management
                "communication": 0.5,       # Communication challenges
                "procedure_compliance": 0.5 # Lower procedure compliance
            }
        )
    
    def get_failure_rate(self, failure_type: str, base_rate: float) -> float:
        """
        Calculate adjusted failure rate for this vessel profile
        
        Args:
            failure_type: Type of failure (e.g., "propulsion", "electrical")
            base_rate: Base failure rate from MARCAT data
            
        Returns:
            Adjusted failure rate for this vessel
        """
        multiplier = self.failure_multipliers.get(failure_type, 1.0)
        return base_rate * multiplier
    
    def get_logging_frequency(self, region: str, base_frequency: float) -> float:
        """
        Calculate adjusted logging frequency for this vessel profile
        
        Args:
            region: Operational region (e.g., "engine_room", "bridge")
            base_frequency: Base logging frequency
            
        Returns:
            Adjusted logging frequency for this vessel
        """
        multiplier = self.logging_multipliers.get(region, 1.0)
        return base_frequency * multiplier
    
    def get_maintenance_interval(self, maintenance_type: str) -> int:
        """
        Get maintenance interval for this vessel profile
        
        Args:
            maintenance_type: Type of maintenance
            
        Returns:
            Maintenance interval in days
        """
        return self.maintenance_intervals.get(maintenance_type, 30)
    
    def get_human_factor(self, factor_type: str) -> float:
        """
        Get human factor adjustment for this vessel profile
        
        Args:
            factor_type: Type of human factor
            
        Returns:
            Human factor adjustment (0-1 scale)
        """
        return self.human_factors.get(factor_type, 0.7)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization"""
        return {
            "vessel_id": self.vessel_id,
            "vessel_age": self.vessel_age,
            "company_type": self.company_type.value,
            "automation_level": self.automation_level.value,
            "crew_stability": self.crew_stability.value,
            "profile_name": self.profile_name,
            "description": self.description,
            "failure_multipliers": self.failure_multipliers,
            "logging_multipliers": self.logging_multipliers,
            "maintenance_intervals": self.maintenance_intervals,
            "human_factors": self.human_factors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VesselProfile':
        """Create profile from dictionary"""
        return cls(
            vessel_id=data["vessel_id"],
            vessel_age=data["vessel_age"],
            company_type=CompanyType(data["company_type"]),
            automation_level=AutomationLevel(data["automation_level"]),
            crew_stability=CrewStability(data["crew_stability"]),
            profile_name=data["profile_name"],
            description=data["description"],
            failure_multipliers=data["failure_multipliers"],
            logging_multipliers=data["logging_multipliers"],
            maintenance_intervals=data["maintenance_intervals"],
            human_factors=data["human_factors"]
        )


class VesselProfileManager:
    """
    Manages vessel profiles for fleet-wide dataset generation
    
    This provides consistent vessel profile creation and management
    for generating fleet datasets with systematic variations.
    """
    
    def __init__(self):
        self.profiles: Dict[str, VesselProfile] = {}
    
    def create_fleet_profiles(self, fleet_size: int = 15) -> Dict[str, VesselProfile]:
        """
        Create a balanced fleet of vessel profiles
        
        Args:
            fleet_size: Number of vessels to create
            
        Returns:
            Dictionary of vessel profiles keyed by vessel_id
        """
        profiles = {}
        
        # Create balanced distribution across profile types
        profiles_per_type = fleet_size // 3
        remainder = fleet_size % 3
        
        # Modern Major vessels (5 vessels)
        for i in range(profiles_per_type + (1 if remainder > 0 else 0)):
            vessel_id = f"MM_{i+1:02d}"  # MM_01, MM_02, etc.
            profiles[vessel_id] = VesselProfile.create_modern_major(vessel_id)
        
        # Aging Midtier vessels (5 vessels)
        for i in range(profiles_per_type + (1 if remainder > 1 else 0)):
            vessel_id = f"AM_{i+1:02d}"  # AM_01, AM_02, etc.
            profiles[vessel_id] = VesselProfile.create_aging_midtier(vessel_id)
        
        # Legacy Small vessels (5 vessels)
        for i in range(profiles_per_type):
            vessel_id = f"LS_{i+1:02d}"  # LS_01, LS_02, etc.
            profiles[vessel_id] = VesselProfile.create_legacy_small(vessel_id)
        
        self.profiles = profiles
        return profiles
    
    def get_profile(self, vessel_id: str) -> Optional[VesselProfile]:
        """Get profile for specific vessel"""
        return self.profiles.get(vessel_id)
    
    def get_profiles_by_type(self, profile_type: str) -> list[VesselProfile]:
        """Get all profiles of a specific type"""
        return [p for p in self.profiles.values() if profile_type.lower() in p.profile_name.lower()]
    
    def get_fleet_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of the fleet"""
        if not self.profiles:
            return {"error": "No profiles created"}
        
        ages = [p.vessel_age for p in self.profiles.values()]
        profile_types = {}
        
        for profile in self.profiles.values():
            profile_types[profile.profile_name] = profile_types.get(profile.profile_name, 0) + 1
        
        return {
            "total_vessels": len(self.profiles),
            "age_statistics": {
                "min_age": min(ages),
                "max_age": max(ages),
                "avg_age": sum(ages) / len(ages)
            },
            "profile_distribution": profile_types,
            "company_distribution": {
                ct.value: len([p for p in self.profiles.values() if p.company_type == ct])
                for ct in CompanyType
            }
        }


# # Example usage and testing
# if __name__ == "__main__":
#     # Create vessel profile manager
#     manager = VesselProfileManager()
    
#     # Create fleet profiles
#     fleet_profiles = manager.create_fleet_profiles(15)
    
#     # Display fleet statistics
#     stats = manager.get_fleet_statistics()
#     print("Fleet Statistics:")
#     print(f"Total vessels: {stats['total_vessels']}")
#     print(f"Age range: {stats['age_statistics']['min_age']}-{stats['age_statistics']['max_age']} years")
#     print(f"Average age: {stats['age_statistics']['avg_age']:.1f} years")
#     print("\nProfile distribution:")
#     for profile_type, count in stats['profile_distribution'].items():
#         print(f"  {profile_type}: {count} vessels")
    
#     # Example: Get failure rate for specific vessel
#     vessel_id = "MM_01"
#     profile = manager.get_profile(vessel_id)
#     if profile:
#         base_propulsion_rate = 0.019  # Base MARCAT rate: 6.96/year → 0.019/day
#         adjusted_rate = profile.get_failure_rate("propulsion", base_propulsion_rate)
#         print(f"\n{vessel_id} ({profile.profile_name}):")
#         print(f"  Base propulsion failure rate: {base_propulsion_rate:.3f}/day")
#         print(f"  Adjusted failure rate: {adjusted_rate:.3f}/day")
#         print(f"  Multiplier: {profile.failure_multipliers['propulsion']}")
