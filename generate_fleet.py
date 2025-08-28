"""
New Fleet Dataset Generation with Controlled Cargo Distribution
Maritime Vessel Operational Risk Modeling Project

This generates complete fleet datasets ensuring:
1. Deliberate 33/33/33 cargo type distribution (iron ore, coal, grain)
2. Voyage consistency - same cargo type across all operational areas
3. Authentic vessel profiles with systematic variations
4. Proper cargo-specific monitoring patterns from cargo.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import json
import os
from pathlib import Path
import random
import itertools
from collections import defaultdict

# Import our custom modules
from core.vessel_profile import VesselProfile, VesselProfileManager
from core.voyage_plan import VoyagePlanGenerator, VoyagePlan

# Import generators
try:
    from generators.engine_room import EnhancedEngineRoomGenerator
    from generators.bridge import EnhancedBridgeGenerator  # Correct class name
    from generators.cargo import EnhancedCargoOperationsGenerator
    from generators.deck import EnhancedDeckOperationsGenerator 
    from generators.auxilliary_support import EnhancedAuxiliarySafetySystemsGenerator
    from generators.life_support_systems import EnhancedLifeSupportSystemsGenerator
    # from core.simple_cascade_coordinator import apply_inter_regional_cascades, get_cascade_summary
  # Your existing Cargo.py
except ImportError as e:
    print(f"Warning: Some generators not available: {e}")


def apply_inter_regional_cascades(datasets: Dict[str, pd.DataFrame], vessel_profile: str) -> Dict[str, pd.DataFrame]:
    """Apply inter-regional cascade effects - EMBEDDED VERSION"""
    
    if not datasets or len(datasets) < 2:
        return datasets
    
    print(f"🔗 Applying validated maritime cascades across {len(datasets)} regions...")
    
    # Vessel profile multipliers
    multipliers = {
        'modern_major': {'cascade_probability': 0.7, 'cascade_severity': 0.8},
        'aging_midtier': {'cascade_probability': 1.0, 'cascade_severity': 1.0},
        'legacy_small': {'cascade_probability': 1.2, 'cascade_severity': 1.15}
    }
    
    multiplier = multipliers.get(vessel_profile, multipliers['aging_midtier'])
    
    # Apply cascades
    enhanced_datasets = {}
    total_cascades = 0
    
    for region, dataset in datasets.items():
        if dataset.empty:
            enhanced_datasets[region] = dataset
            continue
            
        enhanced_df = dataset.copy()
        
        # Apply cascades from other regions
        for source_region, source_dataset in datasets.items():
            if source_region != region and not source_dataset.empty:
                cascades = _apply_region_cascade(enhanced_df, region, source_dataset, source_region, multiplier)
                total_cascades += cascades
        
        enhanced_datasets[region] = enhanced_df
    
    print(f"   ⚡ Applied {total_cascades} cascade effects")
    return enhanced_datasets


def _apply_region_cascade(target_df: pd.DataFrame, target_region: str, 
                         source_df: pd.DataFrame, source_region: str, multiplier: Dict[str, float]) -> int:
    """Apply cascade from source to target region"""
    
    # Find failures in source
    source_failures = []
    
    if 'active_failures' in source_df.columns:
        failure_rows = source_df[source_df['active_failures'].notna() & (source_df['active_failures'] != '')]
        for idx, row in failure_rows.iterrows():
            source_failures.append({
                'timestamp': row['timestamp'],
                'severity': row.get('risk_score', 0.5)
            })
    
    if 'risk_score' in source_df.columns:
        high_risk = source_df[source_df['risk_score'] > 0.6]
        for idx, row in high_risk.iterrows():
            if not any(f['timestamp'] == row['timestamp'] for f in source_failures):
                source_failures.append({
                    'timestamp': row['timestamp'],
                    'severity': row['risk_score']
                })
    
    if not source_failures:
        return 0
    
    # Get cascade rules
    cascade_rules = _get_simple_cascade_rules()
    
    if source_region not in cascade_rules or target_region not in cascade_rules[source_region]:
        return 0
    
    effects = cascade_rules[source_region][target_region]
    cascades_applied = 0
    
    for failure_info in source_failures:
        # Find matching timestamps (within 2 hours)
        failure_time = pd.to_datetime(failure_info['timestamp'])
        target_df['temp_timestamp'] = pd.to_datetime(target_df['timestamp'])
        
        time_mask = abs(target_df['temp_timestamp'] - failure_time) <= pd.Timedelta(hours=2)
        matching_rows = target_df[time_mask].index
        
        if len(matching_rows) > 0:
            severity = failure_info['severity'] * multiplier['cascade_severity']
            applied = _apply_effects(target_df, matching_rows, effects, severity, source_region, multiplier)
            cascades_applied += applied
    
    target_df.drop('temp_timestamp', axis=1, inplace=True, errors='ignore')
    return cascades_applied


def _get_simple_cascade_rules() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Simplified cascade rules for direct integration"""
    
    return {
        'engine_room': {
            'bridge': {
                'gps_accuracy_m': {'add': 8, 'probability': 0.8},
                'navigation_equipment_failures': {'increase': 2, 'probability': 0.9},
                'speed_over_ground_knots': {'multiply': 0.2, 'probability': 0.95}
            },
            'deck': {
                'deck_cranes_operational_count': {'multiply': 0.1, 'probability': 0.95},
                'operations_formally_halted': {'set_value': 1, 'probability': 0.85}
            },
            'cargo': {
                'ballast_tank_1_level_pct': {'add': -15, 'probability': 0.9},
                'cargo_shift_risk_indicator': {'multiply': 1.8, 'probability': 0.8}
            },
            'auxiliary_support': {
                'fire_detection_reliability': {'multiply': 0.6, 'probability': 0.8}
            },
            'life_support_systems': {
                'hvac_efficiency': {'multiply': 0.3, 'probability': 0.8},
                'crew_fatigue_level': {'add': 0.25, 'probability': 0.9}
            }
        },
        'bridge': {
            'deck': {
                'operations_formally_halted': {'set_value': 1, 'probability': 0.7}
            },
            'cargo': {
                'loading_rate_tph': {'multiply': 0.6, 'probability': 0.6}
            }
        },
        'cargo': {
            'bridge': {
                'speed_over_ground_knots': {'multiply': 0.5, 'probability': 0.8},
                'course_over_ground': {'add': 5, 'probability': 0.7}
            },
            'deck': {
                'operations_formally_halted': {'set_value': 1, 'probability': 0.9}
            }
        },
        'deck': {
            'cargo': {
                'loading_rate_tph': {'multiply': 0.2, 'probability': 0.95}
            }
        },
        'auxiliary_support': {
            'engine_room': {
                'main_engine_operating_temperature_c': {'add': 15, 'probability': 0.7}
            },
            'life_support_systems': {
                'crew_fatigue_level': {'add': 0.2, 'probability': 0.8}
            }
        },
        'life_support_systems': {
            'bridge': {
                'watch_alertness_score': {'multiply': 0.7, 'probability': 0.8}
            },
            'engine_room': {
                'procedure_compliance_factor': {'multiply': 0.8, 'probability': 0.7}
            },
            'deck': {
                'procedure_compliance_score': {'multiply': 0.75, 'probability': 0.8}
            },
            'cargo': {
                'cargo_shift_risk_indicator': {'multiply': 1.3, 'probability': 0.6}
            }
        }
    }


def _apply_effects(df: pd.DataFrame, row_indices: List[int], effects: Dict[str, Dict], 
                  severity: float, source_region: str, multiplier: Dict[str, float]) -> int:
    """Apply cascade effects to dataframe rows"""
    
    applied_count = 0
    
    for column, effect_config in effects.items():
        if column not in df.columns:
            continue
            
        probability = effect_config['probability'] * multiplier['cascade_probability']
        if random.random() > probability:
            continue
        
        for idx in row_indices:
            current_value = df.at[idx, column]
            
            try:
                if 'multiply' in effect_config:
                    factor = effect_config['multiply']
                    if factor < 1:
                        factor = 1 - (1 - factor) * severity
                    else:
                        factor = 1 + (factor - 1) * severity
                    
                    if isinstance(current_value, (int, float)) and not pd.isna(current_value):
                        new_value = current_value * factor
                        if column.endswith('_pct') or column.endswith('_percentage'):
                            new_value = max(0, min(100, new_value))
                        elif column.endswith('_operational') or column.endswith('_count'):
                            new_value = max(0, int(new_value))
                        df.at[idx, column] = new_value
                        applied_count += 1
                
                elif 'add' in effect_config:
                    addition = effect_config['add'] * severity
                    if isinstance(current_value, (int, float)) and not pd.isna(current_value):
                        new_value = current_value + addition
                        if column.endswith('_pct'):
                            new_value = max(0, min(100, new_value))
                        df.at[idx, column] = new_value
                        applied_count += 1
                
                elif 'increase' in effect_config:
                    increase = int(effect_config['increase'] * severity)
                    if isinstance(current_value, (int, float)) and not pd.isna(current_value):
                        df.at[idx, column] = current_value + increase
                        applied_count += 1
                
                elif 'set_value' in effect_config:
                    df.at[idx, column] = effect_config['set_value']
                    applied_count += 1
                    
            except:
                continue
    
    # Add cascade tracking
    if 'cascade_active' not in df.columns:
        df['cascade_active'] = False
    if 'cascade_source' not in df.columns:
        df['cascade_source'] = ''
    
    for idx in row_indices:
        df.at[idx, 'cascade_active'] = True
        current_source = df.at[idx, 'cascade_source']
        new_source = f"{source_region}_cascade"
        if current_source and current_source != '':
            df.at[idx, 'cascade_source'] = f"{current_source},{new_source}"
        else:
            df.at[idx, 'cascade_source'] = new_source
    
    return applied_count


def get_cascade_summary(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Get cascade summary"""
    
    total_cascade_records = 0
    regions_with_cascades = 0
    
    for region, df in datasets.items():
        if 'cascade_active' in df.columns:
            cascade_count = df['cascade_active'].sum()
            if cascade_count > 0:
                regions_with_cascades += 1
                total_cascade_records += int(cascade_count)
    
    return {
        'total_cascade_records': total_cascade_records,
        'regions_with_cascades': regions_with_cascades
    }




class ControlledFleetGenerator:
    """
    Controlled Fleet Dataset Generator with Deliberate Cargo Distribution
    
    Ensures maritime authenticity by:
    - Controlling cargo type distribution (33/33/33)
    - Maintaining voyage consistency across all operational areas
    - Using cargo.py as the authoritative source for cargo operations
    """
    
    def __init__(self, output_dir: str = "controlled_fleet_datasets"):
        """Initialize controlled fleet generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize core systems
        self.profile_manager = VesselProfileManager()
        self.voyage_generator = VoyagePlanGenerator(self.profile_manager)
        
        # Storage for generated data
        self.fleet_profiles = {}
        self.fleet_voyages = {}
        self.cargo_assignments = {}  # Track cargo type assignments
        self.fleet_datasets = {}
        
        # Cargo types with equal distribution
        self.cargo_types = ["iron_ore", "coal", "grain"]
        
        # Statistics tracking
        self.generation_stats = {
            'vessels_created': 0,
            'voyages_created': 0,
            'cargo_distribution': {cargo: 0 for cargo in self.cargo_types},
            'datasets_created': 0,
            'total_log_entries': 0,
            'generation_time': 0
        }
        
        print(f"🚢 Controlled Fleet Generator initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📦 Cargo types: {', '.join(self.cargo_types)}")
    
    def generate_fleet_profiles(self, fleet_size: int = 9) -> Dict[str, VesselProfile]:
        """Generate vessel profiles for the fleet"""
        print(f"\n🏭 Generating {fleet_size} vessel profiles...")
        
        self.fleet_profiles = self.profile_manager.create_fleet_profiles(fleet_size)
        self.generation_stats['vessels_created'] = len(self.fleet_profiles)
        
        # Display fleet composition
        profile_counts = defaultdict(int)
        for profile in self.fleet_profiles.values():
            profile_counts[profile.profile_name] += 1
        
        print(f"Fleet composition:")
        for profile_type, count in profile_counts.items():
            print(f"  {profile_type}: {count} vessels")
        
        return self.fleet_profiles
    
    def create_controlled_cargo_assignments(self, 
                                          voyages_per_vessel: int = 3) -> Dict[str, List[str]]:
        """
        Create controlled cargo type assignments ensuring 33/33/33 distribution
        
        Args:
            voyages_per_vessel: Number of voyages per vessel
            
        Returns:
            Dictionary mapping vessel_id to list of cargo types for each voyage
        """
        if not self.fleet_profiles:
            raise ValueError("Fleet profiles not generated. Call generate_fleet_profiles() first.")
        
        total_voyages = len(self.fleet_profiles) * voyages_per_vessel
        print(f"\n📦 Creating controlled cargo assignments for {total_voyages} total voyages...")
        
        # Calculate target distribution
        voyages_per_cargo = total_voyages // len(self.cargo_types)
        remainder = total_voyages % len(self.cargo_types)
        
        # Create cargo assignment pool
        cargo_pool = []
        for i, cargo_type in enumerate(self.cargo_types):
            count = voyages_per_cargo + (1 if i < remainder else 0)
            cargo_pool.extend([cargo_type] * count)
            self.generation_stats['cargo_distribution'][cargo_type] = count
        
        # Shuffle for random distribution across vessels
        random.shuffle(cargo_pool)
        
        # Assign cargo types to voyages
        cargo_assignments = {}
        pool_index = 0
        
        for vessel_id in self.fleet_profiles.keys():
            vessel_cargo_types = []
            for _ in range(voyages_per_vessel):
                vessel_cargo_types.append(cargo_pool[pool_index])
                pool_index += 1
            cargo_assignments[vessel_id] = vessel_cargo_types
        
        self.cargo_assignments = cargo_assignments
        
        # Display distribution
        print(f"Cargo distribution:")
        for cargo_type, count in self.generation_stats['cargo_distribution'].items():
            percentage = (count / total_voyages) * 100
            print(f"  {cargo_type}: {count} voyages ({percentage:.1f}%)")
        
        return cargo_assignments
    
    def _find_route_for_cargo_type(self, target_cargo_type: str) -> Optional[int]:
        """
        Find route index that matches the target cargo type
        
        Args:
            target_cargo_type: Desired cargo type (iron_ore, coal, grain)
            
        Returns:
            Route index that provides the target cargo type, or None if not found
        """
        # Map our cargo types to route cargo types
        cargo_type_mapping = {
            "iron_ore": "Iron Ore",
            "coal": "Coal", 
            "grain": "Grain"
        }
        
        route_cargo_type = cargo_type_mapping.get(target_cargo_type)
        if not route_cargo_type:
            print(f"Warning: Unknown cargo type {target_cargo_type}, using random route")
            return None
        
        # Search through available routes
        if hasattr(self.voyage_generator, 'common_routes'):
            for idx, route in enumerate(self.voyage_generator.common_routes):
                if route.get('cargo_type') == route_cargo_type:
                    return idx
        
        print(f"Warning: No route found for cargo type {route_cargo_type}, using random route")
        return None
    def generate_fleet_voyages_with_cargo(self, 
                                    voyages_per_vessel: int = 3) -> Dict[str, List[VoyagePlan]]:
        """
        Generate voyage plans with controlled cargo type assignments
        
        Args:
            voyages_per_vessel: Number of voyages per vessel
            
        Returns:
            Dictionary of voyage plans by vessel with assigned cargo types
        """
        if not self.cargo_assignments:
            self.create_controlled_cargo_assignments(voyages_per_vessel)
        
        print(f"\n🗺️ Generating {voyages_per_vessel} voyages per vessel with cargo assignments...")
        
        # 🔍 DEBUG: Check cargo assignments first
        print(f"\n🔍 CARGO ASSIGNMENTS DEBUG:")
        for vessel_id, cargo_list in self.cargo_assignments.items():
            print(f"  {vessel_id}: {cargo_list} ({len(cargo_list)} voyages)")
        
        fleet_voyages = {}
        base_start_date = datetime(2024, 1, 15, 8, 0, 0)
        
        # 🔍 DEBUG: Track voyage generation
        total_expected = len(self.cargo_assignments) * voyages_per_vessel
        total_generated = 0
        
        for vessel_id, cargo_types in self.cargo_assignments.items():
            # 🔍 DEBUG: Add these lines
            print(f"\n🚢 Generating {len(cargo_types)} voyages for {vessel_id}")
            total_expected = len(cargo_types)
            vessel_generated = 0
            
            vessel_voyages = []
            current_date = base_start_date
            
            print(f"\n🚢 Generating {len(cargo_types)} voyages for {vessel_id}")
            
            for voyage_idx, assigned_cargo_type in enumerate(cargo_types):
                try:
                    # 🔍 DEBUG: Add this line at start of loop
                    print(f"  Attempting voyage {voyage_idx + 1} ({assigned_cargo_type})...")
                    # Find route that matches the assigned cargo type
                    target_route_index = self._find_route_for_cargo_type(assigned_cargo_type)
                    
                    # Generate voyage plan with specific route (which determines cargo type)
                    voyage_plan = self.voyage_generator.generate_voyage_plan(
                        vessel_id=vessel_id,
                        route_index=target_route_index,  # Use specific route for cargo type
                        start_date=current_date
                    )
                    # 🔍 DEBUG: Add these lines after voyage_plan creation
                    if voyage_plan:
                        vessel_generated += 1
                        print(f"  ✅ SUCCESS: {voyage_plan.voyage_id}")
                    else:
                        print(f"  ❌ FAILED: No voyage plan created")
                    
                    # Ensure cargo type is set correctly (backup method)
                    if hasattr(voyage_plan, 'cargo_operations') and voyage_plan.cargo_operations:
                        for cargo_op in voyage_plan.cargo_operations:
                            if cargo_op.cargo_type != assigned_cargo_type:
                                print(f"Warning: Adjusting cargo type from {cargo_op.cargo_type} to {assigned_cargo_type}")
                                cargo_op.cargo_type = assigned_cargo_type
                    
                    # Set voyage-level cargo type attribute for generators
                    voyage_plan.cargo_type = assigned_cargo_type
                    
                    vessel_voyages.append(voyage_plan)
                    total_generated += 1
                    
                    print(f"  ✅ Voyage {voyage_idx + 1}: {voyage_plan.voyage_id} ({assigned_cargo_type})")
                    
                    # Next voyage starts after current voyage ends + port time
                    port_time = random.uniform(24, 72)  # 1-3 days in port
                    current_date = voyage_plan.end_date + timedelta(hours=port_time)
                    
                except Exception as e:
                    print(f"  ❌ FAILED Voyage {voyage_idx + 1} for {vessel_id}: {e}")
                    import traceback
                    traceback.print_exc()  # Print full error details
                    continue  # This might be dropping voyages!
            
            fleet_voyages[vessel_id] = vessel_voyages
            print(f"  📊 {vessel_id}: Generated {len(vessel_voyages)} of {len(cargo_types)} expected voyages")
        
        # 🔍 DEBUG: Final voyage count summary
        print(f"\n📊 VOYAGE GENERATION SUMMARY:")
        print(f"  Expected: {total_expected} voyages")
        print(f"  Generated: {total_generated} voyages")
        print(f"  Missing: {total_expected - total_generated} voyages")
        
        self.fleet_voyages = fleet_voyages
        total_voyages = sum(len(voyages) for voyages in fleet_voyages.values())
        self.generation_stats['voyages_created'] = total_voyages
        
        print(f"Generated {total_voyages} total voyages with controlled cargo distribution")

                # 🔍 DEBUG: Add these lines before return
        print(f"\n📊 FINAL VOYAGE COUNT DEBUG:")
        total_voyages_debug = 0
        for vessel_id, voyages in fleet_voyages.items():
            print(f"  {vessel_id}: {len(voyages)} voyages")
            total_voyages_debug += len(voyages)
        print(f"  TOTAL: {total_voyages_debug} voyages")
        
        return fleet_voyages
    
    def generate_voyage_datasets(self, 
                               vessel_id: str, 
                               voyage_plan: VoyagePlan,
                               operational_areas: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate datasets for all operational areas for a single voyage
        Ensures all areas use the same cargo type and voyage parameters
        
        Args:
            vessel_id: Vessel identifier
            voyage_plan: Voyage plan with cargo type
            operational_areas: List of areas to generate data for
            
        Returns:
            Dictionary of datasets by operational area
        """
        
        if operational_areas is None:
            operational_areas = ['engine_room', 'bridge', 'cargo', 'deck', 'auxiliary_support', 'life_support_systems']        
        voyage_datasets = {}
        cargo_type = getattr(voyage_plan, 'cargo_type', 'iron_ore')
        
        print(f"    Generating {len(operational_areas)} operational areas for {voyage_plan.voyage_id} ({cargo_type})")

        # DEBUG: Check what we're passing to cargo.py
        print(f"DEBUG: vessel_id = {vessel_id}")
        print(f"DEBUG: voyage_plan.vessel_profile = {getattr(voyage_plan, 'vessel_profile', 'MISSING!')}")
        print(f"DEBUG: cargo_type = {getattr(voyage_plan, 'cargo_type', 'MISSING!')}")
        
        for area in operational_areas:
            try:
                if area == 'auxiliary_support':
                    uncertainty_mode = True 
                    generator = EnhancedAuxiliarySafetySystemsGenerator(voyage_plan, 
                        uncertainty_enabled=uncertainty_mode)
                    dataset = generator.generate_dataset()
                elif area == 'life_support_systems':
                    uncertainty_mode = True
                    generator = EnhancedLifeSupportSystemsGenerator(voyage_plan, 
                        uncertainty_enabled=uncertainty_mode)
                    dataset = generator.generate_dataset()
                elif area == 'bridge':
                    uncertainty_mode = True
                    generator = EnhancedBridgeGenerator(voyage_plan,
                        uncertainty_enabled=uncertainty_mode)
                    dataset = generator.generate_dataset()
                elif area == 'cargo':
                    uncertainty_mode = True
                    generator = EnhancedCargoOperationsGenerator(voyage_plan, 
                        uncertainty_enabled=uncertainty_mode)
                    dataset = generator.generate_dataset()
                elif area == 'deck':
                    uncertainty_mode = True
                    generator = EnhancedDeckOperationsGenerator(voyage_plan, 
                        uncertainty_enabled=uncertainty_mode)
                    dataset = generator.generate_dataset()
                elif area == 'engine_room':
                    uncertainty_mode = True
                    
                    generator = EnhancedEngineRoomGenerator(voyage_plan,
                        uncertainty_enabled=uncertainty_mode)
                    dataset = generator.generate_dataset()
                else:
                    print(f"    Warning: Generator for {area} not implemented, skipping...")
                    continue

                print(f"DEBUG: operational_areas = {operational_areas}")


                
                # Add metadata to ensure consistency
                if not dataset.empty:
                    dataset['vessel_id'] = vessel_id
                    dataset['voyage_id'] = voyage_plan.voyage_id
                    dataset['cargo_type'] = cargo_type  # Ensure cargo type is recorded
                    dataset['operational_area'] = area
                    dataset['vessel_profile'] = voyage_plan.vessel_profile.profile_name
                    
                    voyage_datasets[area] = dataset
                    self.generation_stats['total_log_entries'] += len(dataset)
                    
                    # Validate cargo type consistency
                    if 'cargo_type' in dataset.columns:
                        unique_cargo_types = dataset['cargo_type'].unique()
                        if len(unique_cargo_types) > 1:
                            print(f"    Warning: Multiple cargo types in {area} dataset: {unique_cargo_types}")
                        elif unique_cargo_types[0] != cargo_type:
                            print(f"    Warning: Cargo type mismatch in {area}: expected {cargo_type}, got {unique_cargo_types[0]}")
                
            except Exception as e:
                print(f"    Error generating {area} data: {e}")
                continue

            voyage_datasets = apply_inter_regional_cascades(voyage_datasets, voyage_plan.vessel_profile.profile_name)
            # cascade_summary = get_cascade_summary(voyage_datasets)
            # if cascade_summary['total_cascade_records'] > 0:
            #     print(f"      ⚡ {cascade_summary['total_cascade_records']} cascade effects applied")

        return voyage_datasets
    
    def generate_complete_fleet_datasets(self,
                                       fleet_size: int = 9,
                                       voyages_per_vessel: int = 3,
                                       operational_areas: List[str] = None) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Generate complete fleet datasets with controlled cargo distribution
        
        Args:
            fleet_size: Number of vessels in fleet
            voyages_per_vessel: Number of voyages per vessel
            operational_areas: List of operational areas to generate
            
        Returns:
            Nested dictionary: {vessel_id: {voyage_id: {area: DataFrame}}}
        """
        if operational_areas is None:
            operational_areas = ['engine_room', 'bridge', 'cargo','deck', 'auxiliary_support', 'life_support_systems']
        
        start_time = datetime.now()
        
        print(f"\n🚀 Starting complete fleet dataset generation...")
        print(f"Fleet size: {fleet_size} vessels")
        print(f"Voyages per vessel: {voyages_per_vessel}")
        print(f"Operational areas: {', '.join(operational_areas)}")
        
        # Step 1: Generate fleet profiles
        self.generate_fleet_profiles(fleet_size)
        
        # Step 2: Create controlled cargo assignments
        self.create_controlled_cargo_assignments(voyages_per_vessel)
        
        # Step 3: Generate voyages with cargo assignments
        self.generate_fleet_voyages_with_cargo(voyages_per_vessel)
        
        # Step 4: Generate datasets for all voyages
        print(f"\n📊 Generating operational datasets...")
        
        complete_datasets = {}
        
        for vessel_id, voyages in self.fleet_voyages.items():
            print(f"\n🚢 Processing {vessel_id} ({len(voyages)} voyages)...")
            vessel_datasets = {}
            
            for voyage_idx, voyage_plan in enumerate(voyages):
                cargo_type = getattr(voyage_plan, 'cargo_type', 'iron_ore')
                print(f"  Voyage {voyage_idx + 1}/{len(voyages)}: {voyage_plan.voyage_id} ({cargo_type})")
                
                # Generate datasets for this voyage
                voyage_datasets = self.generate_voyage_datasets(
                    vessel_id=vessel_id,
                    voyage_plan=voyage_plan,
                    operational_areas=operational_areas
                )
                
                vessel_datasets[voyage_plan.voyage_id] = voyage_datasets
            
            complete_datasets[vessel_id] = vessel_datasets
        
        self.fleet_datasets = complete_datasets
        self.generation_stats['datasets_created'] = len(complete_datasets)
        self.generation_stats['generation_time'] = (datetime.now() - start_time).total_seconds()
        
        # Validate cargo distribution
        self.validate_cargo_distribution()
        
        return complete_datasets
    
    def validate_cargo_distribution(self):
        """Validate that cargo distribution is correct across all datasets"""
        print(f"\n🔍 Validating cargo distribution across all datasets...")
        
        cargo_counts = defaultdict(int)
        area_cargo_counts = defaultdict(lambda: defaultdict(int))
        deck_hardware_counts = defaultdict(int)
                
        for vessel_id, vessel_data in self.fleet_datasets.items():
            for voyage_id, voyage_data in vessel_data.items():
                for area, dataset in voyage_data.items():
                    if not dataset.empty and 'cargo_type' in dataset.columns:
                        cargo_type = dataset['cargo_type'].iloc[0]  # Should be consistent
                        cargo_counts[cargo_type] += 1
                        area_cargo_counts[area][cargo_type] += 1

                     # ADD DECK-SPECIFIC VALIDATION
                    if area == 'deck' and not dataset.empty and 'vessel_hardware_type' in dataset.columns:
                        hardware_type = dataset['vessel_hardware_type'].iloc[0]
                        deck_hardware_counts[hardware_type] += 1    
        
        print(f"Cargo distribution validation:")
        total_voyages = sum(cargo_counts.values())
        for cargo_type, count in cargo_counts.items():
            percentage = (count / total_voyages) * 100
            expected = self.generation_stats['cargo_distribution'][cargo_type] * len(self.fleet_datasets[list(self.fleet_datasets.keys())[0]])
            print(f"  {cargo_type}: {count} voyages ({percentage:.1f}%) - Expected: {expected}")
        
        # Check consistency across operational areas
        print(f"\nCross-area consistency check:")
        for area, area_counts in area_cargo_counts.items():
            print(f"  {area}: {dict(area_counts)}")

        # ADD DECK HARDWARE VALIDATION
        if deck_hardware_counts:
            print(f"\nDeck hardware distribution:")
            total_deck_voyages = sum(deck_hardware_counts.values())
            for hardware_type, count in deck_hardware_counts.items():
                percentage = (count / total_deck_voyages) * 100
                print(f"  {hardware_type}: {count} voyages ({percentage:.1f}%)")    
    
    def save_fleet_datasets(self, 
                          format: str = 'csv') -> Dict[str, List[str]]:
        """
        Save fleet datasets using your preferred structure:
        - datasets/area/vessel_id/ for individual vessel journeys  
        - fleet_area_complete.csv for combined area data
        
        This matches your existing structure exactly:
        datasets/bridge/AM_01/, datasets/engine_room/AM_01/, etc.
        
        Args:
            format: Output format ('csv', 'parquet', 'json')
            
        Returns:
            Dictionary of saved file paths by area
        """
        if not self.fleet_datasets:
            raise ValueError("No datasets to save. Run generate_complete_fleet_datasets() first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = self.output_dir / f"controlled_fleet_generation_{timestamp}"
        output_subdir.mkdir(exist_ok=True)
        
        # Create datasets directory structure (matching your existing structure)
        datasets_dir = output_subdir / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving fleet datasets to {output_subdir}...")
        print(f"📁 Structure: datasets/[area]/[vessel_id]/ (matching your existing structure)")
        
        saved_files = {}
        area_combined_data = defaultdict(list)
        
        # Get all operational areas
        operational_areas = set()
        for vessel_data in self.fleet_datasets.values():
            for voyage_data in vessel_data.values():
                operational_areas.update(voyage_data.keys())
        
        # Process each operational area separately
        for area in sorted(operational_areas):
            area_dir = datasets_dir / area
            area_dir.mkdir(exist_ok=True)
            area_files = []
            
            print(f"\n📦 Processing {area} datasets...")
            
            # Process each vessel for this area
            for vessel_id in sorted(self.fleet_datasets.keys()):
                vessel_data = self.fleet_datasets[vessel_id]
                vessel_dir = area_dir / vessel_id
                vessel_dir.mkdir(exist_ok=True)
                
                # Collect all voyage data for this vessel in this area
                vessel_area_datasets = []
                for voyage_id, voyage_data in vessel_data.items():
                    if area in voyage_data and not voyage_data[area].empty:
                        dataset = voyage_data[area].copy()
                        vessel_area_datasets.append(dataset)
                        # Also collect for combined dataset
                        area_combined_data[area].append(dataset)
                
                # Save individual vessel file if data exists
                if vessel_area_datasets:
                    # Combine all voyages for this vessel
                    combined_vessel_data = pd.concat(vessel_area_datasets, ignore_index=True)
                    
                    # Sort by timestamp for temporal consistency
                    if 'timestamp' in combined_vessel_data.columns:
                        combined_vessel_data = combined_vessel_data.sort_values('timestamp').reset_index(drop=True)
                    
                    # Generate filename (matches your pattern)
                    vessel_filename = f"{vessel_id}_{area}.{format}"
                    vessel_filepath = vessel_dir / vessel_filename
                    
                    # Save file
                    if format == 'csv':
                        combined_vessel_data.to_csv(vessel_filepath, index=False)
                    elif format == 'parquet':
                        combined_vessel_data.to_parquet(vessel_filepath, index=False)
                    elif format == 'json':
                        combined_vessel_data.to_json(vessel_filepath, orient='records', date_format='iso')
                    
                    area_files.append(str(vessel_filepath))
                    
                    # Show cargo distribution for cargo area
                    if area == 'cargo' and 'cargo_type' in combined_vessel_data.columns:
                        cargo_counts = combined_vessel_data['cargo_type'].value_counts()
                        cargo_summary = ", ".join([f"{cargo}: {count}" for cargo, count in cargo_counts.items()])
                        print(f"  {vessel_id}: {len(combined_vessel_data)} records ({cargo_summary})")
                    else:
                        print(f"  {vessel_id}: {len(combined_vessel_data)} records")
            
            saved_files[area] = area_files
        
        # Create combined files for each area (at root level)
        print(f"\n📋 Creating combined area files...")
        for area in sorted(area_combined_data.keys()):
            area_datasets = area_combined_data[area]
            if area_datasets:
                # Combine all vessels for this area
                combined_dataset = pd.concat(area_datasets, ignore_index=True)
                
                # Sort by timestamp for proper temporal sequence
                if 'timestamp' in combined_dataset.columns:
                    combined_dataset = combined_dataset.sort_values('timestamp').reset_index(drop=True)
                
                # Save combined file
                combined_filename = f"fleet_{area}_complete.{format}"
                combined_filepath = output_subdir / combined_filename
                
                if format == 'csv':
                    combined_dataset.to_csv(combined_filepath, index=False)
                elif format == 'parquet':
                    combined_dataset.to_parquet(combined_filepath, index=False)
                elif format == 'json':
                    combined_dataset.to_json(combined_filepath, orient='records', date_format='iso')
                
                # Show summary with cargo distribution for cargo area
                if area == 'cargo' and 'cargo_type' in combined_dataset.columns:
                    cargo_counts = combined_dataset['cargo_type'].value_counts()
                    total_records = len(combined_dataset)
                    print(f"  {area}: {total_records} total records")
                    for cargo_type, count in cargo_counts.items():
                        percentage = (count / total_records) * 100
                        print(f"    {cargo_type}: {count} records ({percentage:.1f}%)")
                else:
                    print(f"  {area}: {len(combined_dataset)} total records")
        
        # Save generation summary
        self.save_generation_summary(output_subdir)
        
        print(f"\n✅ Fleet datasets saved successfully!")
        print(f"📁 Individual vessels: {output_subdir}/datasets/[area]/[vessel_id]/")
        print(f"📋 Combined files: {output_subdir}/fleet_[area]_complete.{format}")
        print(f"🎯 Cargo distribution controlled at 33/33/33 across fleet")
        
        return saved_files
    
    def save_generation_summary(self, output_dir: Path):
        """Save generation statistics and metadata"""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'generation_stats': self.generation_stats,
            'cargo_assignments': self.cargo_assignments,
            'fleet_composition': {
                vessel_id: {
                    'profile_name': profile.profile_name,
                    'vessel_age': profile.vessel_age,
                    'automation_level': profile.automation_level.value
                }
                for vessel_id, profile in self.fleet_profiles.items()
            },
            'cargo_distribution_summary': {
                'target_distribution': '33.33% each (iron ore, coal, grain)',
                'actual_distribution': self.generation_stats['cargo_distribution'],
                'total_voyages': self.generation_stats['voyages_created']
            }
        }
        
        summary_file = output_dir / 'controlled_generation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create readable summary report
        report_lines = [
            "CONTROLLED FLEET GENERATION SUMMARY",
            "=" * 50,
            f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Generation Time: {self.generation_stats['generation_time']:.1f} seconds",
            "",
            "FLEET COMPOSITION:",
            f"  Vessels: {self.generation_stats['vessels_created']}",
            f"  Total Voyages: {self.generation_stats['voyages_created']}",
            f"  Total Log Entries: {self.generation_stats['total_log_entries']:,}",
            "",
            "CARGO DISTRIBUTION:",
        ]
        
        total = sum(self.generation_stats['cargo_distribution'].values())
        for cargo, count in self.generation_stats['cargo_distribution'].items():
            percentage = (count / total * 100) if total > 0 else 0
            report_lines.append(f"  {cargo}: {count} voyages ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "OPERATIONAL AREAS GENERATED:",
            "  ✅ Engine Room (vessel profile variations)",
            "  ✅ Bridge (navigation patterns)",
            "  ✅ Cargo Operations (cargo-specific monitoring)",
            "  ✅ Deck Operations (weather and sea conditions)",
            "KEY FEATURES:",
            "  ✅ Controlled 33/33/33 cargo distribution",
            "  ✅ Voyage consistency across all operational areas",
            "  ✅ Cargo.py as authoritative source",
            "  ✅ Authentic vessel profile variations",
            "  ✅ Maritime operational realism"
        ])
        
        report_file = output_dir / 'generation_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n📋 Generation Summary:")
        print(f"  Vessels: {self.generation_stats['vessels_created']}")
        print(f"  Voyages: {self.generation_stats['voyages_created']}")
        print(f"  Log Entries: {self.generation_stats['total_log_entries']:,}")
        print(f"  Generation Time: {self.generation_stats['generation_time']:.1f}s")
        print(f"  Summary saved: {summary_file}")
        print(f"  Report saved: {report_file}")



# Example usage and testing
if __name__ == "__main__":
    # Run deck ge

    print("🚢 CONTROLLED FLEET DATASET GENERATION")
    print("=" * 60)
    print("Features:")
    print("✅ Controlled 33/33/33 cargo distribution")
    print("✅ Voyage consistency across operational areas")
    print("✅ Cargo.py as authoritative source")
    print("✅ Authentic vessel profile variations")
    print("✅ Weather-responsive deck operations")  # ADD THIS LINE
    print("✅ Geared vs gearless vessel distinction")  # ADD THIS LINE

    
    # Configuration
    config = {
        'fleet_size': 9,                              # 9 vessels total
        'voyages_per_vessel': 3,                      # 5 voyages each = 45 total voyages
        'operational_areas': ['engine_room', 'bridge', 'cargo','deck', 'auxiliary_support', 'life_support_systems'],  # All four areas
        'output_format': 'csv'                        # CSV for easy analysis
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create controlled fleet generator
    fleet_generator = ControlledFleetGenerator()
    
    # Generate complete fleet datasets
    print(f"\n🚀 Starting controlled fleet generation...")
    
    complete_datasets = fleet_generator.generate_complete_fleet_datasets(
        fleet_size=config['fleet_size'],
        voyages_per_vessel=config['voyages_per_vessel'],
        operational_areas=config['operational_areas']
    )
    print(f"DEBUG: fleet_datasets keys: {list(fleet_generator.fleet_datasets.keys())}")
    print(f"DEBUG: Sample voyage areas: {list(list(fleet_generator.fleet_datasets.values())[0].values())[0].keys()}")
    # Save all datasets using your preferred structure
    saved_files = fleet_generator.save_fleet_datasets(
        format=config['output_format']
    )
    
    print(f"\n🎯 Generation Complete!")
    print(f"Generated datasets for {len(complete_datasets)} vessels")
    print(f"Files saved to: {fleet_generator.output_dir}")
    
    # Display sample data from each operational area
    print(f"\n📊 Sample Data Preview:")
    sample_vessel = list(complete_datasets.keys())[0]
    sample_voyage = list(complete_datasets[sample_vessel].keys())[0]
    
     
    for area, dataset in complete_datasets[sample_vessel][sample_voyage].items():
        if not dataset.empty:
            cargo_type = dataset['cargo_type'].iloc[0] if 'cargo_type' in dataset.columns else 'unknown'
            
            # ADD DECK-SPECIFIC PREVIEW INFO
            if area == 'deck' and 'vessel_hardware_type' in dataset.columns:
                hardware_type = dataset['vessel_hardware_type'].iloc[0]
                operations_halted = dataset['operations_formally_halted'].mean() * 100
                print(f"\n{area.upper()} ({hardware_type}, {operations_halted:.1f}% operations halted):")
            else:
                print(f"\n{area.upper()} ({cargo_type}):")
            
            print(f"  Records: {len(dataset)}")
            if 'timestamp' in dataset.columns:
                print(f"  Time span: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")
            
            # Show deck-specific sample data
            if area == 'deck':
                deck_sample = {
                    'cranes_operational': dataset['deck_cranes_operational_count'].iloc[0],
                    'personnel_on_deck': dataset['total_personnel_on_deck'].iloc[0],
                    'deck_wetness': dataset['deck_wetness_level'].iloc[0]
                }
                print(f"  Sample: {deck_sample}")
            else:
                # Existing sample logic for other areas
                display_cols = [col for col in dataset.columns[:5] if col != 'timestamp']
                if display_cols and len(dataset) > 0:
                    print(f"  Sample: {dataset[display_cols].iloc[0].to_dict()}")