# 🚢 Maritime Fleet Operations Dataset Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maritime AI](https://img.shields.io/badge/Maritime-AI-blue.svg)](https://github.com/yourusername/maritime-fleet-generator)

> **Synthetic bulk carrier operational data generator for maritime AI/ML research and development**

A comprehensive system for generating realistic synthetic maritime vessel operational data across six critical ship operational areas. Addresses the industry's critical shortage of labeled operational data for machine learning model training while ensuring privacy and regulatory compliance.

## 🎯 **Project Overview**

The maritime industry faces significant challenges in AI/ML development due to:
- Limited access to operational data (commercial sensitivity)
- Privacy and regulatory constraints
- Lack of standardized datasets for research

This project generates **27,259 realistic synthetic records** across six operational areas of bulk carrier vessels, incorporating authentic maritime patterns, vessel variations, and regulatory compliance factors.

## 📊 **Dataset Statistics**

| Operational Area | Records Generated | Key Features |
|------------------|-------------------|--------------|
| **Engine Room** | 10,893 | Main engine performance, fuel consumption, maintenance |
| **Bridge Navigation** | 6,244 | GPS coordinates, weather, autopilot, watch schedules |
| **Cargo Operations** | 4,896 | Loading/unloading, ballast management, cargo-specific patterns |
| **Deck Operations** | 601 | Weather-responsive ops, crane management, safety protocols |
| **Auxiliary Systems** | 1,118 | Ballast pumps, safety systems, SOLAS compliance |
| **Life Support** | 3,507 | HVAC, galley operations, crew quarters monitoring |
| **TOTAL** | **27,259** | **Complete vessel operational profile** |

### **Cargo Distribution Analysis**
- **Grain**: 1,890 records (38.6%)
- **Iron Ore**: 1,548 records (31.6%) 
- **Coal**: 1,458 records (29.8%)

## 🏗️ **System Architecture**

### Core Components

1. **Fleet Generator** (`generate_fleet.py`)
   - Controlled cargo distribution (33/33/33: iron ore, coal, grain)
   - 9 vessels × 3 voyages = 27 complete operational cycles
   - Cross-operational consistency enforcement

2. **Vessel Profiles** (`core/vessel_profile.py`)
   - **Modern Major**: Age 5, high automation, stable crew
   - **Aging Midtier**: Age 12, medium automation, rotating crew  
   - **Legacy Small**: Age 18, low automation, mixed experience crew

3. **Operational Area Generators** (`generators/`)
   - Physics-based correlations (0.7-0.9 correlation coefficients)
   - Maritime watch schedules and crew timing simulation
   - Weather-responsive operational patterns
   - Cargo-specific monitoring variations

## 🚢 **Vessel Operational Areas**

### **1. Engine Room Operations**
- Main engine performance monitoring
- Fuel consumption patterns (physics-based correlations)
- Temperature and load monitoring
- Maintenance scheduling and watch protocols

### **2. Bridge Navigation** 
- Continuous GPS position tracking
- Speed and course over ground
- Weather condition monitoring
- Autopilot engagement patterns
- Traffic density awareness

### **3. Cargo Operations**
- Cargo-specific loading patterns (iron ore, coal, grain)
- Ballast water management coordination
- Phase-driven logging (loading/transit/unloading)
- Cargo monitoring frequency variations

### **4. Deck Operations**
- Weather-responsive deck operations
- Crane equipment management (geared vs gearless)
- Personnel safety monitoring
- Sea condition adaptations

### **5. Auxiliary & Safety Systems**
- Ballast pump operations
- Fire detection and suppression
- Bilge monitoring systems
- SOLAS compliance tracking

### **6. Life Support Systems**
- HVAC operations and efficiency
- Galley equipment monitoring  
- Crew quarters environmental control
- Waste management systems

## 🔧 **Key Technical Features**

### **Maritime Authenticity**
- **Vessel Profile Integration**: Three distinct vessel categories with realistic operational variations
- **Physics-Based Correlations**: Engine load-temperature correlations (0.7-0.9) vary by vessel condition
- **Crew Timing Simulation**: Authentic maritime watch schedules (4-hour watches, handover protocols)
- **Weather Dependencies**: Operational adjustments based on sea conditions

### **Data Quality Assurance**
- **Voyage Consistency**: Same cargo type maintained across all operational areas
- **Temporal Realism**: Human timing variations, maintenance schedules
- **Equipment Variations**: Age-based performance degradation
- **Regulatory Compliance**: SOLAS standards integration

### **Scalability & Control**
- **Controlled Distributions**: Precise cargo type allocation (33/33/33)
- **Configurable Parameters**: Fleet size, voyage duration, operational intensity
- **Modular Design**: Each operational area independently configurable
- **Export Formats**: CSV, JSON, Parquet support

## 🚀 **Quick Start**

### Installation
```bash
git clone https://github.com/yourusername/maritime-fleet-generator.git
cd maritime-fleet-generator
pip install -r requirements.txt
```

### Basic Usage
```python
from generate_fleet import ControlledFleetGenerator

# Initialize generator
fleet_gen = ControlledFleetGenerator()

# Generate complete fleet datasets
datasets = fleet_gen.generate_complete_fleet_datasets(
    fleet_size=9,
    voyages_per_vessel=3,
    operational_areas=['engine_room', 'bridge', 'cargo', 'deck', 'auxiliary_support', 'life_support_systems']
)

# Save datasets
fleet_gen.save_fleet_datasets(format='csv')
```

### Output Structure
```
datasets/
├── fleet_engine_room_complete.csv      # Combined engine data
├── fleet_bridge_complete.csv           # Combined navigation data  
├── fleet_cargo_complete.csv            # Combined cargo operations
├── fleet_deck_complete.csv             # Combined deck operations
├── fleet_auxiliary_support_complete.csv # Combined auxiliary systems
├── fleet_life_support_systems_complete.csv # Combined life support
└── generation_summary.json             # Dataset statistics
```

## 📈 **Sample Data Structure**

### Engine Room Data
```csv
timestamp,vessel_id,voyage_id,cargo_type,main_engine_power_kw,main_engine_rpm,fuel_consumption_mt_per_day,operating_temperature_c,vessel_profile
2024-01-15 08:00:00,VESSEL_001,VOYAGE_001,iron_ore,12450.0,85.2,42.3,78.5,Modern Major
```

### Bridge Navigation Data  
```csv
timestamp,vessel_id,voyage_id,latitude,longitude,speed_over_ground,course_over_ground,weather_condition,autopilot_engaged
2024-01-15 08:00:00,VESSEL_001,VOYAGE_001,25.2756,55.2962,14.2,095,moderate_sea,true
```

## 🔬 **Applications & Use Cases**

### **Machine Learning Training**
- Predictive maintenance model development
- Operational risk assessment algorithms
- Performance optimization systems
- Anomaly detection in maritime operations

### **Research & Development**
- Maritime automation system testing
- Digital twin development
- Operational efficiency research
- Safety system validation

### **Industry Applications**
- Fleet management system development
- Insurance risk modeling
- Regulatory compliance training
- Maritime digitization initiatives

## 📊 **Data Validation & Quality**

### **Statistical Validation**
- Physics-based correlation verification
- Temporal pattern consistency checks
- Cross-operational area data alignment
- Vessel profile impact analysis

### **Maritime Domain Validation**
- Industry expert review protocols
- Regulatory compliance verification
- Operational realism assessment
- Equipment specification accuracy

## 🤝 **Contributing**

We welcome contributions from the maritime and data science communities!

### Areas for Enhancement
- Additional vessel types (tankers, container ships)
- Enhanced weather modeling
- Advanced failure simulation
- Real-time streaming capabilities

### How to Contribute
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 **Requirements**

```
pandas>=1.3.0
numpy>=1.21.0
datetime
pathlib
json
typing
```

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Maritime industry professionals who provided domain expertise
- Open source data science community
- Bulk carrier operational specification contributors

## 📞 **Contact & Support**

- **Author**: [Your Name]
- **LinkedIn**: [Your LinkedIn Profile]
- **Email**: [your.email@example.com]
- **Issues**: Please use GitHub Issues for bug reports and feature requests

## 🔗 **Related Projects**

- [Maritime AI Research Hub](https://example.com) - Community resource
- [Shipping Data Standards](https://example.com) - Industry standards
- [Maritime ML Benchmarks](https://example.com) - Performance baselines

---

**⭐ If this project helps your maritime AI research, please consider giving it a star!**

---

*This project is part of ongoing research in maritime artificial intelligence and operational risk modeling. Generated data is synthetic and designed for research and development purposes.*
