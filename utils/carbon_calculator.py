import pandas as pd
import numpy as np

class CarbonCalculator:
    """Calculate carbon emissions for various supply chain activities"""
    
    # Emission factors (kg CO2 per unit)
    EMISSION_FACTORS = {
        'transportation': {
            'diesel_truck': 2.68,  # kg CO2 per liter of diesel
            'electric_truck': 0.05,  # kg CO2 per km (grid average)
            'rail': 0.022,  # kg CO2 per ton-km
            'ship': 0.010,  # kg CO2 per ton-km
            'air': 0.805,  # kg CO2 per ton-km
        },
        'energy': {
            'electricity_grid': 0.5,  # kg CO2 per kWh
            'natural_gas': 1.89,  # kg CO2 per kWh
            'solar': 0.04,  # kg CO2 per kWh
            'wind': 0.011,  # kg CO2 per kWh
        },
        'manufacturing': {
            'average': 2.5,  # kg CO2 per unit
            'energy_intensive': 5.2,  # kg CO2 per unit
            'electronics': 3.8,  # kg CO2 per unit
            'textiles': 4.2,  # kg CO2 per unit
            'food': 1.8,  # kg CO2 per unit
        },
        'warehousing': {
            'per_square_meter': 0.15,  # kg CO2 per sqm per day
            'per_pallet': 0.02,  # kg CO2 per pallet per day
            'lighting': 0.001,  # kg CO2 per kWh for lighting
            'cooling': 0.002,  # kg CO2 per kWh for cooling
        }
    }
    
    @staticmethod
    def calculate_transport_emissions(distance_km, fuel_type='diesel_truck', fuel_consumption_l_per_km=0.3, load_factor=0.8):
        """Calculate transportation emissions"""
        if fuel_type == 'electric_truck':
            emissions = distance_km * CarbonCalculator.EMISSION_FACTORS['transportation'][fuel_type]
        else:
            total_fuel = distance_km * fuel_consumption_l_per_km / load_factor
            emissions = total_fuel * CarbonCalculator.EMISSION_FACTORS['transportation'][fuel_type]
        return emissions
    
    @staticmethod
    def calculate_warehouse_emissions(energy_consumption_kwh, energy_source='electricity_grid', square_meters=0, operational_days=30):
        """Calculate warehouse emissions"""
        # Energy-based emissions
        energy_emissions = energy_consumption_kwh * CarbonCalculator.EMISSION_FACTORS['energy'][energy_source]
        
        # Space-based emissions
        space_emissions = square_meters * operational_days * CarbonCalculator.EMISSION_FACTORS['warehousing']['per_square_meter']
        
        return energy_emissions + space_emissions
    
    @staticmethod
    def calculate_inventory_emissions(inventory_units, product_type='average', storage_days=30):
        """Calculate emissions from inventory holding"""
        manufacturing_emissions = inventory_units * CarbonCalculator.EMISSION_FACTORS['manufacturing'][product_type]
        
        # Storage emissions (simplified)
        storage_emissions = inventory_units * storage_days * 0.001
        
        return manufacturing_emissions + storage_emissions
    
    @staticmethod
    def calculate_supplier_emissions(carbon_footprint_score, order_volume, base_emission_factor=100):
        """Calculate emissions based on supplier sustainability score"""
        # Convert score to emission factor (lower score = better = lower emissions)
        emission_factor = (carbon_footprint_score / 100) * base_emission_factor
        return order_volume * emission_factor
    
    @staticmethod
    def calculate_total_savings(optimized_scenario, baseline_scenario):
        """Calculate total carbon savings between optimized and baseline scenarios"""
        savings = {}
        
        for category in ['transport', 'warehouse', 'inventory', 'supplier']:
            if category in baseline_scenario and category in optimized_scenario:
                savings[category] = baseline_scenario[category] - optimized_scenario[category]
        
        savings['total'] = sum(savings.values())
        savings['percentage'] = (savings['total'] / sum(baseline_scenario.values())) * 100 if sum(baseline_scenario.values()) > 0 else 0
        
        return savings
    
    @staticmethod
    def generate_emissions_report(baseline_data, optimized_data):
        """Generate a comprehensive emissions report"""
        report = {
            'baseline_emissions': baseline_data,
            'optimized_emissions': optimized_data,
            'savings': CarbonCalculator.calculate_total_savings(optimized_data, baseline_data),
            'environmental_impact': {}
        }
        
        # Calculate environmental impact equivalents
        total_savings = report['savings']['total']
        report['environmental_impact'] = {
            'trees_planted': total_savings / 21.77,  # kg CO2 absorbed by one tree per year
            'cars_off_road': total_savings / 4600,   # kg CO2 from average car per year
            'homes_energy': total_savings / 12200,   # kg CO2 from average home per year
        }
        
        return report