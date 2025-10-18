import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_sample_demand_data():
    """Generate sample demand data for demonstration"""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
    base_demand = [100, 120, 130, 110, 150, 140, 160, 170, 180, 160, 140, 120, 130]
    actual_demand = [d * np.random.uniform(0.8, 1.2) for d in base_demand[:len(dates)]]
    forecast_demand = [d * np.random.uniform(0.9, 1.1) for d in actual_demand]
    
    df = pd.DataFrame({
        'date': dates,
        'actual_demand': actual_demand,
        'forecast_demand': forecast_demand,
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Furniture'], len(dates))
    })
    return df

def load_csv_data(file_name):
    """Load CSV data from the data directory"""
    data_path = os.path.join('data', file_name)
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        # If file doesn't exist, generate sample data
        return generate_sample_data(file_name)

def generate_sample_data(file_name):
    """Generate sample data if CSV files are not available"""
    if 'demand' in file_name.lower():
        return generate_demand_data()
    elif 'supplier' in file_name.lower():
        return generate_supplier_data()
    elif 'warehouse' in file_name.lower():
        return generate_warehouse_data()
    elif 'transport' in file_name.lower():
        return generate_transportation_data()
    else:
        return pd.DataFrame()

def generate_demand_data():
    """Generate sample demand data"""
    dates = pd.date_range(start='2023-01-01', end='2024-06-01', freq='M')
    products = ['Electronics', 'Clothing', 'Food', 'Furniture']
    
    data = []
    for date in dates:
        for product in products:
            base_demand = np.random.randint(500, 2000)
            actual_demand = base_demand * np.random.uniform(0.8, 1.2)
            forecast_demand = actual_demand * np.random.uniform(0.9, 1.1)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'product_category': product,
                'actual_demand': int(actual_demand),
                'forecast_demand': int(forecast_demand),
                'price': round(np.random.uniform(50, 500), 2),
                'promotion_flag': np.random.choice([0, 1], p=[0.7, 0.3]),
                'seasonality': round(np.random.uniform(0.7, 1.3), 2)
            })
    
    return pd.DataFrame(data)

def generate_supplier_data():
    """Generate sample supplier data"""
    locations = ['Germany', 'USA', 'Canada', 'Sweden', 'Netherlands', 'Denmark', 'Finland', 'France', 'Norway', 'UK']
    certifications = ['ISO 14001', 'ISO 50001', 'B Corp', 'None']
    
    suppliers = []
    for i in range(1, 16):
        suppliers.append({
            'supplier_id': f'SUP_{i:03d}',
            'name': f'Supplier {i}',
            'location': np.random.choice(locations),
            'carbon_footprint': np.random.randint(60, 100),
            'energy_efficiency': np.random.randint(75, 98),
            'renewable_energy': np.random.randint(50, 95),
            'cost_score': np.random.randint(75, 95),
            'delivery_reliability': np.random.randint(90, 100),
            'certifications': np.random.choice(certifications, p=[0.4, 0.2, 0.2, 0.2]),
            'lead_time_days': np.random.randint(14, 28),
            'minimum_order': np.random.randint(500, 2000)
        })
    
    return pd.DataFrame(suppliers)

def generate_warehouse_data():
    """Generate sample warehouse data"""
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    energy_sources = ['Grid', 'Mixed', 'Solar']
    
    warehouses = []
    for i in range(1, 11):
        warehouses.append({
            'warehouse_id': f'WH_{i:03d}',
            'location': locations[i-1],
            'energy_consumption_kwh': np.random.randint(8000, 15000),
            'square_meters': np.random.randint(4000, 6000),
            'storage_capacity': np.random.randint(8000, 12000),
            'inventory_turnover': round(np.random.uniform(7.0, 10.5), 1),
            'energy_source': np.random.choice(energy_sources, p=[0.5, 0.3, 0.2]),
            'solar_panels': np.random.choice(['Yes', 'No'], p=[0.6, 0.4]),
            'avg_temperature': np.random.randint(16, 26),
            'operational_hours': np.random.choice([12, 16, 24], p=[0.3, 0.4, 0.3])
        })
    
    return pd.DataFrame(warehouses)

def generate_transportation_data():
    """Generate sample transportation route data"""
    routes = []
    route_pairs = [
        ('New York', 'Los Angeles', 4500),
        ('Chicago', 'Houston', 1750),
        ('San Francisco', 'Seattle', 1300),
        ('Miami', 'Atlanta', 1050),
        ('Denver', 'Dallas', 1100),
        ('Boston', 'Washington', 650),
        ('New York', 'Chicago', 1270),
        ('Los Angeles', 'Phoenix', 600),
        ('Seattle', 'Portland', 280),
        ('Atlanta', 'Nashville', 400)
    ]
    
    for i, (origin, destination, distance) in enumerate(route_pairs, 1):
        baseline_fuel = distance * 0.3
        optimized_fuel = baseline_fuel * np.random.uniform(0.75, 0.90)
        emissions = optimized_fuel * 2.5
        
        routes.append({
            'route_id': f'R_{i:03d}',
            'origin': origin,
            'destination': destination,
            'distance_km': distance,
            'baseline_fuel_l': int(baseline_fuel),
            'optimized_fuel_l': int(optimized_fuel),
            'emissions_kg_co2': int(emissions),
            'transport_mode': 'Truck',
            'typical_duration_hours': max(4, int(distance / 80)),
            'cargo_capacity_kg': 10000
        })
    
    return pd.DataFrame(routes)