import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import with error handling
try:
    from models.route_optimizer import GreenRouteOptimizer
except ImportError as e:
    st.error(f"Error importing route optimizer: {e}")
    # Create a fallback class
    class GreenRouteOptimizer:
        def optimize_route(self, locations, vehicle_type):
            return self.get_fallback_route(locations)
        
        def get_fallback_route(self, locations):
            return {
                'route': list(range(len(locations))),
                'total_distance': len(locations) * 150,
                'total_emissions': len(locations) * 31.5
            }
        
        def calculate_savings(self, optimized_route, baseline_route):
            return {
                'distance_savings_km': 50,
                'emission_savings_kg': 10.5,
                'savings_percentage': 15.0,
                'fuel_savings_liters': 15.0
            }

try:
    from utils.carbon_calculator import CarbonCalculator
except ImportError:
    # Fallback carbon calculator
    class CarbonCalculator:
        def calculate_transport_emissions(self, distance, vehicle_type, fuel_efficiency=0.3, load_factor=1.0):
            emission_factors = {
                'diesel_truck': 2.68,
                'electric_truck': 0.05,
                'hybrid_truck': 1.34
            }
            fuel_consumption = distance * fuel_efficiency * load_factor
            return fuel_consumption * emission_factors.get(vehicle_type, 2.68)

try:
    from utils.data_loader import load_csv_data
except ImportError:
    # Fallback data loader
    def load_csv_data(filename):
        # Generate sample transportation data
        data = {
            'route_id': [1, 2, 3, 4, 5],
            'from_location': ['New York', 'Chicago', 'Los Angeles', 'Houston', 'Phoenix'],
            'to_location': ['Chicago', 'Los Angeles', 'Houston', 'Phoenix', 'New York'],
            'distance_km': [1260, 2800, 2200, 1800, 3400],
            'vehicle_type': ['diesel_truck', 'diesel_truck', 'electric_truck', 'hybrid_truck', 'diesel_truck'],
            'emissions_kg': [340, 750, 110, 360, 910]
        }
        return pd.DataFrame(data)

def generate_distance_matrix(locations):
    """Generate a realistic distance matrix between locations"""
    # Realistic distances between major US cities (in km)
    city_distances = {
        'New York': {'New York': 0, 'Los Angeles': 3940, 'Chicago': 1140, 'Houston': 2280, 'Phoenix': 3450,
                    'Philadelphia': 150, 'San Antonio': 2640, 'San Diego': 3900, 'Dallas': 2100, 'San Jose': 4150},
        'Los Angeles': {'New York': 3940, 'Los Angeles': 0, 'Chicago': 2800, 'Houston': 2200, 'Phoenix': 590,
                       'Philadelphia': 3900, 'San Antonio': 2000, 'San Diego': 190, 'Dallas': 2000, 'San Jose': 540},
        'Chicago': {'New York': 1140, 'Los Angeles': 2800, 'Chicago': 0, 'Houston': 1500, 'Phoenix': 2300,
                   'Philadelphia': 1100, 'San Antonio': 1700, 'San Diego': 2800, 'Dallas': 1300, 'San Jose': 3000},
        'Houston': {'New York': 2280, 'Los Angeles': 2200, 'Chicago': 1500, 'Houston': 0, 'Phoenix': 1600,
                   'Philadelphia': 2200, 'San Antonio': 320, 'San Diego': 2100, 'Dallas': 360, 'San Jose': 2300},
        'Phoenix': {'New York': 3450, 'Los Angeles': 590, 'Chicago': 2300, 'Houston': 1600, 'Phoenix': 0,
                   'Philadelphia': 3400, 'San Antonio': 1400, 'San Diego': 560, 'Dallas': 1500, 'San Jose': 1100}
    }
    
    # Default distance for missing cities
    default_distance = 500
    
    n = len(locations)
    distance_matrix = []
    
    for i, loc1 in enumerate(locations):
        row = []
        for j, loc2 in enumerate(locations):
            if i == j:
                row.append(0)
            else:
                # Try to get actual distance, otherwise use default
                dist = city_distances.get(loc1, {}).get(loc2, default_distance)
                row.append(dist)
        distance_matrix.append(row)
    
    return distance_matrix

def main():
    st.title("🚚 Green Route Optimizer")
    
    st.markdown("""
    Optimize transportation routes to minimize fuel consumption and carbon emissions 
    while maintaining delivery efficiency.
    """)
    
    # Initialize optimizer
    optimizer = GreenRouteOptimizer()
    carbon_calc = CarbonCalculator()
    
    # Route input section
    st.subheader("📍 Route Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Predefined locations for demo
        predefined_locations = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
        ]
        
        selected_locations = st.multiselect(
            "Select Delivery Locations",
            predefined_locations,
            default=["New York", "Chicago", "Los Angeles", "Houston"]
        )
        
        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["diesel_truck", "electric_truck", "hybrid_truck"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        cargo_weight = st.slider("Cargo Weight (kg)", 100, 5000, 1000)
        load_factor = st.slider("Load Factor (%)", 50, 100, 80) / 100
        optimization_priority = st.selectbox(
            "Optimization Priority",
            ["Minimize Emissions", "Balance Emissions/Time", "Minimize Time"]
        )
    
    if len(selected_locations) < 2:
        st.warning("Please select at least 2 locations for route optimization.")
        return
    
    # Calculate baseline and optimized routes
    if st.button("🚀 Optimize Route", type="primary"):
        with st.spinner("Calculating optimal route..."):
            try:
                # Generate distance matrix
                distance_matrix = generate_distance_matrix(selected_locations)
                
                # Optimize route
                optimized_route = optimizer.optimize_route(selected_locations, distance_matrix)
                
                if optimized_route:
                    # Convert route indices to location names
                    optimized_locations = [selected_locations[i] for i in optimized_route['route']]
                    
                    # Calculate baseline route (simple order)
                    baseline_distance = 0
                    for i in range(len(selected_locations) - 1):
                        from_loc = selected_locations[i]
                        to_loc = selected_locations[i + 1]
                        baseline_distance += distance_matrix[i][i + 1]
                    
                    baseline_emissions = carbon_calc.calculate_transport_emissions(
                        baseline_distance, vehicle_type, 0.3, load_factor
                    )
                    
                    baseline_route = {
                        'total_distance': baseline_distance,
                        'total_emissions': baseline_emissions,
                        'route': selected_locations  # Original order
                    }
                    
                    # Calculate savings
                    savings = optimizer.calculate_savings(optimized_route, baseline_route)
                    
                    # Display results
                    st.subheader("📊 Optimization Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Distance Saved", f"{savings['distance_savings_km']:.0f} km")
                    with col2:
                        st.metric("Carbon Saved", f"{savings['emission_savings_kg']:.0f} kg CO₂")
                    with col3:
                        st.metric("Fuel Saved", f"{savings['fuel_savings_liters']:.0f} liters")
                    with col4:
                        st.metric("Efficiency Gain", f"{savings['savings_percentage']:.1f}%")
                    
                    # Route comparison
                    st.subheader("🛣️ Route Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info("**Baseline Route**")
                        st.write(" → ".join(baseline_route['route']))
                        st.metric("Total Distance", f"{baseline_route['total_distance']:.0f} km")
                        st.metric("Estimated Emissions", f"{baseline_route['total_emissions']:.0f} kg CO₂")
                    
                    with col2:
                        st.success("**Optimized Route**")
                        st.write(" → ".join(optimized_locations))
                        st.metric("Total Distance", f"{optimized_route['total_distance']:.0f} km")
                        st.metric("Estimated Emissions", f"{optimized_route['total_emissions']:.0f} kg CO₂")
                    
                    # Visualization
                    st.subheader("📈 Route Visualization")
                    
                    # Create route progression chart
                    fig = go.Figure()
                    
                    # Baseline route progression
                    baseline_distances = [0]
                    current_distance = 0
                    for i in range(len(baseline_route['route']) - 1):
                        from_idx = selected_locations.index(baseline_route['route'][i])
                        to_idx = selected_locations.index(baseline_route['route'][i + 1])
                        segment_distance = distance_matrix[from_idx][to_idx]
                        current_distance += segment_distance
                        baseline_distances.append(current_distance)
                    
                    # Optimized route progression
                    optimized_distances = [0]
                    current_distance = 0
                    for i in range(len(optimized_route['route']) - 1):
                        from_idx = optimized_route['route'][i]
                        to_idx = optimized_route['route'][i + 1]
                        segment_distance = distance_matrix[from_idx][to_idx]
                        current_distance += segment_distance
                        optimized_distances.append(current_distance)
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(baseline_route['route']))),
                        y=baseline_distances,
                        mode='lines+markers',
                        name='Baseline Route',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(optimized_locations))),
                        y=optimized_distances,
                        mode='lines+markers',
                        name='Optimized Route',
                        line=dict(color='green', width=3)
                    ))
                    
                    fig.update_layout(
                        title='Route Distance Progression',
                        xaxis_title='Stop Number',
                        yaxis_title='Cumulative Distance (km)',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Environmental impact
                    st.subheader("🌍 Environmental Impact")
                    
                    total_savings = savings['emission_savings_kg']
                    environmental_impact = {
                        'trees_planted': total_savings / 21.77,
                        'cars_off_road': total_savings / 4600,
                        'homes_energy': total_savings / 12200
                    }
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Equivalent Trees Planted",
                            f"{environmental_impact['trees_planted']:.1f}",
                            help="Number of trees needed to absorb equivalent CO2 annually"
                        )
                    
                    with col2:
                        st.metric(
                            "Cars Off the Road",
                            f"{environmental_impact['cars_off_road']:.2f}",
                            help="Equivalent to taking this many cars off the road for a year"
                        )
                    
                    with col3:
                        st.metric(
                            "Home Energy Savings",
                            f"{environmental_impact['homes_energy']:.2f}",
                            help="Equivalent to annual energy use of this many homes"
                        )
                
                else:
                    st.error("Route optimization failed. Please try with different locations.")
                    
            except Exception as e:
                st.error(f"Error during route optimization: {str(e)}")
                st.info("Please try with different locations or check the console for details.")
    
    # Sample data section
    st.markdown("---")
    st.subheader("📋 Sample Transportation Data")
    
    transport_df = load_csv_data('transportation_routes.csv')
    st.dataframe(transport_df.head(8))
    
    # Download option
    csv = transport_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Transportation Data",
        data=csv,
        file_name="transportation_routes.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()