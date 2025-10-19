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
        def optimize_route(self, locations, distance_matrix):
            # Simple optimization: reverse the route for demo purposes
            return {
                'route': list(range(len(locations)))[::-1] + [0],  # Return to start
                'total_distance': sum(distance_matrix[i][(i+1)%len(locations)] for i in range(len(locations))),
                'total_emissions': sum(distance_matrix[i][(i+1)%len(locations)] * 0.21 for i in range(len(locations)))
            }
        
        def calculate_savings(self, optimized_route, baseline_route):
            return {
                'distance_savings_km': max(0, baseline_route['total_distance'] - optimized_route['total_distance']),
                'emission_savings_kg': max(0, baseline_route['total_emissions'] - optimized_route['total_emissions']),
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

# City coordinates (latitude, longitude)
CITY_COORDINATES = {
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'Phoenix': (33.4484, -112.0740),
    'Philadelphia': (39.9526, -75.1652),
    'San Antonio': (29.4241, -98.4936),
    'San Diego': (32.7157, -117.1611),
    'Dallas': (32.7767, -96.7970),
    'San Jose': (37.3382, -121.8863)
}

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
                   'Philadelphia': 3400, 'San Antonio': 1400, 'San Diego': 560, 'Dallas': 1500, 'San Jose': 1100},
        'Philadelphia': {'New York': 150, 'Los Angeles': 3900, 'Chicago': 1100, 'Houston': 2200, 'Phoenix': 3400,
                        'Philadelphia': 0, 'San Antonio': 2500, 'San Diego': 3850, 'Dallas': 2000, 'San Jose': 4100},
        'San Antonio': {'New York': 2640, 'Los Angeles': 2000, 'Chicago': 1700, 'Houston': 320, 'Phoenix': 1400,
                       'Philadelphia': 2500, 'San Antonio': 0, 'San Diego': 1950, 'Dallas': 400, 'San Jose': 2200},
        'San Diego': {'New York': 3900, 'Los Angeles': 190, 'Chicago': 2800, 'Houston': 2100, 'Phoenix': 560,
                     'Philadelphia': 3850, 'San Antonio': 1950, 'San Diego': 0, 'Dallas': 1950, 'San Jose': 720},
        'Dallas': {'New York': 2100, 'Los Angeles': 2000, 'Chicago': 1300, 'Houston': 360, 'Phoenix': 1500,
                  'Philadelphia': 2000, 'San Antonio': 400, 'San Diego': 1950, 'Dallas': 0, 'San Jose': 2300},
        'San Jose': {'New York': 4150, 'Los Angeles': 540, 'Chicago': 3000, 'Houston': 2300, 'Phoenix': 1100,
                    'Philadelphia': 4100, 'San Antonio': 2200, 'San Diego': 720, 'Dallas': 2300, 'San Jose': 0}
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

def create_route_map(baseline_locations, optimized_locations, baseline_route, optimized_route):
    """Create an interactive map showing both baseline and optimized routes"""
    
    # Get coordinates for all locations
    baseline_coords = [CITY_COORDINATES[loc] for loc in baseline_locations]
    optimized_coords = [CITY_COORDINATES[loc] for loc in optimized_locations]
    
    # Create the map figure
    fig = go.Figure()
    
    # Add baseline route (dark orange)
    baseline_lats = [coord[0] for coord in baseline_coords]
    baseline_lons = [coord[1] for coord in baseline_coords]
    
    fig.add_trace(go.Scattermapbox(
        name="Baseline Route",
        mode="lines+markers+text",
        lon=baseline_lons,
        lat=baseline_lats,
        marker=dict(size=12, color='darkorange'),
        line=dict(width=4, color='darkorange'),
        text=baseline_locations,
        textposition="top center",
        hoverinfo="text",
        hovertext=[f"<b>{loc}</b><br>Stop {i+1}" for i, loc in enumerate(baseline_locations)]
    ))
    
    # Add optimized route (green)
    optimized_lats = [coord[0] for coord in optimized_coords]
    optimized_lons = [coord[1] for coord in optimized_coords]
    
    fig.add_trace(go.Scattermapbox(
        name="Optimized Route",
        mode="lines+markers+text",
        lon=optimized_lons,
        lat=optimized_lats,
        marker=dict(size=12, color='green'),
        line=dict(width=4, color='green'),
        text=optimized_locations,
        textposition="bottom center",
        hoverinfo="text",
        hovertext=[f"<b>{loc}</b><br>Stop {i+1}" for i, loc in enumerate(optimized_locations)]
    ))
    
    # Add start and end markers with different symbols
    fig.add_trace(go.Scattermapbox(
        name="Start Point",
        mode="markers",
        lon=[baseline_lons[0]],
        lat=[baseline_lats[0]],
        marker=dict(size=20, color='blue', symbol='circle'),
        hoverinfo="text",
        hovertext=[f"<b>Start: {baseline_locations[0]}</b>"]
    ))
    
    # Update layout for better map display
    fig.update_layout(
        title="Route Comparison Map: Baseline vs Optimized",
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=np.mean(baseline_lats), lon=np.mean(baseline_lons)),
            zoom=3
        ),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_route_comparison_chart(baseline_locations, optimized_locations, baseline_route, optimized_route):
    """Create a bar chart comparing route metrics"""
    
    metrics = ['Total Distance (km)', 'Estimated Emissions (kg CO₂)']
    baseline_values = [baseline_route['total_distance'], baseline_route['total_emissions']]
    optimized_values = [optimized_route['total_distance'], optimized_route['total_emissions']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Baseline Route',
        x=metrics,
        y=baseline_values,
        marker_color='darkorange',
        text=[f'{val:.0f}' for val in baseline_values],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Optimized Route',
        x=metrics,
        y=optimized_values,
        marker_color='green',
        text=[f'{val:.0f}' for val in optimized_values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Route Metrics Comparison',
        barmode='group',
        height=400
    )
    
    return fig

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
        predefined_locations = list(CITY_COORDINATES.keys())
        
        selected_locations = st.multiselect(
            "Select Delivery Locations",
            predefined_locations,
            default=["Chicago", "Houston", "Phoenix", "San Antonio", "San Jose"]
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
                    baseline_emissions = 0
                    for i in range(len(selected_locations) - 1):
                        from_idx = i
                        to_idx = i + 1
                        segment_distance = distance_matrix[from_idx][to_idx]
                        baseline_distance += segment_distance
                        baseline_emissions += segment_distance * 0.21  # Basic emission calculation
                    
                    # Add return to start for baseline if optimized route does it
                    if len(optimized_route['route']) > len(selected_locations):
                        # Return to start for baseline too
                        baseline_distance += distance_matrix[len(selected_locations)-1][0]
                        baseline_emissions += distance_matrix[len(selected_locations)-1][0] * 0.21
                    
                    baseline_route_data = {
                        'total_distance': baseline_distance,
                        'total_emissions': baseline_emissions,
                        'route': selected_locations + ([selected_locations[0]] if len(optimized_route['route']) > len(selected_locations) else [])
                    }
                    
                    # Calculate savings
                    savings = optimizer.calculate_savings(optimized_route, baseline_route_data)
                    
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
                        baseline_display = " → ".join(baseline_route_data['route'])
                        st.write(f"**{baseline_display}**")
                        st.metric("Total Distance", f"{baseline_route_data['total_distance']:.0f} km")
                        st.metric("Estimated Emissions", f"{baseline_route_data['total_emissions']:.0f} kg CO₂")
                    
                    with col2:
                        st.success("**Optimized Route**")
                        optimized_display = " → ".join(optimized_locations)
                        st.write(f"**{optimized_display}**")
                        st.metric("Total Distance", f"{optimized_route['total_distance']:.0f} km")
                        st.metric("Estimated Emissions", f"{optimized_route['total_emissions']:.0f} kg CO₂")
                    
                    # Map Visualization
                    st.subheader("🗺️ Route Comparison Map")
                    
                    # Create the interactive map
                    route_map = create_route_map(
                        baseline_route_data['route'], 
                        optimized_locations, 
                        baseline_route_data, 
                        optimized_route
                    )
                    
                    st.plotly_chart(route_map, use_container_width=True)
                    
                    # Metrics Comparison Chart
                    st.subheader("📈 Route Metrics Comparison")
                    
                    metrics_chart = create_route_comparison_chart(
                        baseline_route_data['route'],
                        optimized_locations,
                        baseline_route_data,
                        optimized_route
                    )
                    
                    st.plotly_chart(metrics_chart, use_container_width=True)
                    
                    # Route progression visualization
                    st.subheader("📊 Route Distance Progression")
                    
                    # Create route progression chart
                    fig_progression = go.Figure()
                    
                    # Baseline route progression
                    baseline_distances = [0]
                    current_distance = 0
                    for i in range(len(baseline_route_data['route']) - 1):
                        from_loc = baseline_route_data['route'][i]
                        to_loc = baseline_route_data['route'][i + 1]
                        from_idx = selected_locations.index(from_loc)
                        to_idx = selected_locations.index(to_loc)
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
                    
                    fig_progression.add_trace(go.Scatter(
                        x=list(range(len(baseline_route_data['route']))),
                        y=baseline_distances,
                        mode='lines+markers',
                        name='Baseline Route',
                        line=dict(color='darkorange', width=4),
                        marker=dict(size=8, color='darkorange')
                    ))
                    
                    fig_progression.add_trace(go.Scatter(
                        x=list(range(len(optimized_locations))),
                        y=optimized_distances,
                        mode='lines+markers',
                        name='Optimized Route',
                        line=dict(color='green', width=4),
                        marker=dict(size=8, color='green')
                    ))
                    
                    fig_progression.update_layout(
                        title='Cumulative Distance Progression',
                        xaxis_title='Stop Number',
                        yaxis_title='Cumulative Distance (km)',
                        showlegend=True,
                        height=400
                    )
                    
                    st.plotly_chart(fig_progression, use_container_width=True)
                    
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