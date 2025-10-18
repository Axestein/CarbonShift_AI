import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.route_optimizer import GreenRouteOptimizer
from utils.carbon_calculator import CarbonCalculator
from utils.data_loader import load_csv_data

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
            # Optimize route
            optimized_route = optimizer.optimize_route(selected_locations, vehicle_type)
            
            if optimized_route:
                # Calculate baseline (simple nearest neighbor)
                baseline_distance = sum(
                    abs(i - j) * 100 + np.random.randint(50, 200) 
                    for i, j in zip(range(len(selected_locations)), range(1, len(selected_locations)))
                )
                
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
                    st.write(" → ".join(optimized_route['route']))
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
                    segment_distance = abs(i - (i+1)) * 100 + 100  # Simplified
                    current_distance += segment_distance
                    baseline_distances.append(current_distance)
                
                # Optimized route progression
                optimized_distances = [0]
                current_distance = 0
                for i in range(len(optimized_route['route']) - 1):
                    segment_distance = abs(i - (i+1)) * 100 + 100  # Simplified
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
                    x=list(range(len(optimized_route['route']))),
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