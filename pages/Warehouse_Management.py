import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.warehouse_optimizer import WarehouseOptimizer
from utils.carbon_calculator import CarbonCalculator
from utils.data_loader import load_csv_data

def main():
    st.title("📦 Smart Warehouse Management")
    
    st.markdown("""
    Optimize warehouse operations to reduce energy consumption, improve inventory management, 
    and minimize carbon footprint through AI-driven insights and clustering analysis.
    """)
    
    # Initialize optimizers
    warehouse_optimizer = WarehouseOptimizer()
    carbon_calc = CarbonCalculator()
    
    # Load warehouse data
    warehouse_df = load_csv_data('warehouse_data.csv')
    
    # Display warehouse data
    st.subheader("🏢 Warehouse Performance Data")
    st.dataframe(warehouse_df)
    
    # Download option
    csv = warehouse_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Warehouse Data",
        data=csv,
        file_name="warehouse_data.csv",
        mime="text/csv"
    )
    
    # Key metrics overview
    st.subheader("📊 Warehouse Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_energy = warehouse_df['energy_consumption_kwh'].sum()
        st.metric("Total Energy Consumption", f"{total_energy:,.0f} kWh/month")
    
    with col2:
        avg_turnover = warehouse_df['inventory_turnover'].mean()
        st.metric("Average Inventory Turnover", f"{avg_turnover:.1f}")
    
    with col3:
        solar_count = (warehouse_df['solar_panels'] == 'Yes').sum()
        st.metric("Solar-Powered Warehouses", f"{solar_count}/{len(warehouse_df)}")
    
    with col4:
        total_capacity = warehouse_df['storage_capacity'].sum()
        st.metric("Total Storage Capacity", f"{total_capacity:,.0f} units")
    
    # Run optimization analysis
    if st.button("🔍 Analyze Warehouse Optimization", type="primary"):
        with st.spinner("Analyzing warehouse performance and calculating optimizations..."):
            # Generate optimization report
            report = warehouse_optimizer.generate_optimization_report(warehouse_df)
            
            # Display overall metrics
            st.subheader("🌍 Overall Optimization Potential")
            
            overall = report['overall_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Energy Savings Potential", 
                    f"{overall['total_potential_savings_kwh']:,.0f} kWh",
                    f"{overall['savings_percentage']:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Cost Savings Potential",
                    f"${overall['total_cost_savings_usd']:,.0f}",
                    "per month"
                )
            
            with col3:
                st.metric(
                    "Carbon Reduction",
                    f"{overall['total_carbon_savings_kg']:,.0f} kg CO₂",
                    "per month"
                )
            
            with col4:
                st.metric(
                    "Equivalent Trees",
                    f"{(overall['total_carbon_savings_kg'] / 21.77):.0f}",
                    "trees planted"
                )
            
            # Cluster analysis
            st.subheader("📈 Warehouse Clustering Analysis")
            
            cluster_data = []
            for cluster_id, summary in report['cluster_analysis'].items():
                cluster_data.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Warehouse Count': summary['count'],
                    'Avg Energy (kWh)': summary['avg_energy_consumption'],
                    'Avg Turnover': summary['avg_turnover'],
                    'Solar Penetration (%)': summary['solar_penetration']
                })
            
            cluster_df = pd.DataFrame(cluster_data)
            st.dataframe(cluster_df)
            
            # Visualization of clusters
            fig_clusters = px.scatter(
                warehouse_df,
                x='energy_consumption_kwh',
                y='inventory_turnover',
                color=report['warehouse_savings'][0]['cluster'] if report['warehouse_savings'] else 'location',
                size='square_meters',
                hover_data=['location', 'energy_source'],
                title='Warehouse Clusters: Energy vs Turnover'
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            # Individual warehouse savings
            st.subheader("💡 Individual Warehouse Optimization Opportunities")
            
            for savings in report['warehouse_savings']:
                with st.expander(f"🏭 {savings['location']} - Potential Savings: {savings['savings_percentage']:.1f}%"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Current Energy", f"{warehouse_df[warehouse_df['location'] == savings['location']]['energy_consumption_kwh'].iloc[0]:,.0f} kWh")
                        st.metric("Potential Savings", f"{savings['total_savings_kwh']:.0f} kWh")
                    
                    with col2:
                        st.metric("Cost Savings", f"${savings['total_cost_savings']:.0f}")
                        st.metric("Carbon Savings", f"{savings['total_carbon_savings']:.0f} kg CO₂")
                    
                    # Display opportunities
                    st.write("**Optimization Opportunities:**")
                    for opportunity in savings['opportunities']:
                        st.write(f"- **{opportunity['type'].replace('_', ' ').title()}**: {opportunity['recommendation']}")
            
            # Recommendations
            st.subheader("🎯 Strategic Recommendations")
            
            for rec in report['recommendations']:
                emoji = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Medium' else "🟢"
                st.write(f"{emoji} **{rec['priority']} Priority - {rec['focus']}**")
                st.write(f"   Action: {rec['action']}")
                st.write(f"   Expected Savings: {rec['expected_savings']}")
                st.write("---")
    
    # Energy consumption analysis
    st.markdown("---")
    st.subheader("⚡ Energy Consumption Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Energy by source
        fig_energy_source = px.pie(
            warehouse_df, 
            names='energy_source', 
            title='Energy Source Distribution',
            color_discrete_sequence=px.colors.sequential.Greens
        )
        st.plotly_chart(fig_energy_source, use_container_width=True)
    
    with col2:
        # Energy efficiency by location
        warehouse_df['energy_per_sqm'] = warehouse_df['energy_consumption_kwh'] / warehouse_df['square_meters']
        fig_efficiency = px.bar(
            warehouse_df.sort_values('energy_per_sqm'),
            x='location',
            y='energy_per_sqm',
            title='Energy Efficiency by Location (kWh per sqm)',
            color='energy_per_sqm',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Carbon emissions calculation
    st.subheader("🌱 Carbon Emissions Analysis")
    
    # Calculate emissions for each warehouse
    emissions_data = []
    for _, warehouse in warehouse_df.iterrows():
        energy_source_map = {
            'Grid': 'electricity_grid',
            'Mixed': 'electricity_grid',  # Simplified
            'Solar': 'solar'
        }
        
        emissions = carbon_calc.calculate_warehouse_emissions(
            warehouse['energy_consumption_kwh'],
            energy_source_map.get(warehouse['energy_source'], 'electricity_grid'),
            warehouse['square_meters']
        )
        
        emissions_data.append({
            'location': warehouse['location'],
            'emissions_kg_co2': emissions,
            'energy_consumption_kwh': warehouse['energy_consumption_kwh'],
            'energy_source': warehouse['energy_source']
        })
    
    emissions_df = pd.DataFrame(emissions_data)
    
    fig_emissions = px.bar(
        emissions_df.sort_values('emissions_kg_co2', ascending=False),
        x='location',
        y='emissions_kg_co2',
        color='energy_source',
        title='Carbon Emissions by Warehouse Location',
        labels={'emissions_kg_co2': 'Carbon Emissions (kg CO₂)'}
    )
    st.plotly_chart(fig_emissions, use_container_width=True)

if __name__ == "__main__":
    main()