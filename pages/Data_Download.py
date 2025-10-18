import streamlit as st
import pandas as pd
from utils.data_loader import load_csv_data

def main():
    st.title("📥 Download Sample Datasets")
    
    st.markdown("""
    Download the sample datasets used in this application for your own analysis 
    or to understand the data structure better.
    """)
    
    # Demand Data
    st.subheader("📊 Demand Forecasting Data")
    demand_df = load_csv_data('sample_demand_data.csv')
    st.dataframe(demand_df.head(10))
    
    st.download_button(
        label="📥 Download Demand Data (CSV)",
        data=demand_df.to_csv(index=False),
        file_name="sample_demand_data.csv",
        mime="text/csv",
        key="demand_download"
    )
    
    st.write("**Columns:**")
    st.write("- `date`: Month of observation")
    st.write("- `product_category`: Type of product")
    st.write("- `actual_demand`: Actual units sold")
    st.write("- `forecast_demand`: Forecasted units")
    st.write("- `price`: Product price")
    st.write("- `promotion_flag`: Whether promotion was active")
    st.write("- `seasonality`: Seasonal adjustment factor")
    
    # Supplier Data
    st.subheader("🏭 Supplier Sustainability Data")
    supplier_df = load_csv_data('supplier_data.csv')
    st.dataframe(supplier_df.head(10))
    
    st.download_button(
        label="📥 Download Supplier Data (CSV)",
        data=supplier_df.to_csv(index=False),
        file_name="supplier_data.csv",
        mime="text/csv",
        key="supplier_download"
    )
    
    st.write("**Columns:**")
    st.write("- `supplier_id`: Unique supplier identifier")
    st.write("- `carbon_footprint`: Environmental impact score (lower is better)")
    st.write("- `energy_efficiency`: Energy usage efficiency (%)")
    st.write("- `renewable_energy`: Renewable energy usage (%)")
    st.write("- `cost_score`: Cost competitiveness score")
    st.write("- `delivery_reliability`: On-time delivery rate (%)")
    st.write("- `certifications`: Sustainability certifications")
    
    # Warehouse Data
    st.subheader("📦 Warehouse Operations Data")
    warehouse_df = load_csv_data('warehouse_data.csv')
    st.dataframe(warehouse_df.head(10))
    
    st.download_button(
        label="📥 Download Warehouse Data (CSV)",
        data=warehouse_df.to_csv(index=False),
        file_name="warehouse_data.csv",
        mime="text/csv",
        key="warehouse_download"
    )
    
    st.write("**Columns:**")
    st.write("- `warehouse_id`: Unique warehouse identifier")
    st.write("- `energy_consumption_kwh`: Monthly energy usage")
    st.write("- `inventory_turnover`: Inventory efficiency metric")
    st.write("- `energy_source`: Primary energy source")
    st.write("- `solar_panels`: Whether solar panels are installed")
    
    # Transportation Data
    st.subheader("🚚 Transportation Routes Data")
    transport_df = load_csv_data('transportation_routes.csv')
    st.dataframe(transport_df.head(10))
    
    st.download_button(
        label="📥 Download Transportation Data (CSV)",
        data=transport_df.to_csv(index=False),
        file_name="transportation_routes.csv",
        mime="text/csv",
        key="transport_download"
    )
    
    st.write("**Columns:**")
    st.write("- `route_id`: Unique route identifier")
    st.write("- `origin/destination`: Route endpoints")
    st.write("- `distance_km`: Route distance")
    st.write("- `baseline_fuel_l`: Fuel consumption without optimization")
    st.write("- `optimized_fuel_l`: Fuel consumption with optimization")
    st.write("- `emissions_kg_co2`: Carbon emissions")
    
    # Data Dictionary
    st.markdown("---")
    st.subheader("📖 Data Dictionary")
    
    data_dict = {
        "Metric": [
            "Carbon Footprint Score", "Energy Efficiency", "Renewable Energy %",
            "Inventory Turnover", "Delivery Reliability", "Sustainability Score"
        ],
        "Description": [
            "Environmental impact measure (0-100, lower is better)",
            "Energy utilization efficiency percentage (0-100%, higher is better)",
            "Percentage of energy from renewable sources (0-100%)",
            "How quickly inventory is sold and replaced (higher is better)",
            "On-time delivery performance percentage (0-100%)",
            "Overall sustainability rating combining multiple factors"
        ],
        "Optimal Range": [
            "0-50", "80-100%", "70-100%", "8.0+", "95-100%", "75-100"
        ]
    }
    
    st.table(pd.DataFrame(data_dict))

if __name__ == "__main__":
    main()