import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from models.demand_forecaster import DemandForecaster
from utils.data_loader import load_sample_demand_data, load_csv_data

def main():
    st.title("📈 Predictive Demand Forecasting")
    
    st.markdown("""
    This module uses machine learning to predict future demand, helping reduce overproduction 
    and minimize inventory-related carbon emissions.
    """)
    
    # Data source selection
    data_source = st.radio("Choose data source:", ["Use Sample Data", "Upload CSV File"])
    
    if data_source == "Use Sample Data":
        # Load sample data
        df = load_sample_demand_data()
        
        st.subheader("Sample Demand Data")
        st.dataframe(df.head(10))
        
    else:
        # File upload
        uploaded_file = st.file_uploader("Upload demand data CSV", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data")
            st.dataframe(df.head(10))
        else:
            st.info("Please upload a CSV file or use sample data.")
            return
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Train model
    if st.button("🎯 Train Forecasting Model", type="primary"):
        with st.spinner("Training AI model..."):
            forecaster = DemandForecaster()
            
            # Train on the data
            training_results = forecaster.train(df)
            
            if training_results:
                st.success("✅ Model trained successfully!")
                
                # Display model performance
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error", f"{training_results['mae']:.2f}")
                with col2:
                    st.metric("Root Mean Square Error", f"{training_results['rmse']:.2f}")
                
                # Generate forecast
                forecast_periods = st.slider("Forecast Periods (months)", 3, 12, 6)
                forecast_df = forecaster.forecast(df, forecast_periods)
                
                # Display forecast results
                st.subheader("📊 Forecast Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_historical = df['actual_demand'].mean()
                    st.metric("Historical Avg Demand", f"{avg_historical:.0f} units")
                    
                with col2:
                    avg_forecast = forecast_df['forecast_demand'].mean()
                    st.metric("Forecasted Avg Demand", f"{avg_forecast:.0f} units")
                    change_pct = ((avg_forecast - avg_historical) / avg_historical) * 100
                    st.metric("Demand Change", f"{change_pct:+.1f}%")
                
                # Plot forecast
                fig = forecaster.plot_forecast(df, forecast_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Carbon impact analysis
                st.subheader("🌍 Carbon Reduction Impact")
                
                inventory_reduction = st.slider("Expected Inventory Reduction (%)", 5, 30, 15)
                carbon_savings = avg_forecast * (inventory_reduction/100) * 2.5  # kg CO2 per unit
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Inventory Reduction", f"{inventory_reduction}%")
                with col2:
                    st.metric("Monthly Carbon Savings", f"{carbon_savings:.0f} kg CO₂")
                with col3:
                    st.metric("Annual Impact", f"{carbon_savings * 12 / 1000:.1f} tCO₂e")
                
                # Feature importance
                if hasattr(forecaster, 'feature_importance') and forecaster.feature_importance is not None:
                    st.subheader("🔍 Feature Importance")
                    fig_importance = px.bar(
                        forecaster.feature_importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Feature Importance in Demand Forecasting'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Sample data download
    st.markdown("---")
    st.subheader("📥 Download Sample Data")
    
    sample_df = load_sample_demand_data()
    csv = sample_df.to_csv(index=False)
    
    st.download_button(
        label="Download Sample Demand Data (CSV)",
        data=csv,
        file_name="sample_demand_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()