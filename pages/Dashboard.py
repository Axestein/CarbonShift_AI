import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def main():
    st.title("🌍 Carbon Footprint Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Carbon Saved", "125 tCO₂e", "15%")
    with col2:
        st.metric("Fuel Cost Savings", "$45,200", "22%")
    with col3:
        st.metric("Route Optimization", "3,450 km", "18%")
    with col4:
        st.metric("Inventory Reduction", "1,200 units", "12%")
    
    # Emissions Overview
    st.subheader("Carbon Emissions Overview")
    
    # Sample data
    categories = ['Transport', 'Warehousing', 'Inventory', 'Supplier']
    baseline = [120, 45, 35, 60]
    optimized = [85, 35, 25, 45]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Baseline', x=categories, y=baseline, marker_color='red'))
    fig.add_trace(go.Bar(name='AI-Optimized', x=categories, y=optimized, marker_color='green'))
    
    fig.update_layout(barmode='group', title='Carbon Emissions by Category')
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Trends
    st.subheader("Monthly Carbon Reduction Trend")
    
    months = pd.date_range('2023-01-01', periods=12, freq='M')
    baseline_emissions = np.random.normal(100, 10, 12)
    ai_emissions = baseline_emissions * np.random.uniform(0.7, 0.9, 12)
    
    trend_df = pd.DataFrame({
        'Month': months,
        'Baseline': baseline_emissions,
        'AI_Optimized': ai_emissions
    })
    
    fig = px.line(trend_df, x='Month', y=['Baseline', 'AI_Optimized'],
                  title='Monthly Carbon Emissions Trend')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()