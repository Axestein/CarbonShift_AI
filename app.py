import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="CarbonShift_AI: AI-Driven Carbon Footprint Reduction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .feature-card {
        background-color: #000000;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    load_css()
    
    # Header Section
    st.markdown('<h1 class="main-header">🌱 AI-Driven Carbon Footprint Reduction</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Supply Chain Management for a Sustainable Future")
    
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### The Problem
        Modern supply chains are complex, global, and resource-intensive, contributing significantly to greenhouse gas emissions. 
        Key challenges include:
        
        - ❌ Inefficient transportation routes
        - ❌ Overproduction and excess inventory
        - ❌ Non-sustainable supplier selection
        - ❌ Energy-inefficient warehousing
        """)
    
    with col2:
        st.markdown("""
        ### Our AI Solution
        Leveraging machine learning and advanced analytics to:
        
        - ✅ Optimize transportation routes
        - ✅ Accurate demand forecasting
        - ✅ Sustainable supplier scoring
        - ✅ Smart warehouse management
        """)
    
    st.markdown("---")
    
    # Key Metrics Overview
    st.subheader("📊 Potential Impact Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>30-40%</h3>
            <p>Reduction in Carbon Emissions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>20-30%</h3>
            <p>Fuel Cost Savings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>15-25%</h3>
            <p>Inventory Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>25-35%</h3>
            <p>Supply Chain Efficiency Gain</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features Overview
    st.markdown("---")
    st.subheader("🚀 Core AI Modules")
    
    # Feature cards in 2x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Predictive Demand Forecasting</h4>
            <p>ML models for accurate demand prediction to reduce overproduction and excess inventory</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🚚 Green Route Optimizer</h4>
            <p>AI-powered route planning to minimize fuel consumption and emissions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🏭 Sustainable Supplier Scoring</h4>
            <p>AI assessment of supplier sustainability performance</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📦 Smart Warehouse Management</h4>
            <p>Optimize inventory and energy usage in warehouse operations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo Call-to-Action
    st.markdown("---")
    st.success("🚀 **Ready to optimize your supply chain?** Use the sidebar to navigate to specific modules and start reducing your carbon footprint today!")
    
    # Carbon Savings Visualization
    st.markdown("---")
    st.subheader("🌍 Potential Environmental Impact")
    
    # Sample data for demonstration
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    baseline_emissions = [120, 115, 125, 130, 128, 135]
    optimized_emissions = [110, 100, 95, 90, 85, 80]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=baseline_emissions, 
                            mode='lines+markers', name='Baseline Emissions',
                            line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=months, y=optimized_emissions, 
                            mode='lines+markers', name='AI-Optimized',
                            line=dict(color='green', width=3)))
    
    fig.update_layout(
        title='Monthly Carbon Emissions Reduction with AI Optimization',
        xaxis_title='Month',
        yaxis_title='Carbon Emissions (tCO2e)',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()