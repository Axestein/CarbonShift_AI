import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_emissions_comparison_chart(baseline, optimized):
    """Create emissions comparison chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Baseline',
        x=['Transport', 'Warehousing', 'Inventory', 'Total'],
        y=[baseline['transport'], baseline['warehousing'], baseline['inventory'], baseline['total']],
        marker_color='red'
    ))
    
    fig.add_trace(go.Bar(
        name='AI-Optimized',
        x=['Transport', 'Warehousing', 'Inventory', 'Total'],
        y=[optimized['transport'], optimized['warehousing'], optimized['inventory'], optimized['total']],
        marker_color='green'
    ))
    
    fig.update_layout(
        title='Carbon Emissions: Baseline vs AI-Optimized',
        yaxis_title='Carbon Emissions (tCO2e)',
        barmode='group'
    )
    
    return fig

def create_supplier_radar_chart(supplier_data):
    """Create radar chart for supplier sustainability scores"""
    categories = ['Carbon Footprint', 'Energy Efficiency', 'Renewable Energy', 
                 'Cost Efficiency', 'Delivery Reliability']
    
    fig = go.Figure()
    
    for _, supplier in supplier_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[
                supplier['carbon_footprint'],
                supplier['energy_efficiency'],
                supplier['renewable_energy'],
                supplier['cost_score'],
                supplier['delivery_reliability']
            ],
            theta=categories,
            fill='toself',
            name=supplier['name']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Supplier Sustainability Radar Chart"
    )
    
    return fig