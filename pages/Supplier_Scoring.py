import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.supplier_scorer import SupplierScorer
from utils.data_loader import load_csv_data

def main():
    st.title("🏭 Sustainable Supplier Scoring")
    
    st.markdown("""
    Evaluate and select suppliers based on their environmental performance, energy efficiency, 
    and sustainability practices using AI-powered scoring.
    """)
    
    # Initialize scorer
    scorer = SupplierScorer()
    
    # Load supplier data
    supplier_df = load_csv_data('supplier_data.csv')
    
    # Display raw data
    st.subheader("📊 Supplier Data Overview")
    st.dataframe(supplier_df)
    
    # Download option
    csv = supplier_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Supplier Data",
        data=csv,
        file_name="supplier_data.csv",
        mime="text/csv"
    )
    
    # Train model
    if st.button("🎯 Calculate Sustainability Scores", type="primary"):
        with st.spinner("Training AI model and calculating scores..."):
            # Train the model
            training_results = scorer.train_model(supplier_df)
            
            if training_results:
                # Predict scores for all suppliers
                supplier_df['sustainability_score'] = scorer.predict_scores(supplier_df)
                
                # Display model performance
                st.subheader("🤖 Model Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Mean Absolute Error", f"{training_results['mae']:.2f}")
                with col2:
                    st.metric("R² Score", f"{training_results['r2_score']:.3f}")
                
                # Feature importance
                st.subheader("📈 Feature Importance")
                
                fig_importance = px.bar(
                    training_results['feature_importance'],
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Feature Importance in Sustainability Scoring'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
    
    # Supplier filtering and recommendations
    st.subheader("🔍 Supplier Filtering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.slider("Minimum Sustainability Score", 0, 100, 70)
    with col2:
        min_reliability = st.slider("Minimum Delivery Reliability", 80, 100, 90)
    with col3:
        max_lead_time = st.slider("Maximum Lead Time (days)", 14, 30, 25)
    
    # Apply filters
    if 'sustainability_score' in supplier_df.columns:
        filtered_df = supplier_df[
            (supplier_df['sustainability_score'] >= min_score) &
            (supplier_df['delivery_reliability'] >= min_reliability) &
            (supplier_df['lead_time_days'] <= max_lead_time)
        ].sort_values('sustainability_score', ascending=False)
        
        st.subheader("🏆 Recommended Suppliers")
        st.dataframe(filtered_df)
        
        # Supplier comparison visualization
        st.subheader("📊 Supplier Comparison")
        
        # Top 5 suppliers radar chart
        top_suppliers = filtered_df.head(5)
        
        categories = ['Carbon Footprint', 'Energy Efficiency', 'Renewable Energy', 
                     'Cost Score', 'Delivery Reliability', 'Sustainability Score']
        
        fig_radar = go.Figure()
        
        for _, supplier in top_suppliers.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[
                    100 - supplier['carbon_footprint'],  # Invert so lower is better
                    supplier['energy_efficiency'],
                    supplier['renewable_energy'],
                    supplier['cost_score'],
                    supplier['delivery_reliability'],
                    supplier['sustainability_score']
                ],
                theta=categories,
                fill='toself',
                name=supplier['name']
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Top 5 Suppliers - Performance Radar Chart",
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Sustainability score distribution
        st.subheader("📈 Sustainability Score Distribution")
        
        fig_dist = px.histogram(
            supplier_df, 
            x='sustainability_score',
            nbins=20,
            title='Distribution of Supplier Sustainability Scores',
            color_discrete_sequence=['green']
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Carbon savings calculator
        st.subheader("🌍 Carbon Savings Calculator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_supplier_score = st.slider("Current Supplier Score", 0, 100, 50)
        with col2:
            selected_supplier_score = st.selectbox(
                "New Supplier Score",
                options=filtered_df['sustainability_score'].unique()
            )
        with col3:
            annual_spend = st.number_input("Annual Spend ($)", min_value=1000, value=100000, step=1000)
        
        if st.button("Calculate Carbon Savings"):
            savings = scorer.calculate_carbon_savings(
                current_supplier_score, selected_supplier_score, annual_spend
            )
            
            st.success("💡 Potential Carbon Savings Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Emissions", f"{savings['current_annual_emissions']:.0f} kg CO₂")
            with col2:
                st.metric("Potential Savings", f"{savings['potential_savings']:.0f} kg CO₂")
            with col3:
                st.metric("Reduction", f"{savings['reduction_percentage']:.1f}%")
            with col4:
                st.metric("Score Improvement", f"{savings['score_improvement']:.1f} points")
    
    # Supplier improvement recommendations
    st.markdown("---")
    st.subheader("💡 Supplier Improvement Recommendations")
    
    if 'sustainability_score' in supplier_df.columns:
        low_scoring_suppliers = supplier_df[supplier_df['sustainability_score'] < 60]
        
        if not low_scoring_suppliers.empty:
            st.warning("🚨 Low-Scoring Suppliers Detected")
            
            for _, supplier in low_scoring_suppliers.iterrows():
                improvements = []
                
                if supplier['carbon_footprint'] > 80:
                    improvements.append("Reduce carbon footprint through energy efficiency measures")
                if supplier['energy_efficiency'] < 80:
                    improvements.append("Improve energy efficiency in manufacturing processes")
                if supplier['renewable_energy'] < 50:
                    improvements.append("Increase renewable energy usage")
                if 'None' in str(supplier['certifications']):
                    improvements.append("Obtain sustainability certifications (ISO 14001, B Corp)")
                
                st.write(f"""
                **{supplier['name']}** (Score: {supplier['sustainability_score']:.1f})
                - **Key Issues:** {', '.join(improvements[:2])}
                - **Potential Score Improvement:** 15-25 points
                """)

if __name__ == "__main__":
    main()