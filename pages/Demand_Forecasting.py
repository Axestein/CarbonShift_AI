import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

try:
    from models.demand_forecaster import DemandForecaster, EnhancedDemandForecaster
except ImportError as e:
    st.error(f"Error importing forecasting modules: {e}")
    st.info("Please ensure all required dependencies are installed.")

try:
    from utils.data_loader import load_sample_demand_data, load_csv_data
except ImportError:
    # Fallback data loader
    def load_sample_demand_data():
        """Generate sample demand data if data_loader is not available"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-01', freq='M')
        np.random.seed(42)
        
        data = {
            'date': dates,
            'actual_demand': 100 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12) + 
                           10 * np.random.randn(len(dates)),
            'product_category': np.random.choice(['A', 'B', 'C'], len(dates)),
            'price': 50 + 10 * np.random.randn(len(dates)),
            'promotion_flag': np.random.choice([0, 1], len(dates), p=[0.7, 0.3])
        }
        
        return pd.DataFrame(data)
    
    def load_csv_data(uploaded_file):
        """Load CSV data with basic error handling"""
        return pd.read_csv(uploaded_file)

def main():
    st.title("📈 Advanced Predictive Demand Forecasting")
    
    st.markdown("""
    This enhanced module uses machine learning with advanced feature engineering and validation 
    to predict future demand, helping reduce overproduction and minimize inventory-related carbon emissions.
    """)
    
    # Data source selection
    data_source = st.radio("Choose data source:", ["Use Sample Data", "Upload CSV File"])
    
    if data_source == "Use Sample Data":
        # Load sample data
        df = load_sample_demand_data()
        
        st.subheader("Sample Demand Data")
        st.dataframe(df.head(10))
        
        # Show data summary
        with st.expander("Data Summary"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
            with col3:
                st.metric("Avg Demand", f"{df['actual_demand'].mean():.0f} units")
        
    else:
        # File upload
        uploaded_file = st.file_uploader("Upload demand data CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = load_csv_data(uploaded_file)
                st.subheader("Uploaded Data")
                st.dataframe(df.head(10))
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
        else:
            st.info("Please upload a CSV file or use sample data.")
            return
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Check if required columns exist
    if 'actual_demand' not in df.columns:
        st.error("❌ 'actual_demand' column is required for forecasting.")
        st.info("Please ensure your data contains an 'actual_demand' column with historical demand values.")
        return
    
    # Model configuration
    st.sidebar.subheader("Model Configuration")
    
    model_type = st.sidebar.selectbox(
        "Select Model Type:",
        ["Standard Random Forest", "Enhanced Random Forest", "Ensemble Model"]
    )
    
    enhanced_features = st.sidebar.checkbox("Use Enhanced Feature Engineering", value=True)
    hyperparameter_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=False)
    forecast_periods = st.sidebar.slider("Forecast Periods (months)", 3, 24, 6)
    
    # Train model
    if st.button("🎯 Train Forecasting Model", type="primary"):
        with st.spinner("Training AI model with advanced features..."):
            try:
                if model_type == "Ensemble Model":
                    forecaster = EnhancedDemandForecaster()
                    ensemble_results = forecaster.train_ensemble(df)
                    
                    if ensemble_results:
                        st.success("✅ Ensemble model trained successfully!")
                        
                        # Display ensemble results
                        st.subheader("🤖 Ensemble Model Performance")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Best Model", ensemble_results['best_model'])
                        with col2:
                            best_perf = ensemble_results['model_performance'][ensemble_results['best_model']]
                            st.metric("Best Model MAE", f"{best_perf:.2f}")
                        with col3:
                            avg_perf = np.mean(list(ensemble_results['model_performance'].values()))
                            st.metric("Average MAE", f"{avg_perf:.2f}")
                        
                        # Show ensemble weights
                        st.write("**Ensemble Weights:**")
                        weights_df = pd.DataFrame.from_dict(ensemble_results['ensemble_weights'], 
                                                          orient='index', columns=['Weight'])
                        st.dataframe(weights_df.style.format({'Weight': '{:.2%}'}))
                        
                        training_results = None  # Ensemble uses different training approach
                    else:
                        st.error("Failed to train ensemble model. Using standard model instead.")
                        forecaster = DemandForecaster()
                        training_results = forecaster.train(df, enhanced_features=enhanced_features)
                else:
                    if model_type == "Enhanced Random Forest":
                        forecaster = DemandForecaster()
                    else:
                        forecaster = DemandForecaster()
                        enhanced_features = False  # Standard model uses basic features
                    
                    # Train on the data
                    training_results = forecaster.train(
                        df, 
                        enhanced_features=enhanced_features,
                        hyperparameter_tuning=hyperparameter_tuning
                    )
                
                if training_results or model_type == "Ensemble Model":
                    st.success("✅ Model trained successfully!")
                    
                    # Display model performance for non-ensemble models
                    if training_results:
                        st.subheader("📊 Model Performance Metrics")
                        
                        # Training vs Test metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Training MAE", f"{training_results['train_metrics']['mae']:.2f}")
                        with col2:
                            st.metric("Test MAE", f"{training_results['test_metrics']['mae']:.2f}")
                        with col3:
                            st.metric("Test MAPE", f"{training_results['test_metrics']['mape']:.1f}%")
                        with col4:
                            st.metric("Test R²", f"{training_results['test_metrics']['r2']:.3f}")
                        
                        # Show hyperparameters if tuned
                        if training_results['best_params']:
                            st.write("**Optimal Hyperparameters:**", training_results['best_params'])
                        
                        # Cross-validation results
                        if not training_results['cv_results'].empty:
                            with st.expander("📋 Cross-Validation Results"):
                                st.dataframe(training_results['cv_results'])
                                
                                # Plot CV results
                                fig_cv = px.box(
                                    training_results['cv_results'].drop(columns=['fold']),
                                    title='Cross-Validation Metrics Distribution'
                                )
                                st.plotly_chart(fig_cv, use_container_width=True)
                    
                    # Generate forecast
                    forecast_df = forecaster.forecast(df, forecast_periods)
                    
                    if forecast_df.empty:
                        st.warning("No forecast could be generated. Check your data and model configuration.")
                        return
                    
                    # Display forecast results
                    st.subheader("📈 Forecast Results")
                    
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
                    
                    # Show forecast table
                    with st.expander("📄 Detailed Forecast Data"):
                        st.dataframe(forecast_df.style.format({
                            'forecast_demand': '{:.0f}',
                            'date': '{:%Y-%m-%d}'
                        }))
                    
                    # Carbon impact analysis
                    st.subheader("🌍 Carbon Reduction Impact")
                    
                    inventory_reduction = st.slider("Expected Inventory Reduction (%)", 5, 30, 15, key="inventory_slider")
                    carbon_per_unit = st.slider("CO₂ per unit (kg)", 1.0, 10.0, 2.5, step=0.1, key="carbon_slider")
                    
                    carbon_savings = avg_forecast * (inventory_reduction/100) * carbon_per_unit
                    
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
                            forecaster.feature_importance.head(10),  # Show top 10 features
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Top 10 Most Important Features in Demand Forecasting'
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during model training or forecasting: {str(e)}")
                st.info("Try using simpler model configurations or check your data format.")
    
    # Model comparison section
    st.markdown("---")
    st.subheader("🔬 Model Comparison")
    
    if st.button("Compare Different Models"):
        with st.spinner("Training multiple models for comparison..."):
            try:
                # Train different configurations
                models_comparison = {}
                
                # Standard model
                standard_forecaster = DemandForecaster()
                standard_results = standard_forecaster.train(df, enhanced_features=False, hyperparameter_tuning=False)
                if standard_results:
                    models_comparison['Standard RF'] = standard_results['test_metrics']['mae']
                
                # Enhanced model
                enhanced_forecaster = DemandForecaster()
                enhanced_results = enhanced_forecaster.train(df, enhanced_features=True, hyperparameter_tuning=False)
                if enhanced_results:
                    models_comparison['Enhanced RF'] = enhanced_results['test_metrics']['mae']
                
                # Tuned model
                tuned_forecaster = DemandForecaster()
                tuned_results = tuned_forecaster.train(df, enhanced_features=True, hyperparameter_tuning=True)
                if tuned_results:
                    models_comparison['Tuned RF'] = tuned_results['test_metrics']['mae']
                
                # Display comparison
                if models_comparison:
                    comparison_df = pd.DataFrame.from_dict(models_comparison, orient='index', columns=['Test MAE'])
                    st.write("**Model Performance Comparison (Lower is Better):**")
                    st.dataframe(comparison_df.sort_values('Test MAE'))
                    
                    # Plot comparison
                    fig_compare = px.bar(
                        comparison_df.reset_index(),
                        x='index',
                        y='Test MAE',
                        title='Model Comparison - Test MAE',
                        labels={'index': 'Model Type', 'Test MAE': 'Mean Absolute Error'}
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
                else:
                    st.warning("Could not compare models. Check if all models trained successfully.")
                    
            except Exception as e:
                st.error(f"Error during model comparison: {str(e)}")
    
    # Sample data download
    st.markdown("---")
    st.subheader("📥 Download Sample Data")
    
    sample_df = load_sample_demand_data()
    csv = sample_df.to_csv(index=False)
    
    st.download_button(
        label="Download Sample Demand Data (CSV)",
        data=csv,
        file_name="sample_demand_data.csv",
        mime="text/csv",
        help="Download sample data to understand the required format"
    )

if __name__ == "__main__":
    main()

#The error occurs because the dataset is too small for cross-validation. Let me fix this by adding proper validation for small datasets and removing the cross-validation requirement for hyperparameter tuning.