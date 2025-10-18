import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

class DemandForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance = None
    
    def prepare_features(self, df):
        """Prepare features for demand forecasting"""
        df = df.copy()
        
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
            df['day_of_week'] = df['date'].dt.dayofweek
        
        # Encode categorical variables
        if 'product_category' in df.columns:
            df['product_category_encoded'] = self.label_encoder.fit_transform(df['product_category'])
        
        # Create lag features
        if 'actual_demand' in df.columns:
            df['demand_lag1'] = df['actual_demand'].shift(1)
            df['demand_lag2'] = df['actual_demand'].shift(2)
            df['demand_lag3'] = df['actual_demand'].shift(3)
            
            # Rolling statistics
            df['demand_rolling_mean_3'] = df['actual_demand'].rolling(window=3).mean()
            df['demand_rolling_std_3'] = df['actual_demand'].rolling(window=3).std()
        
        return df.dropna()
    
    def train(self, df):
        """Train the demand forecasting model"""
        try:
            df_processed = self.prepare_features(df)
            
            # Select available features
            available_features = ['month', 'quarter', 'year', 'day_of_week']
            
            # Add lag features if available
            for col in ['demand_lag1', 'demand_lag2', 'demand_lag3', 
                       'demand_rolling_mean_3', 'demand_rolling_std_3']:
                if col in df_processed.columns:
                    available_features.append(col)
            
            # Add encoded categorical features
            if 'product_category_encoded' in df_processed.columns:
                available_features.append('product_category_encoded')
            
            # Add other features if present
            for col in ['price', 'promotion_flag', 'seasonality']:
                if col in df_processed.columns:
                    available_features.append(col)
            
            X = df_processed[available_features]
            y = df_processed['actual_demand']
            
            # Train model
            self.model.fit(X, y)
            self.is_trained = True
            self.features = available_features
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'mae': mae,
                'rmse': rmse,
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            import streamlit as st
            st.error(f"Error training model: {str(e)}")
            return None
    
    def forecast(self, df, periods=6):
        """Generate demand forecast"""
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        # Prepare future dates
        last_date = df['date'].max() if 'date' in df.columns else pd.Timestamp.now()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                   periods=periods, freq='M')
        
        forecasts = []
        current_data = df.copy()
        
        for date in future_dates:
            # Prepare features for prediction
            current_processed = self.prepare_features(current_data)
            if current_processed.empty:
                break
                
            last_row = current_processed.iloc[-1:].copy()
            
            # Update date features
            last_row['date'] = date
            last_row['month'] = date.month
            last_row['quarter'] = date.quarter
            last_row['year'] = date.year
            last_row['day_of_week'] = date.dayofweek
            
            # Make prediction
            try:
                prediction = self.model.predict(last_row[self.features])[0]
                
                # Update lags for next prediction
                new_row = {
                    'date': date,
                    'actual_demand': prediction,
                    'forecast_demand': prediction
                }
                
                # Copy other columns if they exist
                for col in ['product_category', 'price', 'promotion_flag', 'seasonality']:
                    if col in df.columns:
                        new_row[col] = last_row[col].iloc[0] if col in last_row.columns else df[col].iloc[-1]
                
                forecasts.append(new_row)
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                
            except Exception as e:
                break
        
        return pd.DataFrame(forecasts)
    
    def plot_forecast(self, historical_df, forecast_df):
        """Create forecast visualization"""
        fig = go.Figure()
        
        # Historical actual demand
        if 'actual_demand' in historical_df.columns:
            fig.add_trace(go.Scatter(
                x=historical_df['date'],
                y=historical_df['actual_demand'],
                mode='lines+markers',
                name='Historical Actual',
                line=dict(color='blue', width=2)
            ))
        
        # Historical forecast if available
        if 'forecast_demand' in historical_df.columns:
            fig.add_trace(go.Scatter(
                x=historical_df['date'],
                y=historical_df['forecast_demand'],
                mode='lines+markers',
                name='Historical Forecast',
                line=dict(color='orange', width=2, dash='dash')
            ))
        
        # Future forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast_demand'],
            mode='lines+markers',
            name='Future Forecast',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title='Demand Forecasting: Historical and Future Projections',
            xaxis_title='Date',
            yaxis_title='Demand (units)',
            template='plotly_white',
            height=500
        )
        
        return fig