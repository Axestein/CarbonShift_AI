import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Check if xgboost is available
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. Using alternative models.")

class DemandForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance = None
        self.features = []
        self.best_params = None
        
    def prepare_features(self, df):
        """Prepare basic features for demand forecasting"""
        df = df.copy()
        
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Encode categorical variables
        if 'product_category' in df.columns:
            df['product_category_encoded'] = self.label_encoder.fit_transform(df['product_category'])
        
        # Create lag features
        if 'actual_demand' in df.columns:
            df['demand_lag1'] = df['actual_demand'].shift(1)
            df['demand_lag2'] = df['actual_demand'].shift(2)
            df['demand_lag3'] = df['actual_demand'].shift(3)
            df['demand_lag6'] = df['actual_demand'].shift(6)
            
            # Rolling statistics
            df['demand_rolling_mean_3'] = df['actual_demand'].rolling(window=3).mean()
            df['demand_rolling_std_3'] = df['actual_demand'].rolling(window=3).std()
            df['demand_rolling_mean_6'] = df['actual_demand'].rolling(window=6).mean()
        
        return df.dropna()
    
    def enhanced_feature_engineering(self, df):
        """Add more sophisticated features"""
        df = self.prepare_features(df)
        
        # Seasonal features
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # More lag features
        for lag in [1, 3, 6, 12]:
            if len(df) > lag:
                df[f'demand_lag_{lag}'] = df['actual_demand'].shift(lag)
        
        # Rolling statistics with multiple windows
        for window in [3, 6, 12]:
            if len(df) > window:
                df[f'demand_rolling_mean_{window}'] = df['actual_demand'].rolling(window=window).mean()
                df[f'demand_rolling_std_{window}'] = df['actual_demand'].rolling(window=window).std()
        
        # Exponential moving averages
        df['demand_ema_3'] = df['actual_demand'].ewm(span=3).mean()
        df['demand_ema_6'] = df['actual_demand'].ewm(span=6).mean()
        
        # Trend features
        df['demand_trend'] = df['actual_demand'].diff()
        
        # Statistical features
        if df['actual_demand'].std() > 0:
            df['demand_zscore'] = (df['actual_demand'] - df['actual_demand'].mean()) / df['actual_demand'].std()
        else:
            df['demand_zscore'] = 0
        
        return df.dropna()
    
    def comprehensive_metrics(self, y_true, y_pred):
        """Calculate comprehensive accuracy metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,  # Convert to percentage
            'r2': r2_score(y_true, y_pred),
            'bias': np.mean(y_pred - y_true),
            'tracking_signal': np.sum(y_pred - y_true) / (np.std(y_pred - y_true) + 1e-8) if np.std(y_pred - y_true) > 0 else 0
        }
    
    def time_series_validation(self, df, n_splits=5):
        """Time-series cross validation"""
        df_processed = self.enhanced_feature_engineering(df)
        
        # Select features
        available_features = self._get_available_features(df_processed)
        X = df_processed[available_features]
        y = df_processed['actual_demand']
        
        tscv = TimeSeriesSplit(n_splits=min(n_splits, len(X)//2))
        scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Use a simple model for CV to speed up
            cv_model = RandomForestRegressor(n_estimators=50, random_state=42)
            cv_model.fit(X_train, y_train)
            y_pred = cv_model.predict(X_test)
            
            fold_metrics = self.comprehensive_metrics(y_test, y_pred)
            fold_metrics['fold'] = fold + 1
            scores.append(fold_metrics)
        
        return pd.DataFrame(scores)
    
    def _get_available_features(self, df):
        """Get list of available features in the dataset"""
        available_features = []
        
        # Basic temporal features
        basic_features = ['month', 'quarter', 'year', 'day_of_week', 'day_of_month', 'week_of_year']
        available_features.extend([f for f in basic_features if f in df.columns])
        
        # Seasonal features
        seasonal_features = ['month_sin', 'month_cos']
        available_features.extend([f for f in seasonal_features if f in df.columns])
        
        # Lag features
        lag_features = [f'demand_lag{lag}' for lag in [1, 2, 3, 6]] + \
                      [f'demand_lag_{lag}' for lag in [1, 3, 6, 12]]
        available_features.extend([f for f in lag_features if f in df.columns])
        
        # Rolling features
        rolling_features = [f'demand_rolling_mean_{w}' for w in [3, 6, 12]] + \
                          [f'demand_rolling_std_{w}' for w in [3, 6, 12]]
        available_features.extend([f for f in rolling_features if f in df.columns])
        
        # EMA features
        ema_features = ['demand_ema_3', 'demand_ema_6']
        available_features.extend([f for f in ema_features if f in df.columns])
        
        # Other features
        other_features = ['demand_trend', 'demand_zscore', 'product_category_encoded', 
                         'price', 'promotion_flag', 'seasonality']
        available_features.extend([f for f in other_features if f in df.columns])
        
        return available_features
    
    def train(self, df, enhanced_features=True, hyperparameter_tuning=False):
        """Train the demand forecasting model with enhanced options"""
        try:
            # Use enhanced feature engineering if requested
            if enhanced_features:
                df_processed = self.enhanced_feature_engineering(df)
            else:
                df_processed = self.prepare_features(df)
            
            # Get available features
            self.features = self._get_available_features(df_processed)
            
            if len(self.features) == 0:
                raise ValueError("No features available for training. Check your data columns.")
            
            X = df_processed[self.features]
            y = df_processed['actual_demand']
            
            # Time-based train-test split
            split_idx = max(int(len(X) * 0.8), 1)  # Ensure at least 1 sample
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Hyperparameter tuning if requested
            if hyperparameter_tuning and len(X_train) > 10:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10],
                    'min_samples_split': [2, 5]
                }
                
                grid_search = GridSearchCV(
                    RandomForestRegressor(random_state=42),
                    param_grid, 
                    cv=min(TimeSeriesSplit(3).get_n_splits(X_train), 3),
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
            else:
                # Standard training
                self.model.fit(X_train, y_train)
            
            self.is_trained = True
            
            # Calculate comprehensive metrics
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test) if len(X_test) > 0 else np.array([])
            
            train_metrics = self.comprehensive_metrics(y_train, train_pred)
            
            if len(X_test) > 0:
                test_metrics = self.comprehensive_metrics(y_test, test_pred)
            else:
                test_metrics = {'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0, 'bias': 0, 'tracking_signal': 0}
            
            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Cross-validation results
            cv_results = self.time_series_validation(df) if len(df) > 10 else pd.DataFrame()
            
            return {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_results': cv_results,
                'feature_importance': self.feature_importance,
                'best_params': self.best_params,
                'training_size': len(X_train),
                'testing_size': len(X_test)
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
            try:
                current_processed = self.enhanced_feature_engineering(current_data)
                if current_processed.empty:
                    break
                    
                last_row = current_processed.iloc[-1:].copy()
                
                # Update date features
                last_row['date'] = date
                last_row['month'] = date.month
                last_row['quarter'] = date.quarter
                last_row['year'] = date.year
                last_row['day_of_week'] = date.dayofweek
                last_row['day_of_month'] = date.day
                last_row['week_of_year'] = date.isocalendar().week
                
                # Update seasonal features
                last_row['month_sin'] = np.sin(2 * np.pi * date.month/12)
                last_row['month_cos'] = np.cos(2 * np.pi * date.month/12)
                
                # Ensure we only use features that were used in training
                available_features = [f for f in self.features if f in last_row.columns]
                
                if not available_features:
                    break
                    
                prediction = self.model.predict(last_row[available_features])[0]
                
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
                print(f"Prediction error for date {date}: {e}")
                break
        
        return pd.DataFrame(forecasts)
    
    def plot_forecast(self, historical_df, forecast_df):
        """Create forecast visualization with confidence intervals"""
        fig = go.Figure()
        
        # Historical actual demand
        if 'actual_demand' in historical_df.columns:
            fig.add_trace(go.Scatter(
                x=historical_df['date'],
                y=historical_df['actual_demand'],
                mode='lines+markers',
                name='Historical Actual',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
        
        # Historical forecast if available
        if 'forecast_demand' in historical_df.columns:
            fig.add_trace(go.Scatter(
                x=historical_df['date'],
                y=historical_df['forecast_demand'],
                mode='lines+markers',
                name='Historical Forecast',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=4)
            ))
        
        # Future forecast
        if not forecast_df.empty:
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast_demand'],
                mode='lines+markers',
                name='Future Forecast',
                line=dict(color='green', width=3),
                marker=dict(size=6, symbol='star')
            ))
            
            # Add confidence interval (simplified)
            confidence = forecast_df['forecast_demand'] * 0.1  # 10% confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=(forecast_df['forecast_demand'] + confidence).tolist() + 
                  (forecast_df['forecast_demand'] - confidence).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,255,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
        
        fig.update_layout(
            title='Demand Forecasting: Historical and Future Projections',
            xaxis_title='Date',
            yaxis_title='Demand (units)',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig

class EnhancedDemandForecaster(DemandForecaster):
    """Enhanced forecaster with ensemble capabilities (without xgboost dependency)"""
    
    def __init__(self):
        super().__init__()
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42)
        }
        
        # Add xgboost only if available
        if XGB_AVAILABLE:
            self.models['xgboost'] = XGBRegressor(random_state=42, verbosity=0)
        
        self.weights = None
        self.ensemble_trained = False
    
    def train_ensemble(self, df):
        """Train multiple models and learn ensemble weights"""
        try:
            df_processed = self.enhanced_feature_engineering(df)
            self.features = self._get_available_features(df_processed)
            
            if len(self.features) == 0:
                raise ValueError("No features available for training.")
                
            X = df_processed[self.features]
            y = df_processed['actual_demand']
            
            # Time-based split
            split_idx = max(int(len(X) * 0.7), 1)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train individual models
            predictions = {}
            model_performance = {}
            
            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    predictions[name] = pred
                    model_performance[name] = mean_absolute_error(y_val, pred)
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            if not model_performance:
                raise ValueError("No models could be trained successfully.")
            
            # Learn optimal weights (inverse error weighting)
            total_inverse_error = sum(1/err for err in model_performance.values())
            self.weights = {name: (1/err)/total_inverse_error for name, err in model_performance.items()}
            
            # Set main model as the best performing one
            best_model_name = min(model_performance, key=model_performance.get)
            self.model = self.models[best_model_name]
            self.is_trained = True
            self.ensemble_trained = True
            
            return {
                'model_performance': model_performance,
                'ensemble_weights': self.weights,
                'best_model': best_model_name
            }
            
        except Exception as e:
            import streamlit as st
            st.error(f"Error training ensemble: {str(e)}")
            return None