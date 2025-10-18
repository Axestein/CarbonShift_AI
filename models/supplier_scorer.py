import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st

class SupplierScorer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.feature_importance = None
        
    def preprocess_data(self, df):
        """Preprocess supplier data for training"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['location', 'certifications']
        for col in categorical_columns:
            if col in df_processed.columns:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
        
        # Select features for training
        feature_columns = [
            'carbon_footprint', 'energy_efficiency', 'renewable_energy',
            'cost_score', 'delivery_reliability', 'lead_time_days'
        ]
        
        # Add encoded categorical features
        for col in categorical_columns:
            if col in df_processed.columns:
                feature_columns.append(col)
        
        return df_processed, feature_columns
    
    def calculate_sustainability_score(self, row):
        """Calculate sustainability score using weighted criteria"""
        weights = {
            'carbon_footprint': 0.25,  # Lower is better
            'energy_efficiency': 0.20,  # Higher is better
            'renewable_energy': 0.15,   # Higher is better
            'cost_score': 0.15,         # Higher is better
            'delivery_reliability': 0.15, # Higher is better
            'certification_bonus': 0.10  # Bonus for certifications
        }
        
        # Normalize carbon footprint (invert since lower is better)
        carbon_score = 100 - row['carbon_footprint']
        
        # Certification bonus
        certification_bonus = 0
        if pd.notna(row.get('certifications')):
            certs = str(row['certifications']).split(',')
            if 'ISO 14001' in certs:
                certification_bonus += 20
            if 'ISO 50001' in certs:
                certification_bonus += 15
            if 'B Corp' in certs:
                certification_bonus += 25
        
        # Calculate weighted score
        score = (
            carbon_score * weights['carbon_footprint'] +
            row['energy_efficiency'] * weights['energy_efficiency'] +
            row['renewable_energy'] * weights['renewable_energy'] +
            row['cost_score'] * weights['cost_score'] +
            row['delivery_reliability'] * weights['delivery_reliability'] +
            min(certification_bonus, 50) * weights['certification_bonus']
        )
        
        return min(score, 100)  # Cap at 100
    
    def train_model(self, df):
        """Train the supplier scoring model"""
        try:
            # Calculate target variable (sustainability score)
            df['sustainability_score'] = df.apply(self.calculate_sustainability_score, axis=1)
            
            # Preprocess data
            df_processed, feature_columns = self.preprocess_data(df)
            
            # Prepare features and target
            X = df_processed[feature_columns]
            y = df_processed['sustainability_score']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Calculate training metrics
            y_pred = self.model.predict(X_scaled)
            mae = np.mean(np.abs(y - y_pred))
            r2 = self.model.score(X_scaled, y)
            
            return {
                'mae': mae,
                'r2_score': r2,
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None
    
    def predict_scores(self, df):
        """Predict sustainability scores for new suppliers"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        df_processed, feature_columns = self.preprocess_data(df)
        X = df_processed[feature_columns]
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0, 100)  # Ensure scores are between 0-100
    
    def recommend_suppliers(self, df, min_score=70, max_cost=None, min_reliability=90):
        """Recommend suppliers based on sustainability criteria"""
        if 'sustainability_score' not in df.columns:
            df['sustainability_score'] = self.predict_scores(df)
        
        # Apply filters
        filtered_df = df[df['sustainability_score'] >= min_score]
        filtered_df = filtered_df[filtered_df['delivery_reliability'] >= min_reliability]
        
        if max_cost is not None:
            # Assuming lower cost_score means higher cost (inverse relationship)
            filtered_df = filtered_df[filtered_df['cost_score'] <= (100 - max_cost)]
        
        # Sort by sustainability score (descending)
        recommended_df = filtered_df.sort_values('sustainability_score', ascending=False)
        
        return recommended_df
    
    def calculate_carbon_savings(self, current_supplier_score, new_supplier_score, annual_spend):
        """Calculate potential carbon savings from switching suppliers"""
        score_improvement = new_supplier_score - current_supplier_score
        
        # Assume 1% emission reduction per point of score improvement
        emission_reduction_percentage = score_improvement * 0.01
        
        # Base emission factor (kg CO2 per $1000 spend)
        base_emission_factor = 250  # kg CO2 per $1000
        
        annual_emissions = (annual_spend / 1000) * base_emission_factor
        potential_savings = annual_emissions * emission_reduction_percentage
        
        return {
            'current_annual_emissions': annual_emissions,
            'potential_savings': potential_savings,
            'reduction_percentage': emission_reduction_percentage * 100,
            'score_improvement': score_improvement
        }