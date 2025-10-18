import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

class WarehouseOptimizer:
    def __init__(self):
        self.kmeans = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def preprocess_warehouse_data(self, df):
        """Preprocess warehouse data for optimization"""
        df_processed = df.copy()
        
        # Convert categorical variables
        df_processed['solar_panels'] = df_processed['solar_panels'].map({'Yes': 1, 'No': 0})
        
        # Calculate energy efficiency metric (lower is better)
        df_processed['energy_per_sqm'] = df_processed['energy_consumption_kwh'] / df_processed['square_meters']
        df_processed['energy_per_capacity'] = df_processed['energy_consumption_kwh'] / df_processed['storage_capacity']
        
        # Select features for clustering
        feature_columns = [
            'energy_consumption_kwh', 'square_meters', 'storage_capacity',
            'inventory_turnover', 'energy_per_sqm', 'energy_per_capacity',
            'solar_panels', 'operational_hours'
        ]
        
        return df_processed, feature_columns
    
    def optimize_warehouse_clusters(self, df, n_clusters=3):
        """Cluster warehouses based on performance characteristics"""
        df_processed, feature_columns = self.preprocess_warehouse_data(df)
        X = df_processed[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        df_processed['cluster'] = clusters
        self.is_trained = True
        
        # Calculate cluster characteristics
        cluster_summary = {}
        for cluster_id in range(n_clusters):
            cluster_data = df_processed[df_processed['cluster'] == cluster_id]
            cluster_summary[cluster_id] = {
                'count': len(cluster_data),
                'avg_energy_consumption': cluster_data['energy_consumption_kwh'].mean(),
                'avg_turnover': cluster_data['inventory_turnover'].mean(),
                'avg_energy_per_sqm': cluster_data['energy_per_sqm'].mean(),
                'solar_penetration': cluster_data['solar_panels'].mean() * 100,
                'warehouses': cluster_data['warehouse_id'].tolist()
            }
        
        return df_processed, cluster_summary
    
    def calculate_energy_savings_potential(self, df):
        """Calculate potential energy savings for each warehouse"""
        savings_data = []
        
        for _, warehouse in df.iterrows():
            # Identify improvement opportunities
            opportunities = []
            
            # Energy consumption analysis
            if warehouse['energy_consumption_kwh'] > df['energy_consumption_kwh'].median():
                savings_kwh = warehouse['energy_consumption_kwh'] * 0.15  # 15% savings potential
                opportunities.append({
                    'type': 'energy_efficiency',
                    'savings_kwh': savings_kwh,
                    'cost_savings': savings_kwh * 0.12,  # $0.12 per kWh
                    'carbon_savings': savings_kwh * 0.5,  # 0.5 kg CO2 per kWh
                    'recommendation': 'Upgrade to LED lighting, optimize HVAC'
                })
            
            # Solar potential
            if warehouse['solar_panels'] == 0:  # No solar panels
                solar_potential = warehouse['square_meters'] * 0.15 * 150  # 15% coverage, 150 kWh/sqm/year
                opportunities.append({
                    'type': 'solar_installation',
                    'savings_kwh': solar_potential / 12,  # Monthly
                    'cost_savings': (solar_potential / 12) * 0.12,
                    'carbon_savings': (solar_potential / 12) * 0.5,
                    'recommendation': 'Install solar panels on rooftop'
                })
            
            # Inventory optimization
            if warehouse['inventory_turnover'] < df['inventory_turnover'].median():
                space_savings = warehouse['storage_capacity'] * 0.1  # 10% space optimization
                energy_savings = space_savings * 0.5  # 0.5 kWh per unit space
                opportunities.append({
                    'type': 'inventory_optimization',
                    'savings_kwh': energy_savings,
                    'cost_savings': energy_savings * 0.12,
                    'carbon_savings': energy_savings * 0.5,
                    'recommendation': 'Optimize inventory levels and storage layout'
                })
            
            total_savings_kwh = sum(opp['savings_kwh'] for opp in opportunities)
            total_cost_savings = sum(opp['cost_savings'] for opp in opportunities)
            total_carbon_savings = sum(opp['carbon_savings'] for opp in opportunities)
            
            savings_data.append({
                'warehouse_id': warehouse['warehouse_id'],
                'location': warehouse['location'],
                'opportunities': opportunities,
                'total_savings_kwh': total_savings_kwh,
                'total_cost_savings': total_cost_savings,
                'total_carbon_savings': total_carbon_savings,
                'savings_percentage': (total_savings_kwh / warehouse['energy_consumption_kwh']) * 100
            })
        
        return savings_data
    
    def generate_optimization_report(self, df):
        """Generate comprehensive optimization report"""
        # Cluster warehouses
        df_clustered, cluster_summary = self.optimize_warehouse_clusters(df)
        
        # Calculate savings potential
        savings_potential = self.calculate_energy_savings_potential(df)
        
        # Overall statistics
        total_current_energy = df['energy_consumption_kwh'].sum()
        total_potential_savings = sum(item['total_savings_kwh'] for item in savings_potential)
        total_cost_savings = sum(item['total_cost_savings'] for item in savings_potential)
        total_carbon_savings = sum(item['total_carbon_savings'] for item in savings_potential)
        
        report = {
            'overall_metrics': {
                'total_current_energy_kwh': total_current_energy,
                'total_potential_savings_kwh': total_potential_savings,
                'savings_percentage': (total_potential_savings / total_current_energy) * 100,
                'total_cost_savings_usd': total_cost_savings,
                'total_carbon_savings_kg': total_carbon_savings
            },
            'cluster_analysis': cluster_summary,
            'warehouse_savings': savings_potential,
            'recommendations': self.generate_recommendations(cluster_summary, savings_potential)
        }
        
        return report
    
    def generate_recommendations(self, cluster_summary, savings_potential):
        """Generate targeted recommendations based on analysis"""
        recommendations = []
        
        # High-energy cluster recommendations
        high_energy_clusters = [cid for cid, summary in cluster_summary.items() 
                               if summary['avg_energy_consumption'] > np.mean([s['avg_energy_consumption'] for s in cluster_summary.values()])]
        
        for cluster_id in high_energy_clusters:
            recommendations.append({
                'priority': 'High',
                'cluster': cluster_id,
                'focus': 'Energy Efficiency',
                'action': 'Implement comprehensive energy audit and upgrade inefficient equipment',
                'expected_savings': '15-25% energy reduction'
            })
        
        # Low-turnover cluster recommendations
        low_turnover_clusters = [cid for cid, summary in cluster_summary.items() 
                                if summary['avg_turnover'] < np.mean([s['avg_turnover'] for s in cluster_summary.values()])]
        
        for cluster_id in low_turnover_clusters:
            recommendations.append({
                'priority': 'Medium',
                'cluster': cluster_id,
                'focus': 'Inventory Optimization',
                'action': 'Review inventory management policies and storage layout',
                'expected_savings': '10-20% space utilization improvement'
            })
        
        # Solar potential recommendations
        low_solar_clusters = [cid for cid, summary in cluster_summary.items() 
                             if summary['solar_penetration'] < 50]
        
        for cluster_id in low_solar_clusters:
            recommendations.append({
                'priority': 'Medium',
                'cluster': cluster_id,
                'focus': 'Renewable Energy',
                'action': 'Conduct solar feasibility study and explore installation options',
                'expected_savings': '20-40% grid electricity offset'
            })
        
        return recommendations