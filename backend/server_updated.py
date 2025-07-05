from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from datetime import datetime
import uuid
import os
from typing import Dict, List, Optional
from data_processor import TennisDataProcessor

app = FastAPI(title="Tennis Match Predictor API - Real Data")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RealTennisPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.data_processor = TennisDataProcessor()
        self.players_data = {}
        self.training_data = None
        
    def load_and_train(self, data_path):
        """Load real ATP data and train the model"""
        print("Loading real ATP data...")
        
        # Load the data
        if not self.data_processor.load_atp_data(data_path):
            raise Exception("Failed to load ATP data")
        
        # Get processed data
        self.players_data = self.data_processor.players_data
        self.training_data = self.data_processor.features_df
        
        print(f"Loaded data for {len(self.players_data)} players")
        print(f"Training data shape: {self.training_data.shape}")
        
        # Train the model
        self.train_model()
        
    def train_model(self):
        """Train XGBoost model on real tennis data"""
        if self.training_data is None:
            raise Exception("No training data available")
            
        print("Training model on real ATP data...")
        
        # Define features for training
        numeric_features = [
            'player1_rank', 'player2_rank', 'rank_difference',
            'player1_win_rate', 'player2_win_rate',
            'player1_surface_win_rate', 'player2_surface_win_rate',
            'player1_recent_form', 'player2_recent_form'
        ]
        
        # Encode categorical variables
        self.label_encoders['surface'] = LabelEncoder()
        self.training_data['surface_encoded'] = self.label_encoders['surface'].fit_transform(
            self.training_data['surface']
        )
        
        # Encode players
        all_players = list(set(
            self.training_data['player1'].tolist() + 
            self.training_data['player2'].tolist()
        ))
        
        self.label_encoders['player'] = LabelEncoder()
        self.label_encoders['player'].fit(all_players)
        
        self.training_data['player1_encoded'] = self.label_encoders['player'].transform(
            self.training_data['player1']
        )
        self.training_data['player2_encoded'] = self.label_encoders['player'].transform(
            self.training_data['player2']
        )
        
        # Combine all features
        self.feature_columns = numeric_features + ['surface_encoded', 'player1_encoded', 'player2_encoded']
        
        # Prepare training data
        X = self.training_data[self.feature_columns]
        y = self.training_data['player1_wins']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train XGBoost model
        print("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Calculate accuracy
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f"Model trained! Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'players_count': len(all_players),
            'matches_count': len(self.training_data)
        }
    
    def predict_match(self, player1: str, player2: str, surface: str = 'Hard'):
        """Predict match outcome using real data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        try:
            # Get player data
            p1_data = self.players_data.get(player1, {})
            p2_data = self.players_data.get(player2, {})
            
            if not p1_data or not p2_data:
                # Handle unknown players
                return self._predict_unknown_players(player1, player2, surface)
            
            # Extract features
            p1_rank = p1_data.get('current_rank', 999)
            p2_rank = p2_data.get('current_rank', 999)
            rank_diff = p1_rank - p2_rank
            
            p1_win_rate = p1_data.get('win_rate', 0.5)
            p2_win_rate = p2_data.get('win_rate', 0.5)
            
            # Surface-specific win rates
            p1_surface_stats = p1_data.get('surface_stats', {}).get(surface, {})
            p2_surface_stats = p2_data.get('surface_stats', {}).get(surface, {})
            
            p1_surface_win_rate = p1_surface_stats.get('win_rate', p1_win_rate)
            p2_surface_win_rate = p2_surface_stats.get('win_rate', p2_win_rate)
            
            p1_recent_form = p1_data.get('recent_form', 5)
            p2_recent_form = p2_data.get('recent_form', 5)
            
            # Encode categorical variables
            surface_encoded = self.label_encoders['surface'].transform([surface])[0]
            player1_encoded = self.label_encoders['player'].transform([player1])[0]
            player2_encoded = self.label_encoders['player'].transform([player2])[0]
            
            # Create feature vector
            features = [
                p1_rank, p2_rank, rank_diff,
                p1_win_rate, p2_win_rate,
                p1_surface_win_rate, p2_surface_win_rate,
                p1_recent_form, p2_recent_form,
                surface_encoded, player1_encoded, player2_encoded
            ]
            
            # Make prediction
            X = np.array(features).reshape(1, -1)
            prob = self.model.predict_proba(X)[0]
            
            # Get head-to-head data
            h2h = self.data_processor.get_head_to_head(player1, player2)
            
            return {
                'player1': player1,
                'player2': player2,
                'surface': surface,
                'player1_win_probability': float(prob[1]),
                'player2_win_probability': float(prob[0]),
                'predicted_winner': player1 if prob[1] > prob[0] else player2,
                'confidence': float(max(prob)),
                'player1_rank': p1_rank,
                'player2_rank': p2_rank,
                'player1_win_rate': round(p1_win_rate * 100, 1),
                'player2_win_rate': round(p2_win_rate * 100, 1),
                'player1_surface_win_rate': round(p1_surface_win_rate * 100, 1),
                'player2_surface_win_rate': round(p2_surface_win_rate * 100, 1),
                'head_to_head': f"{h2h['player1_wins']}-{h2h['player2_wins']}",
                'h2h_matches': h2h['total_matches']
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._predict_unknown_players(player1, player2, surface)
    
    def _predict_unknown_players(self, player1, player2, surface):
        """Handle prediction for unknown players"""
        return {
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'player1_win_probability': 0.5,
            'player2_win_probability': 0.5,
            'predicted_winner': player1,
            'confidence': 0.5,
            'player1_rank': 999,
            'player2_rank': 999,
            'player1_win_rate': 50.0,
            'player2_win_rate': 50.0,
            'player1_surface_win_rate': 50.0,
            'player2_surface_win_rate': 50.0,
            'head_to_head': "0-0",
            'h2h_matches': 0,
            'note': "One or both players not found in database"
        }

# Initialize predictor
predictor = RealTennisPredictor()

@app.on_event("startup")
async def startup_event():
    """Load data and train model on startup"""
    try:
        data_path = "/home/ubuntu/Tennis_repo/data/atp_matches_2024.csv"
        predictor.load_and_train(data_path)
        print("Real ATP data loaded and model trained successfully!")
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_trained": predictor.model is not None,
        "players_loaded": len(predictor.players_data),
        "data_source": "Real ATP 2024 Data"
    }

@app.get("/api/players")
async def get_players():
    """Get list of available players with real stats"""
    if not predictor.players_data:
        return {"players": []}
    
    # Get top players by ranking
    top_players = predictor.data_processor.get_top_players(100)
    
    players_list = [
        {
            "name": player["name"],
            "rank": player["current_rank"],
            "total_matches": player["total_matches"],
            "wins": player["total_wins"],
            "losses": player["total_losses"],
            "win_rate": round(player["win_rate"] * 100, 1),
            "recent_form": f"{player['recent_form']}/10",
            "last_match": player["last_match_date"],
            "grass_win_rate": round(player["surface_stats"].get("Grass", {}).get("win_rate", 0) * 100, 1),
            "hard_win_rate": round(player["surface_stats"].get("Hard", {}).get("win_rate", 0) * 100, 1),
            "clay_win_rate": round(player["surface_stats"].get("Clay", {}).get("win_rate", 0) * 100, 1)
        }
        for player in top_players
    ]
    
    return {"players": players_list}

@app.post("/api/predict")
async def predict_match(request: dict):
    """Predict tennis match outcome using real data"""
    try:
        player1 = request.get("player1")
        player2 = request.get("player2")
        surface = request.get("surface", "Hard")
        
        if not player1 or not player2:
            raise HTTPException(status_code=400, detail="Both players are required")
            
        if player1 == player2:
            raise HTTPException(status_code=400, detail="Players must be different")
        
        prediction = predictor.predict_match(player1, player2, surface)
        
        return prediction
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/player/{player_name}")
async def get_player_details(player_name: str):
    """Get detailed player information"""
    try:
        player_info = predictor.data_processor.get_player_info(player_name)
        
        if not player_info:
            raise HTTPException(status_code=404, detail="Player not found")
        
        return {
            "player": player_info,
            "surface_breakdown": player_info.get("surface_stats", {}),
            "ranking_info": {
                "current_rank": player_info.get("current_rank"),
                "total_matches": player_info.get("total_matches"),
                "win_percentage": round(player_info.get("win_rate", 0) * 100, 1)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/head-to-head/{player1}/{player2}")
async def get_head_to_head(player1: str, player2: str):
    """Get head-to-head record between two players"""
    try:
        h2h = predictor.data_processor.get_head_to_head(player1, player2)
        
        return {
            "player1": player1,
            "player2": player2,
            "head_to_head": h2h,
            "summary": f"{player1} leads {h2h['player1_wins']}-{h2h['player2_wins']}" if h2h['player1_wins'] > h2h['player2_wins']
                      else f"{player2} leads {h2h['player2_wins']}-{h2h['player1_wins']}" if h2h['player2_wins'] > h2h['player1_wins']
                      else f"Tied {h2h['player1_wins']}-{h2h['player2_wins']}" if h2h['total_matches'] > 0
                      else "No previous meetings"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/wimbledon/predictions")
async def get_wimbledon_predictions():
    """Get Wimbledon 2025 predictions for top matchups using real data"""
    try:
        # Get top grass court players
        top_players = predictor.data_processor.get_top_players(20)
        
        # Filter for players with good grass court records
        grass_specialists = [
            p for p in top_players 
            if p.get('surface_stats', {}).get('Grass', {}).get('matches', 0) > 0
        ][:8]
        
        if len(grass_specialists) < 4:
            # Fallback to top ranked players
            grass_specialists = top_players[:8]
        
        # Create interesting matchups
        wimbledon_matches = []
        for i in range(0, min(len(grass_specialists), 8), 2):
            if i + 1 < len(grass_specialists):
                wimbledon_matches.append((
                    grass_specialists[i]['name'], 
                    grass_specialists[i + 1]['name']
                ))
        
        predictions = []
        for player1, player2 in wimbledon_matches:
            pred = predictor.predict_match(player1, player2, 'Grass')
            predictions.append(pred)
        
        return {
            "tournament": "Wimbledon 2025",
            "surface": "Grass",
            "predictions": predictions,
            "generated_at": datetime.utcnow().isoformat(),
            "data_source": "Real ATP 2024 Data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/info")
async def get_model_info():
    """Get model information and feature importance"""
    try:
        if predictor.model is None:
            return {"error": "Model not trained"}
        
        # Get feature importance
        importance = predictor.model.feature_importances_
        feature_importance = [
            {"feature": col, "importance": float(imp)}
            for col, imp in zip(predictor.feature_columns, importance)
        ]
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "model_type": "XGBoost Classifier (Real Data)",
            "feature_count": len(predictor.feature_columns),
            "feature_importance": feature_importance,
            "players_count": len(predictor.players_data),
            "training_samples": len(predictor.training_data) if predictor.training_data is not None else 0,
            "data_source": "ATP 2024 Official Data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

