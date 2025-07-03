from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import requests
from datetime import datetime
import uuid
import pickle
import io
from typing import Dict, List, Optional

# MongoDB setup
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'tennis_predictor')

client = MongoClient(mongo_url)
db = client[db_name]

app = FastAPI(title="Tennis Match Predictor API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model and encoders
model = None
label_encoders = {}
feature_columns = []

class TennisPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.player_stats = {}
        
    def create_sample_data(self):
        """Create realistic tennis match data for training"""
        np.random.seed(42)
        
        # Top ATP players with realistic stats
        players = [
            'Novak Djokovic', 'Carlos Alcaraz', 'Daniil Medvedev', 'Jannik Sinner',
            'Andrey Rublev', 'Stefanos Tsitsipas', 'Alexander Zverev', 'Holger Rune',
            'Taylor Fritz', 'Grigor Dimitrov', 'Tommy Paul', 'Ben Shelton',
            'Frances Tiafoe', 'Sebastian Korda', 'Hubert Hurkacz', 'Felix Auger-Aliassime'
        ]
        
        # Player rankings (simplified)
        rankings = {player: i+1 for i, player in enumerate(players)}
        
        # Surface types for Wimbledon (grass court)
        surfaces = ['Grass', 'Hard', 'Clay']
        
        matches = []
        for i in range(1000):  # Generate 1000 sample matches
            player1 = np.random.choice(players)
            player2 = np.random.choice([p for p in players if p != player1])
            
            surface = np.random.choice(surfaces, p=[0.3, 0.5, 0.2])  # More grass for Wimbledon focus
            
            # Realistic features
            rank_diff = rankings[player1] - rankings[player2]
            player1_rank = rankings[player1]
            player2_rank = rankings[player2]
            
            # Simulate head-to-head (random but consistent)
            h2h_key = tuple(sorted([player1, player2]))
            h2h_wins_p1 = hash(h2h_key + (player1,)) % 10
            h2h_wins_p2 = hash(h2h_key + (player2,)) % 10
            
            # Recent form (wins in last 10 matches)
            recent_form_p1 = np.random.randint(3, 9)
            recent_form_p2 = np.random.randint(3, 9)
            
            # Calculate win probability based on realistic factors
            prob_p1_wins = 0.5
            if rank_diff < 0:  # Player 1 is higher ranked (lower number)
                prob_p1_wins += abs(rank_diff) * 0.02
            else:
                prob_p1_wins -= rank_diff * 0.02
                
            # Surface advantage (Djokovic strong on grass, etc.)
            if surface == 'Grass':
                if player1 in ['Novak Djokovic', 'Andy Murray']:
                    prob_p1_wins += 0.1
                if player2 in ['Novak Djokovic', 'Andy Murray']:
                    prob_p1_wins -= 0.1
                    
            # Recent form influence
            prob_p1_wins += (recent_form_p1 - recent_form_p2) * 0.02
            
            # Head-to-head influence
            if h2h_wins_p1 + h2h_wins_p2 > 0:
                prob_p1_wins += (h2h_wins_p1 - h2h_wins_p2) / (h2h_wins_p1 + h2h_wins_p2) * 0.1
            
            prob_p1_wins = max(0.1, min(0.9, prob_p1_wins))
            
            winner = player1 if np.random.random() < prob_p1_wins else player2
            
            matches.append({
                'player1': player1,
                'player2': player2,
                'player1_rank': player1_rank,
                'player2_rank': player2_rank,
                'surface': surface,
                'rank_difference': rank_diff,
                'h2h_p1_wins': h2h_wins_p1,
                'h2h_p2_wins': h2h_wins_p2,
                'recent_form_p1': recent_form_p1,
                'recent_form_p2': recent_form_p2,
                'winner': winner,
                'player1_wins': 1 if winner == player1 else 0
            })
            
        return pd.DataFrame(matches)
    
    def train_model(self):
        """Train XGBoost model on tennis data"""
        print("Creating training data...")
        df = self.create_sample_data()
        
        # Feature engineering
        features = [
            'player1_rank', 'player2_rank', 'rank_difference',
            'h2h_p1_wins', 'h2h_p2_wins', 'recent_form_p1', 'recent_form_p2'
        ]
        
        # Encode categorical variables
        self.label_encoders['surface'] = LabelEncoder()
        df['surface_encoded'] = self.label_encoders['surface'].fit_transform(df['surface'])
        features.append('surface_encoded')
        
        self.label_encoders['player1'] = LabelEncoder()
        self.label_encoders['player2'] = LabelEncoder()
        
        # Get all unique players
        all_players = list(set(df['player1'].tolist() + df['player2'].tolist()))
        self.label_encoders['player1'].fit(all_players)
        self.label_encoders['player2'].fit(all_players)
        
        df['player1_encoded'] = self.label_encoders['player1'].transform(df['player1'])
        df['player2_encoded'] = self.label_encoders['player2'].transform(df['player2'])
        features.extend(['player1_encoded', 'player2_encoded'])
        
        self.feature_columns = features
        
        # Prepare training data
        X = df[features]
        y = df['player1_wins']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        print("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f"Model trained! Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
        
        # Store player stats for frontend
        self.player_stats = {
            player: {
                'rank': int(df[df['player1'] == player]['player1_rank'].iloc[0]) if not df[df['player1'] == player].empty 
                       else int(df[df['player2'] == player]['player2_rank'].iloc[0]),
                'matches_played': len(df[(df['player1'] == player) | (df['player2'] == player)]),
                'wins': len(df[((df['player1'] == player) & (df['player1_wins'] == 1)) | 
                             ((df['player2'] == player) & (df['player1_wins'] == 0))])
            }
            for player in all_players
        }
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'players_count': len(all_players),
            'matches_count': len(df)
        }
    
    def predict_match(self, player1: str, player2: str, surface: str = 'Grass'):
        """Predict match outcome"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        try:
            # Get player stats
            p1_rank = self.player_stats.get(player1, {}).get('rank', 50)
            p2_rank = self.player_stats.get(player2, {}).get('rank', 50)
            rank_diff = p1_rank - p2_rank
            
            # Simulate head-to-head and recent form
            h2h_key = tuple(sorted([player1, player2]))
            h2h_p1 = hash(h2h_key + (player1,)) % 10
            h2h_p2 = hash(h2h_key + (player2,)) % 10
            recent_form_p1 = 6  # Default recent form
            recent_form_p2 = 6
            
            # Encode inputs
            player1_encoded = self.label_encoders['player1'].transform([player1])[0]
            player2_encoded = self.label_encoders['player2'].transform([player2])[0]
            surface_encoded = self.label_encoders['surface'].transform([surface])[0]
            
            # Create feature vector
            features = [
                p1_rank, p2_rank, rank_diff, h2h_p1, h2h_p2, 
                recent_form_p1, recent_form_p2, surface_encoded, 
                player1_encoded, player2_encoded
            ]
            
            # Make prediction
            X = np.array(features).reshape(1, -1)
            prob = self.model.predict_proba(X)[0]
            
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
                'head_to_head': f"{h2h_p1}-{h2h_p2}"
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return default prediction if encoding fails
            return {
                'player1': player1,
                'player2': player2,
                'surface': surface,
                'player1_win_probability': 0.5,
                'player2_win_probability': 0.5,
                'predicted_winner': player1,
                'confidence': 0.5,
                'player1_rank': 50,
                'player2_rank': 50,
                'head_to_head': "Unknown"
            }

# Initialize predictor
predictor = TennisPredictor()

@app.on_event("startup")
async def startup_event():
    """Train model on startup"""
    global predictor
    try:
        training_results = predictor.train_model()
        print(f"Model training completed: {training_results}")
    except Exception as e:
        print(f"Error during model training: {e}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_trained": predictor.model is not None}

@app.get("/api/players")
async def get_players():
    """Get list of available players"""
    if not predictor.player_stats:
        return {"players": []}
    
    players_list = [
        {
            "name": name,
            "rank": stats["rank"],
            "matches_played": stats["matches_played"],
            "wins": stats["wins"],
            "win_rate": round(stats["wins"] / max(stats["matches_played"], 1) * 100, 1)
        }
        for name, stats in predictor.player_stats.items()
    ]
    
    # Sort by rank
    players_list.sort(key=lambda x: x["rank"])
    
    return {"players": players_list}

@app.post("/api/predict")
async def predict_match(request: dict):
    """Predict tennis match outcome"""
    try:
        player1 = request.get("player1")
        player2 = request.get("player2")
        surface = request.get("surface", "Grass")
        
        if not player1 or not player2:
            raise HTTPException(status_code=400, detail="Both players are required")
            
        if player1 == player2:
            raise HTTPException(status_code=400, detail="Players must be different")
        
        prediction = predictor.predict_match(player1, player2, surface)
        
        # Store prediction in database
        prediction_record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow(),
            "prediction": prediction
        }
        
        db.predictions.insert_one(prediction_record)
        
        return prediction
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/history")
async def get_prediction_history(limit: int = 10):
    """Get recent predictions"""
    try:
        predictions = list(db.predictions.find(
            {}, 
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        
        return {"predictions": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/wimbledon/predictions")
async def get_wimbledon_predictions():
    """Get Wimbledon 2025 predictions for top matchups"""
    try:
        top_players = [
            'Novak Djokovic', 'Carlos Alcaraz', 'Daniil Medvedev', 'Jannik Sinner',
            'Andrey Rublev', 'Stefanos Tsitsipas', 'Alexander Zverev', 'Holger Rune'
        ]
        
        wimbledon_matches = [
            ('Novak Djokovic', 'Carlos Alcaraz'),
            ('Jannik Sinner', 'Daniil Medvedev'),
            ('Alexander Zverev', 'Stefanos Tsitsipas'),
            ('Holger Rune', 'Andrey Rublev')
        ]
        
        predictions = []
        for player1, player2 in wimbledon_matches:
            pred = predictor.predict_match(player1, player2, 'Grass')
            predictions.append(pred)
        
        return {
            "tournament": "Wimbledon 2025",
            "surface": "Grass",
            "predictions": predictions,
            "generated_at": datetime.utcnow().isoformat()
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
            "model_type": "XGBoost Classifier",
            "feature_count": len(predictor.feature_columns),
            "feature_importance": feature_importance,
            "players_count": len(predictor.player_stats)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)