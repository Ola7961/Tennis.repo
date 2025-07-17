from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import uvicorn
import os
import glob
from typing import Dict, List, Optional

app = FastAPI(title="Tennis Predictor API", version="2.0.0")
@app.get("/")
def root():
    return {"status": "Backend is live 🎾"}


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None

class PredictionRequest(BaseModel):
    player1: str
    player2: str
    surface: str = "Grass"

class TennisPredictor:
    def __init__(self):
        self.xgb_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        self.players_stats = {}
        self.feature_names = [
            'player1_rank', 'player2_rank', 'player1_win_rate', 'player2_win_rate',
            'player1_surface_win_rate', 'player2_surface_win_rate', 'player1_recent_form',
            'player2_recent_form', 'rank_diff', 'win_rate_diff', 'surface_win_rate_diff',
            'h2h_advantage'
        ]
    
    def load_and_process_data(self):
        print("Loading all ATP men's singles data from 1968-2024...")
        
        # Load ATP singles match data from 1968-2024 for player stats and model training
        data_files = []
        for year in range(1968, 2025):
           data_files.append(f'data/atp_matches_{year}.csv')


        all_matches = []
        
        for file in sorted(data_files):
            try:
                df = pd.read_csv(file)
                all_matches.append(df)
                print(f"Loaded {len(df)} matches from {os.path.basename(file)}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        if not all_matches:
            raise Exception("No match data files found for 1968-2024 men's singles")
        
        # Combine all data
        df = pd.concat(all_matches, ignore_index=True)
        print(f"Loaded {len(df)} matches from selected files")
        
        # Clean and process data
        df = df.dropna(subset=['winner_name', 'loser_name', 'surface'])
        
        # Calculate player statistics
        self._calculate_player_stats(df)
        
        # Create features for training
        features, labels = self._create_features(df)
        
        return features, labels
    
    def _calculate_player_stats(self, df):
        print("Calculating player statistics...")
        
        # Get all unique players
        all_players = set(df['winner_name'].unique()) | set(df['loser_name'].unique())
        
        for player in all_players:
            # Get all matches for this player
            player_matches = df[(df['winner_name'] == player) | (df['loser_name'] == player)]
            
            if len(player_matches) == 0:
                continue
            
            # Calculate basic stats
            wins = len(df[df['winner_name'] == player])
            total_matches = len(player_matches)
            win_rate = (wins / total_matches * 100) if total_matches > 0 else 0
            
            # Calculate surface-specific stats
            grass_matches = player_matches[player_matches['surface'] == 'Grass']
            grass_wins = len(grass_matches[grass_matches['winner_name'] == player])
            grass_win_rate = (grass_wins / len(grass_matches) * 100) if len(grass_matches) > 0 else win_rate
            
            hard_matches = player_matches[player_matches['surface'] == 'Hard']
            hard_wins = len(hard_matches[hard_matches['winner_name'] == player])
            hard_win_rate = (hard_wins / len(hard_matches) * 100) if len(hard_matches) > 0 else win_rate
            
            clay_matches = player_matches[player_matches['surface'] == 'Clay']
            clay_wins = len(clay_matches[clay_matches['winner_name'] == player])
            clay_win_rate = (clay_wins / len(clay_matches) * 100) if len(clay_matches) > 0 else win_rate
            
            # Calculate recent form (last 10 matches)
            recent_matches = player_matches.tail(10)
            recent_wins = len(recent_matches[recent_matches['winner_name'] == player])
            recent_form = (recent_wins / len(recent_matches) * 100) if len(recent_matches) > 0 else win_rate
            
            # Estimate ranking (inverse of win rate with some randomness)
            estimated_rank = max(1, int(101 - win_rate + np.random.normal(0, 10)))
            
            self.players_stats[player] = {
                'name': player,
                'total_matches': total_matches,
                'wins': wins,
                'win_rate': round(win_rate, 1),
                'grass_win_rate': round(grass_win_rate, 1),
                'hard_win_rate': round(hard_win_rate, 1),
                'clay_win_rate': round(clay_win_rate, 1),
                'recent_form': round(recent_form, 1),
                'rank': estimated_rank
            }
        
        print(f"Calculated stats for {len(self.players_stats)} players")
    
    def _create_features(self, df):
        print("Creating features for training...")
        
        features = []
        labels = []
        
        for _, match in df.iterrows():
            winner = match['winner_name']
            loser = match['loser_name']
            surface = match['surface']
            
            if winner not in self.players_stats or loser not in self.players_stats:
                continue
            
            # Create features for winner vs loser (label = 1)
            feature_vector = self._get_match_features(winner, loser, surface)
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(1)
            
            # Create features for loser vs winner (label = 0)
            feature_vector = self._get_match_features(loser, winner, surface)
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(0)
        
        print(f"Created {len(features)} feature vectors")
        return np.array(features), np.array(labels)
    
    def _get_match_features(self, player1, player2, surface):
        if player1 not in self.players_stats or player2 not in self.players_stats:
            return None
        
        p1_stats = self.players_stats[player1]
        p2_stats = self.players_stats[player2]
        
        # Surface-specific win rates
        surface_rates = {
            'Grass': ('grass_win_rate', 'grass_win_rate'),
            'Hard': ('hard_win_rate', 'hard_win_rate'),
            'Clay': ('clay_win_rate', 'clay_win_rate')
        }
        
        p1_surface_rate = p1_stats.get(surface_rates.get(surface, ('win_rate', 'win_rate'))[0], p1_stats['win_rate'])
        p2_surface_rate = p2_stats.get(surface_rates.get(surface, ('win_rate', 'win_rate'))[1], p2_stats['win_rate'])
        
        # Calculate head-to-head (simplified)
        h2h_advantage = 0  # Could be enhanced with actual H2H data
        
        features = [
            p1_stats['rank'],
            p2_stats['rank'],
            p1_stats['win_rate'],
            p2_stats['win_rate'],
            p1_surface_rate,
            p2_surface_rate,
            p1_stats['recent_form'],
            p2_stats['recent_form'],
            p1_stats['rank'] - p2_stats['rank'],  # rank_diff
            p1_stats['win_rate'] - p2_stats['win_rate'],  # win_rate_diff
            p1_surface_rate - p2_surface_rate,  # surface_win_rate_diff
            h2h_advantage
        ]
        
        return features
    
    def train_models(self, features, labels):
        print("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Scale features for neural network
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        print("Training XGBoost model...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
        
        # Train Neural Network
        print("Training Neural Network model...")
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.nn_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        xgb_train_acc = accuracy_score(y_train, self.xgb_model.predict(X_train))
        xgb_test_acc = accuracy_score(y_test, self.xgb_model.predict(X_test))
        
        nn_train_acc = accuracy_score(y_train, self.nn_model.predict(X_train_scaled))
        nn_test_acc = accuracy_score(y_test, self.nn_model.predict(X_test_scaled))
        
        print(f"XGBoost - Train: {xgb_train_acc:.3f}, Test: {xgb_test_acc:.3f}")
        print(f"Neural Network - Train: {nn_train_acc:.3f}, Test: {nn_test_acc:.3f}")
        
        return {
            'xgb_train_accuracy': xgb_train_acc,
            'xgb_test_accuracy': xgb_test_acc,
            'nn_train_accuracy': nn_train_acc,
            'nn_test_accuracy': nn_test_acc,
            'total_samples': len(features),
            'features_count': len(self.feature_names)
        }
    
    def predict_match(self, player1, player2, surface="Grass"):
        if not self.xgb_model or not self.nn_model:
            raise Exception("Models not trained")
        
        if player1 not in self.players_stats or player2 not in self.players_stats:
            raise Exception(f"Player data not found for {player1} or {player2}")
        
        # Get features
        features = self._get_match_features(player1, player2, surface)
        if features is None:
            raise Exception("Could not create features for prediction")
        
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from both models
        xgb_prob = self.xgb_model.predict_proba(features)[0]
        nn_prob = self.nn_model.predict_proba(features_scaled)[0]
        
        # Ensemble prediction (simple average)
        ensemble_prob = (xgb_prob + nn_prob) / 2
        
        player1_win_prob = ensemble_prob[1]
        player2_win_prob = ensemble_prob[0]
        
        predicted_winner = player1 if player1_win_prob > player2_win_prob else player2
        confidence = max(player1_win_prob, player2_win_prob)
        
        # Get player stats
        p1_stats = self.players_stats[player1]
        p2_stats = self.players_stats[player2]
        
        surface_rates = {
            'Grass': 'grass_win_rate',
            'Hard': 'hard_win_rate', 
            'Clay': 'clay_win_rate'
        }
        surface_key = surface_rates.get(surface, 'win_rate')
        
        return {
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'player1_win_probability': player1_win_prob,
            'player2_win_probability': player2_win_prob,
            'player1_rank': p1_stats['rank'],
            'player2_rank': p2_stats['rank'],
            'player1_win_rate': p1_stats['win_rate'],
            'player2_win_rate': p2_stats['win_rate'],
            'player1_surface_win_rate': p1_stats.get(surface_key, p1_stats['win_rate']),
            'player2_surface_win_rate': p2_stats.get(surface_key, p2_stats['win_rate']),
            'head_to_head': "No H2H data",
            'h2h_matches': 0
        }
    
    def get_players_list(self, limit=None):
        players = sorted(self.players_stats.values(), key=lambda x: x['rank'])
        if limit:
            players = players[:limit]
        return players
    
    def search_players(self, query, limit=10):
        query = query.lower()
        matching_players = []
        
        for player_name, stats in self.players_stats.items():
            if query in player_name.lower():
                matching_players.append({
                    'name': player_name,
                    'rank': stats['rank'],
                    'win_rate': stats['win_rate'],
                    'total_matches': stats['total_matches']
                })
        
        # Sort by rank and limit results
        matching_players.sort(key=lambda x: x['rank'])
        return matching_players[:limit]


model = TennisPredictor()

@app.on_event("startup")
async def startup_event():
    global model
    try:
        features, labels = model.load_and_process_data()
        model.train_models(features, labels)
    except Exception as e:
    import traceback
    print("❌ Exception in model startup:")
    traceback.print_exc()

        # Optionally, handle this error more gracefully, e.g., by logging and exiting

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    try:
        prediction_result = model.predict_match(request.player1, request.player2, request.surface)
        return prediction_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/players/search")
async def search_players(q: str):
    try:
        players = model.search_players(q)
        return {"players": players}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/info")
async def get_model_info():
    try:
        if not model.xgb_model or not model.nn_model:
            raise HTTPException(status_code=503, detail="Model not trained yet.")
        
        # Get feature importance from XGBoost
        feature_importances = model.xgb_model.feature_importances_
        feature_names = model.feature_names
        
        # Create a list of dictionaries for feature importance
        xgb_feature_importance = [
            {"feature": name, "importance": importance}
            for name, importance in zip(feature_names, feature_importances)
        ]
        
        # Sort by importance
        xgb_feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "players_count": len(model.players_stats),
            "feature_count": len(model.feature_names),
            "xgb_train_accuracy": model.xgb_train_accuracy,
            "xgb_test_accuracy": model.xgb_test_accuracy,
            "nn_train_accuracy": model.nn_train_accuracy,
            "nn_test_accuracy": model.nn_test_accuracy,
            "xgb_feature_importance": xgb_feature_importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/")
async def root():
    return {"message": "Backend running"}


