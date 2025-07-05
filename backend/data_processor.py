import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

class TennisDataProcessor:
    def __init__(self):
        self.data = None
        self.players_data = {}
        self.label_encoders = {}
        self.all_players_list = []
        
    def load_atp_data(self, data_dir):
        """Load ATP match data from multiple CSV files in a directory"""
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("atp_matches_") and f.endswith(".csv")]
        
        if not all_files:
            print(f"No ATP match data files found in {data_dir}")
            return False
            
        df_list = []
        for f in all_files:
            try:
                df = pd.read_csv(f)
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        if not df_list:
            print("No valid ATP match data could be loaded.")
            return False
            
        self.data = pd.concat(df_list, ignore_index=True)
        print(f"Loaded {len(self.data)} matches from all files")
        
        # Clean and process the data
        self.process_data()
        return True
    
    def process_data(self):
        """Process and clean the ATP data"""
        if self.data is None:
            return
            
        # Remove matches with missing essential data
        self.data = self.data.dropna(subset=[
            'winner_name', 'loser_name', 'surface', 
            'winner_rank', 'loser_rank', 'winner_id', 'loser_id'
        ])
        
        # Convert tourney_date to datetime
        self.data['tourney_date'] = pd.to_datetime(self.data['tourney_date'], format='%Y%m%d')
        
        # Create player statistics
        self.create_player_stats()
        
        # Create features for machine learning
        self.create_features()
        
    def create_player_stats(self):
        """Create comprehensive player statistics"""
        all_players_names = set(self.data['winner_name'].tolist() + self.data['loser_name'].tolist())
        self.all_players_list = sorted(list(all_players_names))
        
        for player in self.all_players_list:
            # Get all matches for this player
            player_matches = self.data[
                (self.data['winner_name'] == player) | (self.data['loser_name'] == player)
            ].copy()
            
            if len(player_matches) == 0:
                continue
                
            # Calculate statistics
            wins = len(player_matches[player_matches['winner_name'] == player])
            losses = len(player_matches[player_matches['loser_name'] == player])
            total_matches = wins + losses
            
            # Get latest ranking (most recent match)
            latest_match = player_matches.sort_values('tourney_date').iloc[-1]
            if latest_match['winner_name'] == player:
                current_rank = latest_match.get('winner_rank', 999)
                player_id = latest_match.get('winner_id')
            else:
                current_rank = latest_match.get('loser_rank', 999)
                player_id = latest_match.get('loser_id')
                
            # Surface-specific stats
            surface_stats = {}
            for surface in ['Hard', 'Clay', 'Grass']:
                surface_matches = player_matches[player_matches['surface'] == surface]
                surface_wins = len(surface_matches[surface_matches['winner_name'] == player])
                surface_total = len(surface_matches)
                surface_stats[surface] = {
                    'wins': surface_wins,
                    'matches': surface_total,
                    'win_rate': surface_wins / max(surface_total, 1) if surface_total > 0 else 0.5
                }
            
            # Recent form (last 10 matches)
            recent_matches = player_matches.sort_values('tourney_date').tail(10)
            recent_wins = len(recent_matches[recent_matches['winner_name'] == player])
            
            self.players_data[player] = {
                'id': player_id,
                'name': player,
                'total_wins': wins,
                'total_losses': losses,
                'total_matches': total_matches,
                'win_rate': wins / max(total_matches, 1) if total_matches > 0 else 0.5,
                'current_rank': int(current_rank) if pd.notna(current_rank) else 999,
                'surface_stats': surface_stats,
                'recent_form': recent_wins,
                'last_match_date': latest_match['tourney_date'].strftime('%Y-%m-%d')
            }
    
    def create_features(self):
        """Create features for machine learning"""
        features_list = []
        
        for _, match in self.data.iterrows():
            winner = match['winner_name']
            loser = match['loser_name']
            surface = match['surface']
            
            # Get player stats
            winner_stats = self.players_data.get(winner, {})
            loser_stats = self.players_data.get(loser, {})
            
            # Ensure players have valid stats before creating features
            if not winner_stats or not loser_stats:
                continue

            # Create feature vector for winner as player1
            features = {
                'player1': winner,
                'player2': loser,
                'player1_rank': winner_stats.get('current_rank', 999),
                'player2_rank': loser_stats.get('current_rank', 999),
                'surface': surface,
                'rank_difference': winner_stats.get('current_rank', 999) - loser_stats.get('current_rank', 999),
                'player1_win_rate': winner_stats.get('win_rate', 0.5),
                'player2_win_rate': loser_stats.get('win_rate', 0.5),
                'player1_surface_win_rate': winner_stats.get('surface_stats', {}).get(surface, {}).get('win_rate', 0.5),
                'player2_surface_win_rate': loser_stats.get('surface_stats', {}).get(surface, {}).get('win_rate', 0.5),
                'player1_recent_form': winner_stats.get('recent_form', 5),
                'player2_recent_form': loser_stats.get('recent_form', 5),
                'player1_wins': 1  # Winner is player1 in this case
            }
            
            features_list.append(features)
            
            # Also add the reverse (loser as player1) for balanced training
            features_reverse = {
                'player1': loser,
                'player2': winner,
                'player1_rank': loser_stats.get('current_rank', 999),
                'player2_rank': winner_stats.get('current_rank', 999),
                'surface': surface,
                'rank_difference': loser_stats.get('current_rank', 999) - winner_stats.get('current_rank', 999),
                'player1_win_rate': loser_stats.get('win_rate', 0.5),
                'player2_win_rate': winner_stats.get('win_rate', 0.5),
                'player1_surface_win_rate': loser_stats.get('surface_stats', {}).get(surface, {}).get('win_rate', 0.5),
                'player2_surface_win_rate': winner_stats.get('surface_stats', {}).get(surface, {}).get('win_rate', 0.5),
                'player1_recent_form': loser_stats.get('recent_form', 5),
                'player2_recent_form': winner_stats.get('recent_form', 5),
                'player1_wins': 0  # Loser is player1 in this case
            }
            
            features_list.append(features_reverse)
        
        self.features_df = pd.DataFrame(features_list)
        print(f"Created {len(self.features_df)} training examples from {len(self.data)} matches")
    
    def get_head_to_head(self, player1, player2):
        """Get head-to-head record between two players"""
        if self.data is None:
            return {'player1_wins': 0, 'player2_wins': 0, 'total_matches': 0}
            
        # Find all matches between these players
        h2h_matches = self.data[
            ((self.data['winner_name'] == player1) & (self.data['loser_name'] == player2)) |
            ((self.data['winner_name'] == player2) & (self.data['loser_name'] == player1))
        ]
        
        player1_wins = len(h2h_matches[h2h_matches['winner_name'] == player1])
        player2_wins = len(h2h_matches[h2h_matches['winner_name'] == player2])
        
        return {
            'player1_wins': player1_wins,
            'player2_wins': player2_wins,
            'total_matches': len(h2h_matches)
        }
    
    def get_top_players(self, limit=50):
        """Get top players by ranking"""
        if not self.players_data:
            return []
            
        # Sort players by ranking (lower number = better rank)
        sorted_players = sorted(
            self.players_data.values(),
            key=lambda x: x['current_rank']
        )
        
        return sorted_players[:limit]
    
    def get_player_info(self, player_name):
        """Get detailed information about a specific player"""
        return self.players_data.get(player_name, {})
    
    def get_all_player_names(self):
        """Get a list of all player names"""
        return self.all_players_list

    def save_processed_data(self, output_dir):
        """Save processed data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save player statistics
        players_df = pd.DataFrame(self.players_data.values())
        players_df.to_csv(os.path.join(output_dir, 'players_stats.csv'), index=False)
        
        # Save features for ML
        if hasattr(self, 'features_df'):
            self.features_df.to_csv(os.path.join(output_dir, 'ml_features.csv'), index=False)
        
        print(f"Saved processed data to {output_dir}")

if __name__ == "__main__":
    # Test the data processor
    processor = TennisDataProcessor()
    
    # Load the ATP data
    data_dir = "/home/ubuntu/Tennis_repo/data"
    if processor.load_atp_data(data_dir):
        print("Data loaded successfully!")
        
        # Show some statistics
        top_players = processor.get_top_players(10)
        print("\nTop 10 players by ranking:")
        for i, player in enumerate(top_players, 1):
            print(f"{i}. {player['name']} (Rank: {player['current_rank']}, Win Rate: {player['win_rate']:.3f})")
        
        # Save processed data
        processor.save_processed_data("/home/ubuntu/Tennis_repo/data/processed")
    else:
        print("Failed to load data")


