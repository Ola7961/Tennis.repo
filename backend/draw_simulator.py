import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

class DrawSimulator:
    def __init__(self, model, players_data):
        self.model = model
        self.players_data = players_data
        
    def predict_match_score(self, player1: str, player2: str, surface: str = "Grass") -> Tuple[str, str, float]:
        """
        Predict match winner and generate realistic tennis score
        Returns: (winner, score, confidence)
        """
        try:
            # Get prediction from the model
            prediction = self.model.predict_match(player1, player2, surface)
            winner = prediction['predicted_winner']
            confidence = prediction['confidence']
            
            # Generate realistic tennis score based on confidence
            score = self._generate_tennis_score(confidence, winner == player1)
            
            return winner, score, confidence
            
        except Exception as e:
            print(f"Error predicting match {player1} vs {player2}: {e}")
            # Fallback to random prediction
            winner = random.choice([player1, player2])
            score = self._generate_tennis_score(0.6, winner == player1)
            return winner, score, 0.6
    
    def _generate_tennis_score(self, confidence: float, player1_wins: bool) -> str:
        """
        Generate realistic tennis scores based on match confidence
        Higher confidence = more dominant score
        """
        # Define score patterns based on confidence levels
        if confidence > 0.8:  # Very confident prediction
            if player1_wins:
                scores = ["6-3 6-2", "6-4 6-1", "6-2 6-3", "6-1 6-4", "6-0 6-3"]
            else:
                scores = ["3-6 2-6", "4-6 1-6", "2-6 3-6", "1-6 4-6", "0-6 3-6"]
        elif confidence > 0.7:  # High confidence
            if player1_wins:
                scores = ["6-4 6-3", "6-3 6-4", "7-5 6-4", "6-2 7-5", "6-4 7-6"]
            else:
                scores = ["4-6 3-6", "3-6 4-6", "5-7 4-6", "2-6 5-7", "4-6 6-7"]
        elif confidence > 0.6:  # Medium confidence
            if player1_wins:
                scores = ["7-6 6-4", "6-4 7-6", "7-5 7-5", "6-3 7-6", "7-6 7-5"]
            else:
                scores = ["6-7 4-6", "4-6 6-7", "5-7 5-7", "3-6 6-7", "6-7 5-7"]
        else:  # Low confidence - close match
            if player1_wins:
                scores = ["7-6 7-6", "6-7 7-6 6-4", "7-5 6-7 6-3", "6-4 6-7 7-5", "7-6 6-7 7-6"]
            else:
                scores = ["6-7 6-7", "7-6 6-7 4-6", "5-7 7-6 3-6", "4-6 7-6 5-7", "6-7 7-6 6-7"]
        
        return random.choice(scores)
    
    def simulate_tournament_round(self, matches: List[Tuple[str, str]], surface: str = "Grass") -> List[Dict]:
        """
        Simulate a complete round of matches
        Returns list of match results with winners and scores
        """
        results = []
        
        for player1, player2 in matches:
            winner, score, confidence = self.predict_match_score(player1, player2, surface)
            
            results.append({
                'player1': player1,
                'player2': player2,
                'winner': winner,
                'score': score,
                'confidence': confidence,
                'surface': surface
            })
        
        return results
    
    def simulate_full_tournament(self, initial_draw: Dict, surface: str = "Grass") -> Dict:
        """
        Simulate the entire tournament from current round to final
        Returns complete tournament results with all match predictions
        """
        tournament_results = {
            'rounds': {},
            'final_winner': None,
            'tournament_path': []
        }
        
        # Start with current round players
        current_players = initial_draw.get('remaining_players', [])
        round_number = 1
        round_names = {
            1: "Third Round",
            2: "Fourth Round", 
            3: "Quarter Finals",
            4: "Semi Finals",
            5: "Final"
        }
        
        while len(current_players) > 1:
            round_name = round_names.get(round_number, f"Round {round_number}")
            
            # Create matches for this round
            matches = []
            for i in range(0, len(current_players), 2):
                if i + 1 < len(current_players):
                    matches.append((current_players[i], current_players[i + 1]))
            
            # Simulate round
            round_results = self.simulate_tournament_round(matches, surface)
            tournament_results['rounds'][round_name] = round_results
            
            # Get winners for next round
            current_players = [result['winner'] for result in round_results]
            round_number += 1
            
            # Stop if we have a winner
            if len(current_players) == 1:
                tournament_results['final_winner'] = current_players[0]
                break
        
        return tournament_results
    
    def get_player_tournament_path(self, player_name: str, tournament_results: Dict) -> List[Dict]:
        """
        Get the predicted path of a specific player through the tournament
        """
        path = []
        
        for round_name, matches in tournament_results['rounds'].items():
            for match in matches:
                if player_name in [match['player1'], match['player2']]:
                    path.append({
                        'round': round_name,
                        'opponent': match['player2'] if match['player1'] == player_name else match['player1'],
                        'result': 'Win' if match['winner'] == player_name else 'Loss',
                        'score': match['score'],
                        'confidence': match['confidence']
                    })
                    
                    # If player lost, stop tracking their path
                    if match['winner'] != player_name:
                        break
        
        return path
    
    def calculate_round_probabilities(self, players: List[str], surface: str = "Grass") -> Dict:
        """
        Calculate probability of each player reaching different rounds
        """
        probabilities = {}
        
        for player in players:
            probabilities[player] = {
                'current_round': 1.0,  # Already in current round
                'next_round': 0.0,
                'quarter_final': 0.0,
                'semi_final': 0.0,
                'final': 0.0,
                'winner': 0.0
            }
        
        # Run multiple simulations to calculate probabilities
        num_simulations = 1000
        
        for _ in range(num_simulations):
            # Create a copy of players for this simulation
            sim_players = players.copy()
            round_stage = 'next_round'
            
            while len(sim_players) > 1:
                # Create matches
                matches = []
                for i in range(0, len(sim_players), 2):
                    if i + 1 < len(sim_players):
                        matches.append((sim_players[i], sim_players[i + 1]))
                
                # Simulate matches and get winners
                winners = []
                for player1, player2 in matches:
                    winner, _, _ = self.predict_match_score(player1, player2, surface)
                    winners.append(winner)
                
                # Update probabilities for players who advanced
                for winner in winners:
                    if winner in probabilities:
                        probabilities[winner][round_stage] += 1.0 / num_simulations
                
                # Move to next round
                sim_players = winners
                if round_stage == 'next_round':
                    round_stage = 'quarter_final'
                elif round_stage == 'quarter_final':
                    round_stage = 'semi_final'
                elif round_stage == 'semi_final':
                    round_stage = 'final'
                elif round_stage == 'final':
                    round_stage = 'winner'
        
        return probabilities

def create_wimbledon_draw_structure():
    """
    Create the official Wimbledon 2025 draw structure
    """
    return {
        "tournament": "Wimbledon 2025",
        "category": "Gentlemen's Singles",
        "current_round": "Third Round",
        "total_players": 128,
        "remaining_players": [
            # Top Half
            "Jannik Sinner", "Miomir Kecmanovic", "Matteo Berrettini", "Pavel Kotov",
            "Ben Shelton", "Denis Shapovalov", "Stefanos Tsitsipas", "Emil Ruusuvuori",
            "Casper Ruud", "Fabio Fognini", "Ugo Humbert", "Brandon Nakashima",
            "Holger Rune", "Quentin Halys", "Alexander Zverev", "Cameron Norrie",
            
            # Bottom Half  
            "Carlos Alcaraz", "Frances Tiafoe", "Sebastian Korda", "Alejandro Tabilo",
            "Tommy Paul", "Roberto Bautista Agut", "Lorenzo Musetti", "Giovanni Mpetshi Perricard",
            "Alex de Minaur", "Arthur Fils", "Sebastian Baez", "Jiri Lehecka",
            "Novak Djokovic", "Alexei Popyrin", "Lorenzo Sonego", "Daniil Medvedev"
        ],
        "sections": {
            "Top Section": {
                "top_seed": 1,
                "seed_name": "Jannik Sinner",
                "key_players": ["Jannik Sinner", "Matteo Berrettini", "Ben Shelton", "Stefanos Tsitsipas"]
            },
            "Second Section": {
                "top_seed": 8,
                "seed_name": "Casper Ruud", 
                "key_players": ["Casper Ruud", "Ugo Humbert", "Holger Rune", "Alexander Zverev"]
            },
            "Third Section": {
                "top_seed": 3,
                "seed_name": "Carlos Alcaraz",
                "key_players": ["Carlos Alcaraz", "Tommy Paul", "Lorenzo Musetti", "Alex de Minaur"]
            },
            "Bottom Section": {
                "top_seed": 2,
                "seed_name": "Novak Djokovic",
                "key_players": ["Novak Djokovic", "Sebastian Baez", "Lorenzo Sonego", "Daniil Medvedev"]
            }
        }
    }

