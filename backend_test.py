import requests
import unittest
import json
import sys
from datetime import datetime

class TennisMatchPredictorAPITest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TennisMatchPredictorAPITest, self).__init__(*args, **kwargs)
        self.base_url = "https://745174a3-bedf-4323-8bd6-2b49c69156a8.preview.emergentagent.com"
        self.tests_run = 0
        self.tests_passed = 0

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_01_health_check(self):
        """Test health check endpoint"""
        print("\n🔍 Testing health check endpoint...")
        
        response = requests.get(f"{self.base_url}/api/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_trained", data)
        self.assertTrue(data["model_trained"])
        
        print("✅ Health check endpoint is working")

    def test_02_get_players(self):
        """Test get players endpoint"""
        print("\n🔍 Testing get players endpoint...")
        
        response = requests.get(f"{self.base_url}/api/players")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("players", data)
        self.assertIsInstance(data["players"], list)
        
        # Check if we have players
        self.assertGreater(len(data["players"]), 0)
        
        # Check player data structure
        player = data["players"][0]
        self.assertIn("name", player)
        self.assertIn("rank", player)
        self.assertIn("matches_played", player)
        self.assertIn("wins", player)
        self.assertIn("win_rate", player)
        
        # Check for specific players
        player_names = [p["name"] for p in data["players"]]
        expected_players = ["Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner"]
        for expected_player in expected_players:
            self.assertIn(expected_player, player_names, f"Expected player {expected_player} not found")
        
        print(f"✅ Get players endpoint returned {len(data['players'])} players")
        return data["players"]

    def test_03_predict_match(self):
        """Test predict match endpoint"""
        print("\n🔍 Testing predict match endpoint...")
        
        # Get players first
        response = requests.get(f"{self.base_url}/api/players")
        self.assertEqual(response.status_code, 200)
        players = response.json()["players"]
        
        player1 = players[0]["name"]
        player2 = players[1]["name"]
        
        # Test with different surface types
        surfaces = ["Grass", "Hard", "Clay"]
        
        for surface in surfaces:
            print(f"  Testing prediction for {player1} vs {player2} on {surface}...")
            
            payload = {
                "player1": player1,
                "player2": player2,
                "surface": surface
            }
            
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=payload
            )
            
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn("player1", data)
            self.assertIn("player2", data)
            self.assertIn("surface", data)
            self.assertIn("player1_win_probability", data)
            self.assertIn("player2_win_probability", data)
            self.assertIn("predicted_winner", data)
            self.assertIn("confidence", data)
            self.assertIn("player1_rank", data)
            self.assertIn("player2_rank", data)
            self.assertIn("head_to_head", data)
            
            # Check probabilities sum to approximately 1
            total_prob = data["player1_win_probability"] + data["player2_win_probability"]
            self.assertAlmostEqual(total_prob, 1.0, places=1)
            
            # Check predicted winner matches highest probability
            if data["player1_win_probability"] > data["player2_win_probability"]:
                self.assertEqual(data["predicted_winner"], player1)
            else:
                self.assertEqual(data["predicted_winner"], player2)
                
            print(f"  ✅ Prediction for {surface} surface: {data['predicted_winner']} with {data['confidence']:.2f} confidence")
        
        # Test error case - same player
        payload = {
            "player1": player1,
            "player2": player1,
            "surface": "Grass"
        }
        
        response = requests.post(
            f"{self.base_url}/api/predict",
            json=payload
        )
        
        self.assertIn(response.status_code, [400, 500])  # Accept either 400 or 500
        print("  ✅ Error handling for same player works correctly")
        
        # Test error case - missing player
        payload = {
            "player1": player1,
            "surface": "Grass"
        }
        
        response = requests.post(
            f"{self.base_url}/api/predict",
            json=payload
        )
        
        self.assertIn(response.status_code, [400, 500])  # Accept either 400 or 500
        print("  ✅ Error handling for missing player works correctly")
        
        print("✅ Predict match endpoint is working")

    def test_04_wimbledon_predictions(self):
        """Test Wimbledon predictions endpoint"""
        print("\n🔍 Testing Wimbledon predictions endpoint...")
        
        response = requests.get(f"{self.base_url}/api/wimbledon/predictions")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("tournament", data)
        self.assertEqual(data["tournament"], "Wimbledon 2025")
        self.assertIn("surface", data)
        self.assertEqual(data["surface"], "Grass")
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], list)
        self.assertGreater(len(data["predictions"]), 0)
        self.assertIn("generated_at", data)
        
        # Check prediction structure
        prediction = data["predictions"][0]
        self.assertIn("player1", prediction)
        self.assertIn("player2", prediction)
        self.assertIn("surface", prediction)
        self.assertEqual(prediction["surface"], "Grass")
        self.assertIn("player1_win_probability", prediction)
        self.assertIn("player2_win_probability", prediction)
        self.assertIn("predicted_winner", prediction)
        self.assertIn("confidence", prediction)
        
        # Check for specific matchups
        expected_matchups = [
            ("Novak Djokovic", "Carlos Alcaraz"),
            ("Jannik Sinner", "Daniil Medvedev")
        ]
        
        found_matchups = []
        for pred in data["predictions"]:
            found_matchups.append((pred["player1"], pred["player2"]))
            found_matchups.append((pred["player2"], pred["player1"]))
        
        for player1, player2 in expected_matchups:
            self.assertTrue(
                (player1, player2) in found_matchups or (player2, player1) in found_matchups,
                f"Expected matchup {player1} vs {player2} not found"
            )
        
        print(f"✅ Wimbledon predictions endpoint returned {len(data['predictions'])} predictions")

    def test_05_model_info(self):
        """Test model info endpoint"""
        print("\n🔍 Testing model info endpoint...")
        
        response = requests.get(f"{self.base_url}/api/model/info")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("model_type", data)
        self.assertEqual(data["model_type"], "XGBoost Classifier")
        self.assertIn("feature_count", data)
        self.assertIn("feature_importance", data)
        self.assertIsInstance(data["feature_importance"], list)
        self.assertGreater(len(data["feature_importance"]), 0)
        self.assertIn("players_count", data)
        
        # Check feature importance structure
        feature = data["feature_importance"][0]
        self.assertIn("feature", feature)
        self.assertIn("importance", feature)
        
        # Check if feature importance values are valid
        total_importance = sum(f["importance"] for f in data["feature_importance"])
        self.assertAlmostEqual(total_importance, 1.0, places=1)
        
        print(f"✅ Model info endpoint returned {len(data['feature_importance'])} features")
        print(f"  Top features: {', '.join(f['feature'] for f in data['feature_importance'][:3])}")

    def run_all_tests(self):
        """Run all tests and print summary"""
        test_methods = [method for method in dir(self) if method.startswith('test_')]
        test_methods.sort()  # Ensure tests run in order
        
        print(f"\n🧪 Running {len(test_methods)} API tests for Tennis Match Predictor...")
        
        self.tests_run = len(test_methods)
        self.tests_passed = 0
        
        for method in test_methods:
            try:
                getattr(self, method)()
                self.tests_passed += 1
            except Exception as e:
                print(f"❌ Test {method} failed: {str(e)}")
        
        print(f"\n📊 Tests passed: {self.tests_passed}/{self.tests_run}")
        return self.tests_passed == self.tests_run

def main():
    tester = TennisMatchPredictorAPITest()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())