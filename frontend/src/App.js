import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [players, setPlayers] = useState([]);
  const [selectedPlayer1, setSelectedPlayer1] = useState('');
  const [selectedPlayer2, setSelectedPlayer2] = useState('');
  const [surface, setSurface] = useState('Grass');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [wimbledonPredictions, setWimbledonPredictions] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [activeTab, setActiveTab] = useState('predict');

  useEffect(() => {
    fetchPlayers();
    fetchWimbledonPredictions();
    fetchModelInfo();
  }, []);

  const fetchPlayers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/players`);
      const data = await response.json();
      setPlayers(data.players || []);
    } catch (error) {
      console.error('Error fetching players:', error);
    }
  };

  const fetchWimbledonPredictions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/wimbledon/predictions`);
      const data = await response.json();
      setWimbledonPredictions(data);
    } catch (error) {
      console.error('Error fetching Wimbledon predictions:', error);
    }
  };

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/model/info`);
      const data = await response.json();
      setModelInfo(data);
    } catch (error) {
      console.error('Error fetching model info:', error);
    }
  };

  const handlePredict = async () => {
    if (!selectedPlayer1 || !selectedPlayer2) {
      alert('Please select both players');
      return;
    }

    if (selectedPlayer1 === selectedPlayer2) {
      alert('Please select different players');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          player1: selectedPlayer1,
          player2: selectedPlayer2,
          surface: surface
        })
      });

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error making prediction:', error);
      alert('Error making prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const renderPredictionCard = (pred, index) => (
    <div key={index} className="bg-white rounded-xl shadow-lg p-6 mb-4 border border-gray-200">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold text-gray-800">
          {pred.player1} vs {pred.player2}
        </h3>
        <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
          {pred.surface}
        </span>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="text-center">
          <div className="text-lg font-semibold text-gray-700">{pred.player1}</div>
          <div className="text-sm text-gray-500">Rank #{pred.player1_rank}</div>
          <div className="text-2xl font-bold text-blue-600 mt-2">
            {Math.round(pred.player1_win_probability * 100)}%
          </div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-gray-700">{pred.player2}</div>
          <div className="text-sm text-gray-500">Rank #{pred.player2_rank}</div>
          <div className="text-2xl font-bold text-blue-600 mt-2">
            {Math.round(pred.player2_win_probability * 100)}%
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center pt-4 border-t border-gray-200">
        <div>
          <span className="text-sm text-gray-600">Predicted Winner: </span>
          <span className="font-bold text-green-600">{pred.predicted_winner}</span>
        </div>
        <div>
          <span className="text-sm text-gray-600">Confidence: </span>
          <span className={`font-bold ${getConfidenceColor(pred.confidence)}`}>
            {Math.round(pred.confidence * 100)}%
          </span>
        </div>
      </div>
      
      {pred.head_to_head && pred.head_to_head !== "Unknown" && (
        <div className="mt-2 text-sm text-gray-600">
          Head-to-head: {pred.head_to_head}
        </div>
      )}
    </div>
  );

  const renderPlayerRankings = () => (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
        <h2 className="text-2xl font-bold">ATP Player Rankings</h2>
        <p className="text-blue-100">Current top players in our prediction model</p>
      </div>
      
      <div className="p-6">
        <div className="grid gap-4">
          {players.slice(0, 10).map((player, index) => (
            <div key={player.name} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
              <div className="flex items-center">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold mr-4">
                  {player.rank}
                </div>
                <div>
                  <div className="font-semibold text-gray-800">{player.name}</div>
                  <div className="text-sm text-gray-600">
                    {player.wins}/{player.matches_played} matches ({player.win_rate}% win rate)
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-lg font-bold text-blue-600">{player.win_rate}%</div>
                <div className="text-sm text-gray-500">Win Rate</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderModelInfo = () => (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-6">
        <h2 className="text-2xl font-bold">XGBoost Model Insights</h2>
        <p className="text-green-100">Feature importance and model performance</p>
      </div>
      
      {modelInfo && (
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-3xl font-bold text-blue-600">{modelInfo.players_count}</div>
              <div className="text-sm text-gray-600">Players</div>
            </div>
            <div className="bg-green-50 p-4 rounded-lg text-center">
              <div className="text-3xl font-bold text-green-600">{modelInfo.feature_count}</div>
              <div className="text-sm text-gray-600">Features</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <div className="text-3xl font-bold text-purple-600">XGBoost</div>
              <div className="text-sm text-gray-600">Algorithm</div>
            </div>
          </div>

          <h3 className="text-xl font-bold text-gray-800 mb-4">Feature Importance</h3>
          <div className="space-y-3">
            {modelInfo.feature_importance?.slice(0, 8).map((feature, index) => (
              <div key={index} className="flex items-center">
                <div className="w-32 text-sm text-gray-600 flex-shrink-0">
                  {feature.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </div>
                <div className="flex-1 mx-4">
                  <div className="bg-gray-200 rounded-full h-3">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full"
                      style={{ width: `${feature.importance * 100}%` }}
                    ></div>
                  </div>
                </div>
                <div className="text-sm font-medium text-gray-700 w-12">
                  {Math.round(feature.importance * 100)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-green-600 via-blue-600 to-purple-600 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-2">🎾 Wimbledon 2025 Predictor</h1>
            <p className="text-xl text-blue-100">AI-Powered Tennis Match Predictions using XGBoost</p>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-md sticky top-0 z-10">
        <div className="container mx-auto px-4">
          <div className="flex justify-center space-x-8">
            {[
              { id: 'predict', label: 'Match Predictor', icon: '🎯' },
              { id: 'wimbledon', label: 'Wimbledon 2025', icon: '🏆' },
              { id: 'rankings', label: 'Player Rankings', icon: '📊' },
              { id: 'model', label: 'Model Insights', icon: '🤖' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-6 font-medium transition-colors border-b-2 ${
                  activeTab === tab.id
                    ? 'text-blue-600 border-blue-600'
                    : 'text-gray-600 border-transparent hover:text-blue-600'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {activeTab === 'predict' && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-xl shadow-lg overflow-hidden mb-8">
              <div className="bg-gradient-to-r from-blue-600 to-green-600 text-white p-6">
                <h2 className="text-3xl font-bold mb-2">Tennis Match Predictor</h2>
                <p className="text-blue-100">Select two players to predict the match outcome</p>
              </div>
              
              <div className="p-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Player 1</label>
                    <select
                      value={selectedPlayer1}
                      onChange={(e) => setSelectedPlayer1(e.target.value)}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="">Select Player 1</option>
                      {players.map(player => (
                        <option key={player.name} value={player.name}>
                          #{player.rank} {player.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Player 2</label>
                    <select
                      value={selectedPlayer2}
                      onChange={(e) => setSelectedPlayer2(e.target.value)}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="">Select Player 2</option>
                      {players.map(player => (
                        <option key={player.name} value={player.name}>
                          #{player.rank} {player.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Surface</label>
                    <select
                      value={surface}
                      onChange={(e) => setSurface(e.target.value)}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="Grass">Grass (Wimbledon)</option>
                      <option value="Hard">Hard Court</option>
                      <option value="Clay">Clay Court</option>
                    </select>
                  </div>
                </div>

                <button
                  onClick={handlePredict}
                  disabled={loading || !selectedPlayer1 || !selectedPlayer2}
                  className="w-full bg-gradient-to-r from-blue-600 to-green-600 text-white py-4 px-8 rounded-lg font-semibold text-lg hover:from-blue-700 hover:to-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Predicting...
                    </span>
                  ) : (
                    'Predict Match Outcome 🎯'
                  )}
                </button>
              </div>
            </div>

            {/* Prediction Results */}
            {prediction && (
              <div className="max-w-2xl mx-auto">
                {renderPredictionCard(prediction)}
              </div>
            )}
          </div>
        )}

        {activeTab === 'wimbledon' && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-xl shadow-lg overflow-hidden mb-8">
              <div className="bg-gradient-to-r from-green-600 to-purple-600 text-white p-6">
                <h2 className="text-3xl font-bold mb-2">🏆 Wimbledon 2025 Predictions</h2>
                <p className="text-green-100">AI predictions for potential semifinal matchups</p>
              </div>
            </div>

            {wimbledonPredictions && (
              <div className="space-y-6">
                {wimbledonPredictions.predictions?.map((pred, index) => renderPredictionCard(pred, index))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'rankings' && (
          <div className="max-w-4xl mx-auto">
            {renderPlayerRankings()}
          </div>
        )}

        {activeTab === 'model' && (
          <div className="max-w-4xl mx-auto">
            {renderModelInfo()}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-300">
            Powered by XGBoost Machine Learning • Tennis data from multiple free sources
          </p>
          <p className="text-sm text-gray-400 mt-2">
            Predictions are for entertainment purposes only
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;