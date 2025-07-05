import React, { useState, useEffect } from 'react';
import './App.css';
import TournamentBracket from './components/TournamentBracket';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL;

// Player Search Component with Autocomplete
function PlayerSearch({ label, value, onChange, placeholder }) {
  const [searchTerm, setSearchTerm] = useState(value || '');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setSearchTerm(value || '');
  }, [value]);

  const searchPlayers = async (query) => {
    if (query.length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/players/search?q=${encodeURIComponent(query)}`);
      const data = await response.json();
      setSuggestions(data.players || []);
      setShowSuggestions(true);
    } catch (error) {
      console.error('Error searching players:', error);
      setSuggestions([]);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const value = e.target.value;
    setSearchTerm(value);
    searchPlayers(value);
  };

  const handleSuggestionClick = (player) => {
    setSearchTerm(player.name);
    onChange(player.name);
    setShowSuggestions(false);
  };

  const handleInputBlur = () => {
    // Delay hiding suggestions to allow for clicks
    setTimeout(() => setShowSuggestions(false), 200);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      onChange(searchTerm);
      setShowSuggestions(false);
    }
  };

  return (
    <div className="player-search-container">
      <label>{label}</label>
      <div className="search-input-wrapper">
        <input
          type="text"
          value={searchTerm}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="search-input"
        />
        {loading && <div className="search-loading">Searching...</div>}
        {showSuggestions && suggestions.length > 0 && (
          <div className="suggestions-dropdown">
            {suggestions.map((player, index) => (
              <div
                key={index}
                className="suggestion-item"
                onClick={() => handleSuggestionClick(player)}
              >
                <div className="suggestion-name">{player.name}</div>
                <div className="suggestion-details">
                  Rank: #{player.rank} | Win Rate: {player.win_rate}% | Matches: {player.total_matches}
                </div>
              </div>
            ))}
          </div>
        )}
        {showSuggestions && suggestions.length === 0 && !loading && searchTerm.length >= 2 && (
          <div className="suggestions-dropdown">
            <div className="suggestion-item no-results">No players found</div>
          </div>
        )}
      </div>
    </div>
  );
}

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
  const [drawData, setDrawData] = useState(null);
  const [drawProbabilities, setDrawProbabilities] = useState(null);
  const [tournamentSimulation, setTournamentSimulation] = useState(null);

  useEffect(() => {
    fetchPlayers();
    fetchWimbledonPredictions();
    fetchModelInfo();
    fetchDrawData();
    fetchDrawProbabilities();
    fetchTournamentSimulation();
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

  const fetchDrawData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/wimbledon/draw`);
      const data = await response.json();
      setDrawData(data);
    } catch (error) {
      console.error('Error fetching draw data:', error);
    }
  };

  const fetchDrawProbabilities = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/wimbledon/probabilities`);
      const data = await response.json();
      setDrawProbabilities(data);
    } catch (error) {
      console.error('Error fetching draw probabilities:', error);
    }
  };

  const fetchTournamentSimulation = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/wimbledon/simulation`);
      const data = await response.json();
      setTournamentSimulation(data);
    } catch (error) {
      console.error('Error fetching tournament simulation:', error);
    }
  };

  const predictMatch = async () => {
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
        }),
      });

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error predicting match:', error);
      alert('Error predicting match. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleMatchClick = (match, roundName) => {
    console.log(`Match clicked: ${match.player1} vs ${match.player2} in ${roundName}`);
    // Could be used to show additional match details or statistics
  };

  const renderPredictTab = () => (
    <div className="tab-content">
      <div className="predictor-header">
        <h2>Tennis Match Predictor</h2>
        <p>Search and select two players to predict the match outcome</p>
      </div>
      
      <div className="form-container">
        <PlayerSearch
          label="Player 1"
          value={selectedPlayer1}
          onChange={setSelectedPlayer1}
          placeholder="Search for Player 1..."
        />
        
        <PlayerSearch
          label="Player 2"
          value={selectedPlayer2}
          onChange={setSelectedPlayer2}
          placeholder="Search for Player 2..."
        />
        
        <div className="form-group">
          <label>Surface</label>
          <select value={surface} onChange={(e) => setSurface(e.target.value)}>
            <option value="Grass">Grass (Wimbledon)</option>
            <option value="Hard">Hard Court</option>
            <option value="Clay">Clay Court</option>
          </select>
        </div>
        
        <button 
          onClick={predictMatch} 
          disabled={loading || !selectedPlayer1 || !selectedPlayer2}
          className="predict-button"
        >
          {loading ? 'Predicting...' : 'Predict Match Outcome 🎯'}
        </button>
      </div>

      {prediction && (
        <div className="prediction-result">
          <h3>Match Prediction</h3>
          <div className="prediction-details">
            <div className="match-info">
              <h4>{prediction.player1} vs {prediction.player2}</h4>
              <p>Surface: {prediction.surface}</p>
            </div>
            
            <div className="prediction-outcome">
              <div className="winner">
                <strong>Predicted Winner: {prediction.predicted_winner}</strong>
                <p>Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
              </div>
              
              <div className="probabilities">
                <div className="prob-bar">
                  <span>{prediction.player1}</span>
                  <div className="bar">
                    <div 
                      className="fill player1" 
                      style={{width: `${prediction.player1_win_probability * 100}%`}}
                    ></div>
                  </div>
                  <span>{(prediction.player1_win_probability * 100).toFixed(1)}%</span>
                </div>
                
                <div className="prob-bar">
                  <span>{prediction.player2}</span>
                  <div className="bar">
                    <div 
                      className="fill player2" 
                      style={{width: `${prediction.player2_win_probability * 100}%`}}
                    ></div>
                  </div>
                  <span>{(prediction.player2_win_probability * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
            
            <div className="stats-comparison">
              <div className="stat-row">
                <span>Ranking</span>
                <span>#{prediction.player1_rank}</span>
                <span>#{prediction.player2_rank}</span>
              </div>
              <div className="stat-row">
                <span>Overall Win Rate</span>
                <span>{prediction.player1_win_rate}%</span>
                <span>{prediction.player2_win_rate}%</span>
              </div>
              <div className="stat-row">
                <span>{prediction.surface} Win Rate</span>
                <span>{prediction.player1_surface_win_rate}%</span>
                <span>{prediction.player2_surface_win_rate}%</span>
              </div>
              <div className="stat-row">
                <span>Head-to-Head</span>
                <span colSpan="2">{prediction.head_to_head} ({prediction.h2h_matches} matches)</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderWimbledonTab = () => (
    <div className="tab-content">
      <div className="wimbledon-header">
        <h2>Wimbledon 2025 Predictions</h2>
        <p>AI-powered predictions for potential Wimbledon matchups</p>
      </div>
      
      {wimbledonPredictions && (
        <div className="wimbledon-predictions">
          {wimbledonPredictions.predictions?.map((pred, index) => (
            <div key={index} className="wimbledon-match">
              <div className="match-header">
                <h4>{pred.player1} vs {pred.player2}</h4>
                <span className="surface-tag">Grass Court</span>
              </div>
              
              <div className="match-prediction">
                <div className="predicted-winner">
                  Winner: <strong>{pred.predicted_winner}</strong>
                  <span className="confidence">({(pred.confidence * 100).toFixed(1)}%)</span>
                </div>
                
                <div className="match-probabilities">
                  <div className="prob-item">
                    <span>{pred.player1}</span>
                    <span>{(pred.player1_win_probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="prob-item">
                    <span>{pred.player2}</span>
                    <span>{(pred.player2_win_probability * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderDrawTab = () => (
    <div className="tab-content">
      <div className="draw-tab-header">
        <h2>Wimbledon 2025 Tournament Bracket</h2>
        <p>Complete tournament simulation with predicted match results and scores</p>
      </div>
      
      <TournamentBracket 
        simulationData={tournamentSimulation}
        onMatchClick={handleMatchClick}
      />
    </div>
  );

  const renderRankingsTab = () => (
    <div className="tab-content">
      <div className="rankings-header">
        <h2>Player Rankings</h2>
        <p>Top ATP players with comprehensive statistics</p>
      </div>
      
      <div className="rankings-table">
        <div className="table-header">
          <span>Rank</span>
          <span>Player</span>
          <span>Matches</span>
          <span>Win Rate</span>
          <span>Grass Win Rate</span>
          <span>Recent Form</span>
        </div>
        
        {players.slice(0, 20).map((player, index) => (
          <div key={index} className="table-row">
            <span>#{player.rank}</span>
            <span className="player-name">{player.name}</span>
            <span>{player.total_matches}</span>
            <span>{player.win_rate}%</span>
            <span>{player.grass_win_rate}%</span>
            <span>{player.recent_form}</span>
          </div>
        ))}
      </div>
    </div>
  );

  const renderModelTab = () => (
    <div className="tab-content">
      <div className="model-header">
        <h2>Model Insights</h2>
        <p>XGBoost + Neural Network ensemble performance and feature importance</p>
      </div>
      
      {modelInfo && (
        <div className="model-info">
          <div className="model-stats">
            <div className="stat-card">
              <h3>{modelInfo.players_count || 0}</h3>
              <p>Players</p>
            </div>
            <div className="stat-card">
              <h3>{modelInfo.feature_count || 0}</h3>
              <p>Features</p>
            </div>
            <div className="stat-card">
              <h3>Ensemble</h3>
              <p>XGB + NN</p>
            </div>
          </div>
          
          <div className="feature-importance">
            <h3>Feature Importance (XGBoost)</h3>
            {modelInfo.xgb_feature_importance?.slice(0, 8).map((feature, index) => (
              <div key={index} className="feature-item">
                <span className="feature-name">{feature.feature}</span>
                <div className="importance-bar">
                  <div 
                    className="importance-fill" 
                    style={{width: `${(feature.importance * 100)}%`}}
                  ></div>
                </div>
                <span className="importance-value">{(feature.importance * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>🎾 Wimbledon 2025 Predictor</h1>
          <p>AI-Powered Tennis Predictions with Neural Network Ensemble</p>
        </div>
      </header>

      <nav className="tab-navigation">
        <button 
          className={activeTab === 'predict' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('predict')}
        >
          🎯 Match Predictor
        </button>
        <button 
          className={activeTab === 'wimbledon' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('wimbledon')}
        >
          🏆 Wimbledon 2025
        </button>
        <button 
          className={activeTab === 'draw' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('draw')}
        >
          🗂️ Tournament Bracket
        </button>
        <button 
          className={activeTab === 'rankings' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('rankings')}
        >
          📊 Player Rankings
        </button>
        <button 
          className={activeTab === 'model' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('model')}
        >
          🤖 Model Insights
        </button>
      </nav>

      <main className="main-content">
        {activeTab === 'predict' && renderPredictTab()}
        {activeTab === 'wimbledon' && renderWimbledonTab()}
        {activeTab === 'draw' && renderDrawTab()}
        {activeTab === 'rankings' && renderRankingsTab()}
        {activeTab === 'model' && renderModelTab()}
      </main>

      <footer className="app-footer">
        <p>Powered by XGBoost + Neural Network Ensemble • Real ATP data from multiple sources</p>
        <p>Predictions are for entertainment purposes only</p>
      </footer>
    </div>
  );
}

export default App;

