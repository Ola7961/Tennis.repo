import React, { useState, useEffect } from 'react';
import './TournamentBracket.css';

const TournamentBracket = ({ simulationData, onMatchClick }) => {
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [expandedRounds, setExpandedRounds] = useState({
    'Third Round': true,
    'Fourth Round': true,
    'Quarter Finals': true,
    'Semi Finals': true,
    'Final': true
  });

  if (!simulationData || !simulationData.rounds) {
    return (
      <div className="tournament-bracket-loading">
        <div className="loading-spinner"></div>
        <p>Loading tournament simulation...</p>
      </div>
    );
  }

  const rounds = simulationData.rounds;
  const finalWinner = simulationData.final_winner;

  const handleMatchClick = (match, roundName) => {
    setSelectedMatch({ ...match, round: roundName });
    if (onMatchClick) {
      onMatchClick(match, roundName);
    }
  };

  const toggleRound = (roundName) => {
    setExpandedRounds(prev => ({
      ...prev,
      [roundName]: !prev[roundName]
    }));
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return '#2e7d32'; // Dark green
    if (confidence > 0.7) return '#388e3c'; // Green
    if (confidence > 0.6) return '#689f38'; // Light green
    if (confidence > 0.5) return '#fbc02d'; // Yellow
    return '#f57c00'; // Orange
  };

  const formatScore = (score) => {
    return score.replace(/(\d+)-(\d+)/g, '$1-$2');
  };

  const renderMatch = (match, roundName, matchIndex) => {
    const isWinner1 = match.winner === match.player1;
    const confidenceColor = getConfidenceColor(match.confidence);
    
    return (
      <div 
        key={`${roundName}-${matchIndex}`}
        className={`tournament-match ${selectedMatch?.player1 === match.player1 && selectedMatch?.player2 === match.player2 ? 'selected' : ''}`}
        onClick={() => handleMatchClick(match, roundName)}
      >
        <div className="match-header">
          <span className="match-surface">{match.surface}</span>
          <div 
            className="confidence-indicator"
            style={{ backgroundColor: confidenceColor }}
            title={`Confidence: ${(match.confidence * 100).toFixed(1)}%`}
          >
            {(match.confidence * 100).toFixed(0)}%
          </div>
        </div>
        
        <div className="match-players">
          <div className={`player-row ${isWinner1 ? 'winner' : 'loser'}`}>
            <div className="player-info">
              <span className="player-name">{match.player1}</span>
              {isWinner1 && <span className="winner-checkmark">✓</span>}
            </div>
            <div className="player-score">
              {isWinner1 ? formatScore(match.score) : formatScore(match.score.split(' ').reverse().join(' '))}
            </div>
          </div>
          
          <div className={`player-row ${!isWinner1 ? 'winner' : 'loser'}`}>
            <div className="player-info">
              <span className="player-name">{match.player2}</span>
              {!isWinner1 && <span className="winner-checkmark">✓</span>}
            </div>
            <div className="player-score">
              {!isWinner1 ? formatScore(match.score) : formatScore(match.score.split(' ').reverse().join(' '))}
            </div>
          </div>
        </div>
        
        <div className="match-footer">
          <span className="match-winner">Winner: {match.winner}</span>
        </div>
      </div>
    );
  };

  const renderRound = (roundName, matches) => {
    const isExpanded = expandedRounds[roundName];
    
    return (
      <div key={roundName} className="tournament-round">
        <div 
          className="round-header"
          onClick={() => toggleRound(roundName)}
        >
          <h3>{roundName}</h3>
          <div className="round-info">
            <span className="match-count">{matches.length} matches</span>
            <span className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>▼</span>
          </div>
        </div>
        
        {isExpanded && (
          <div className="round-matches">
            {matches.map((match, index) => renderMatch(match, roundName, index))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="tournament-bracket">
      <div className="bracket-header">
        <h2>Wimbledon 2025 - Gentlemen's Singles</h2>
        <div className="tournament-info">
          <span className="surface-info">🌱 Grass Court</span>
          {finalWinner && (
            <div className="predicted-champion">
              <span className="champion-label">Predicted Champion:</span>
              <span className="champion-name">{finalWinner}</span>
              <span className="champion-trophy">🏆</span>
            </div>
          )}
        </div>
      </div>

      <div className="bracket-rounds">
        {Object.entries(rounds).map(([roundName, matches]) => 
          renderRound(roundName, matches)
        )}
      </div>

      {selectedMatch && (
        <div className="match-details-modal" onClick={() => setSelectedMatch(null)}>
          <div className="match-details-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{selectedMatch.round}</h3>
              <button 
                className="close-button"
                onClick={() => setSelectedMatch(null)}
              >
                ×
              </button>
            </div>
            
            <div className="modal-body">
              <div className="match-summary">
                <h4>{selectedMatch.player1} vs {selectedMatch.player2}</h4>
                <p className="match-surface">Surface: {selectedMatch.surface}</p>
              </div>
              
              <div className="match-result">
                <div className="result-header">
                  <span className="winner-label">Winner:</span>
                  <span className="winner-name">{selectedMatch.winner}</span>
                </div>
                <div className="score-display">
                  <span className="score-label">Score:</span>
                  <span className="score-value">{formatScore(selectedMatch.score)}</span>
                </div>
                <div className="confidence-display">
                  <span className="confidence-label">Prediction Confidence:</span>
                  <span className="confidence-value">
                    {(selectedMatch.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="bracket-legend">
        <h4>Legend</h4>
        <div className="legend-items">
          <div className="legend-item">
            <div className="confidence-sample" style={{ backgroundColor: '#2e7d32' }}></div>
            <span>Very High Confidence (80%+)</span>
          </div>
          <div className="legend-item">
            <div className="confidence-sample" style={{ backgroundColor: '#388e3c' }}></div>
            <span>High Confidence (70-80%)</span>
          </div>
          <div className="legend-item">
            <div className="confidence-sample" style={{ backgroundColor: '#689f38' }}></div>
            <span>Medium Confidence (60-70%)</span>
          </div>
          <div className="legend-item">
            <div className="confidence-sample" style={{ backgroundColor: '#fbc02d' }}></div>
            <span>Low Confidence (50-60%)</span>
          </div>
          <div className="legend-item">
            <div className="confidence-sample" style={{ backgroundColor: '#f57c00' }}></div>
            <span>Very Low Confidence (&lt;50%)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TournamentBracket;

