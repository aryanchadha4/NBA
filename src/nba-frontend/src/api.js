import axios from 'axios';

const API_BASE_URL = "http://127.0.0.1:8000/api";

export const getGamePrediction = async (homeTeam, awayTeam) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/game-prediction/?home_team=${homeTeam}&away_team=${awayTeam}`
    );

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error in getGamePrediction:", error);
    throw error;
  }
};



export const getPlayerPrediction = async (player) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/player-prediction/`, {
            params: { player }
        });
        return response.data;
    } catch (error) {
        console.error("Error fetching player prediction:", error);
        return null;
    }
};

