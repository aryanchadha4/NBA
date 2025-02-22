import axios from 'axios';

const API_BASE_URL = "http://127.0.0.1:8000/api";

export const getGamePrediction = async (home_team, away_team) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/game-prediction/`, {
            params: { home_team, away_team }
        });
        return response.data;
    } catch (error) {
        console.error("Error fetching game prediction:", error);
        return null;
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

