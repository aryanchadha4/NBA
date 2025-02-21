import axios from 'axios';

const API_URL = "http://127.0.0.1:8000/api/game-prediction/";  // Django API URL

export const getPrediction = async (home_team, away_team) => {
    try {
        const response = await axios.get(API_URL, {
            params: { home_team, away_team }
        });
        return response.data;
    } catch (error) {
        console.error("Error fetching predictions:", error);
        return null;
    }
};
