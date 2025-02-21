import React, { useState } from "react";
import { getPrediction } from "../api";

const PredictionForm = () => {
    const [homeTeam, setHomeTeam] = useState("");
    const [awayTeam, setAwayTeam] = useState("");
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState("");

    const handlePredict = async () => {
        setError(""); // Reset error state
        setPrediction(null); // Reset previous predictions

        if (!homeTeam || !awayTeam) {
            setError("Please enter both home and away teams!");
            return;
        }

        if (homeTeam.toLowerCase() === awayTeam.toLowerCase()) {
            setError("Teams must be different!");
            return;
        }

        const result = await getPrediction(homeTeam, awayTeam);

        if (!result) {
            setError("Failed to fetch predictions. Please try again.");
            return;
        }

        if (result.error) {
            setError(result.error); // Show error message from backend
            return;
        }

        setPrediction(result);
    };

    return (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
            <h2>NBA Game Predictor</h2>
            <input
                type="text"
                placeholder="Enter Home Team"
                value={homeTeam}
                onChange={(e) => setHomeTeam(e.target.value)}
            />
            <input
                type="text"
                placeholder="Enter Away Team"
                value={awayTeam}
                onChange={(e) => setAwayTeam(e.target.value)}
            />
            <button onClick={handlePredict}>Predict Game</button>

            {error && <p style={{ color: "red", marginTop: "10px" }}>{error}</p>}

            {prediction && (
                <div style={{ marginTop: "20px" }}>
                    <h3>Results</h3>
                    <p><strong>Logistic Regression Winner:</strong> {prediction.logistic_winner}</p>
                    <p><strong>Random Forest Winner:</strong> {prediction.random_forest_winner}</p>
                    <p><strong>Neural Network Winner:</strong> {prediction.neural_network_winner}</p>
                </div>
            )}
        </div>
    );
};

export default PredictionForm;

