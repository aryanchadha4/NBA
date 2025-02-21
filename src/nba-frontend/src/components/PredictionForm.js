import React, { useState } from "react";
import { getGamePrediction } from "../api";
import { useNavigate } from "react-router-dom";

const PredictionForm = () => {
    const [homeTeam, setHomeTeam] = useState("");
    const [awayTeam, setAwayTeam] = useState("");
    const [prediction, setPrediction] = useState(null);
    const navigate = useNavigate();

    const handlePredict = async () => {
        if (!homeTeam || !awayTeam) {
            alert("Please enter both teams!");
            return;
        }
        if (homeTeam.toLowerCase() === awayTeam.toLowerCase()) {
            alert("Teams must be different!");
            return;
        }
        const result = await getGamePrediction(homeTeam, awayTeam);
        setPrediction(result);
    };

    return (
        <div style={styles.container}>
            <h2>🏀 <span style={styles.heading}>NBA Game Predictor</span> 🏀</h2>
            <div style={styles.inputContainer}>
                <input
                    type="text"
                    placeholder="Enter Home Team"
                    value={homeTeam}
                    onChange={(e) => setHomeTeam(e.target.value)}
                    style={styles.input}
                />
                <input
                    type="text"
                    placeholder="Enter Away Team"
                    value={awayTeam}
                    onChange={(e) => setAwayTeam(e.target.value)}
                    style={styles.input}
                />
                <button onClick={handlePredict} style={styles.button}>Predict Game</button>
            </div>

            {prediction && (
                <div style={styles.resultsBox}>
                    <h3>🏆 Prediction Results 🏆</h3>
                    <p>📊 <strong>Logistic Regression Winner:</strong> {prediction.logistic_winner}</p>
                    <p>🌲 <strong>Random Forest Winner:</strong> {prediction.random_forest_winner}</p>
                    <p>🧠 <strong>Neural Network Winner:</strong> {prediction.neural_network_winner}</p>
                </div>
            )}

            {/* Back to Home Button */}
            <button onClick={() => navigate("/")} style={styles.backButton}>⬅ Back to Home</button>
        </div>
    );
};

// Styling
const styles = {
    container: {
        textAlign: "center",
        marginTop: "50px",
        backgroundColor: "#F7E7E6",
        padding: "20px",
        borderRadius: "15px",
    },
    heading: { color: "#822659", fontSize: "2rem" },
    inputContainer: { display: "flex", gap: "10px", justifyContent: "center", marginBottom: "20px" },
    input: { padding: "10px", borderRadius: "8px", border: "1px solid #ccc" },
    button: {
        backgroundColor: "#C06C84",
        color: "white",
        padding: "10px 20px",
        borderRadius: "10px",
        border: "none",
        cursor: "pointer",
        transition: "0.3s",
    },
    buttonHover: { backgroundColor: "#822659" },
    resultsBox: { backgroundColor: "#FAF3E0", padding: "15px", borderRadius: "10px", marginTop: "20px" },
    backButton: {
        marginTop: "20px",
        padding: "10px 20px",
        backgroundColor: "#8884FF",
        color: "white",
        borderRadius: "10px",
        border: "none",
        cursor: "pointer",
    },
};

export default PredictionForm;




