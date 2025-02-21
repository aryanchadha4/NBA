import React, { useState } from "react";
import { getPrediction } from "../api";

const PredictionForm = () => {
    const [homeTeam, setHomeTeam] = useState("");
    const [awayTeam, setAwayTeam] = useState("");
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState("");

    const handlePredict = async () => {
        setError("");
        setPrediction(null);

        if (!homeTeam || !awayTeam) {
            setError("Please enter both teams!");
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
            setError(result.error);
            return;
        }

        setPrediction(result);
    };

    return (
        <div style={styles.container}>
            <h1 style={styles.title}>🏀 NBA Game Predictor 🏀</h1>
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
                <button onClick={handlePredict} style={styles.button}>
                    Predict Game
                </button>
            </div>

            {error && <p style={styles.error}>{error}</p>}

            {prediction && (
                <div style={styles.resultContainer}>
                    <h2>🏆 Prediction Results 🏆</h2>
                    <p><strong>📊 Logistic Regression Winner:</strong> {prediction.logistic_winner}</p>
                    <p><strong>🌲 Random Forest Winner:</strong> {prediction.random_forest_winner}</p>
                    <p><strong>🧠 Neural Network Winner:</strong> {prediction.neural_network_winner}</p>
                </div>
            )}
        </div>
    );
};

// 🎨 Styling Object with a Pastel Theme
const styles = {
    container: {
        textAlign: "center",
        marginTop: "50px",
        fontFamily: "Arial, sans-serif",
        backgroundColor: "#FAE3D9", // Light pastel peach background
        minHeight: "100vh",
        padding: "30px",
    },
    title: {
        fontSize: "30px",
        fontWeight: "bold",
        marginBottom: "20px",
        color: "#6A0572", // Deep pastel purple
        textShadow: "2px 2px 4px rgba(0, 0, 0, 0.1)",
    },
    inputContainer: {
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        gap: "10px",
        marginBottom: "20px",
    },
    input: {
        padding: "12px",
        fontSize: "16px",
        width: "220px",
        borderRadius: "10px",
        border: "1px solid #C2B8A3", // Soft pastel beige border
        backgroundColor: "#FFF5E4", // Light cream input background
        color: "#5E548E", // Soft purple text
        textAlign: "center",
        boxShadow: "2px 2px 5px rgba(0, 0, 0, 0.1)",
    },
    button: {
        padding: "12px 20px",
        fontSize: "16px",
        backgroundColor: "#A28089", // Pastel pinkish-purple
        color: "#FFF",
        border: "none",
        borderRadius: "10px",
        cursor: "pointer",
        transition: "background 0.3s ease",
        boxShadow: "2px 2px 5px rgba(0, 0, 0, 0.2)",
    },
    buttonHover: {
        backgroundColor: "#6A0572", // Darker pastel purple on hover
    },
    error: {
        color: "#D7263D", // Soft pastel red
        fontWeight: "bold",
        marginTop: "10px",
    },
    resultContainer: {
        marginTop: "20px",
        padding: "15px",
        borderRadius: "10px",
        backgroundColor: "#FFF5E4", // Light pastel cream
        width: "50%",
        margin: "0 auto",
        boxShadow: "0px 4px 8px rgba(0, 0, 0, 0.15)",
        color: "#5E548E", // Soft purple text
    },
};

export default PredictionForm;



