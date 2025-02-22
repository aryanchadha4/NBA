import React, { useState } from "react";
import { getGamePrediction } from "../api";
import { useNavigate } from "react-router-dom";

const PredictionForm = () => {
    const [homeTeam, setHomeTeam] = useState("");
    const [awayTeam, setAwayTeam] = useState("");
    const [prediction, setPrediction] = useState(null);
    const [modelIndex, setModelIndex] = useState(0); // For cycling through models
    const navigate = useNavigate();

    const modelNames = ["Logistic Regression", "Random Forest", "Neural Network"];

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
        setModelIndex(0); // Reset model index on new prediction
    };

    const handleNextModel = () => {
        setModelIndex((prevIndex) => (prevIndex + 1) % modelNames.length);
    };

    const handlePrevModel = () => {
        setModelIndex((prevIndex) => (prevIndex - 1 + modelNames.length) % modelNames.length);
    };

    // Create an array of predictions corresponding to our model names
    const predictionData = prediction ? [
        prediction.logistic_winner,
        prediction.random_forest_winner,
        prediction.neural_network_winner
    ] : [];

    return (
        <div style={styles.container}>
            <h2 style={styles.heading}>NBA Game Predictor</h2>
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

            {prediction && (
                <div style={styles.resultContainer}>
                    <h3>Prediction Results</h3>
                    <h4 style={styles.modelHeading}>{modelNames[modelIndex]} Model</h4>
                    <p style={styles.predictionText}>
                        Predicted Winner: <strong>{predictionData[modelIndex]}</strong>
                    </p>
                    <div style={styles.navButtons}>
                        <button onClick={handlePrevModel} style={styles.navButton}>
                            ⬅ Previous
                        </button>
                        <button onClick={handleNextModel} style={styles.navButton}>
                            Next ➡
                        </button>
                    </div>
                </div>
            )}

            <button onClick={() => navigate("/")} style={styles.backButton}>
                ⬅ Back to Home
            </button>
        </div>
    );
};

const styles = {
    container: {
        textAlign: "center",
        marginTop: "50px",
        backgroundColor: "#FDE2E4",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
    },
    heading: {
        fontSize: "2rem",
        fontWeight: "bold",
        color: "#822659",
        marginBottom: "20px",
    },
    inputContainer: {
        display: "flex",
        gap: "10px",
        marginBottom: "20px",
    },
    input: {
        padding: "10px",
        fontSize: "1rem",
        borderRadius: "8px",
        border: "1px solid #C06C84",
    },
    button: {
        padding: "10px 15px",
        fontSize: "1rem",
        borderRadius: "8px",
        border: "none",
        backgroundColor: "#C06C84",
        color: "white",
        cursor: "pointer",
        transition: "background 0.3s",
    },
    resultContainer: {
        backgroundColor: "#FAF3DD",
        padding: "20px",
        borderRadius: "10px",
        boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
        width: "70%",
        marginTop: "20px",
        textAlign: "center",
    },
    modelHeading: {
        fontSize: "1.5rem",
        fontWeight: "bold",
        color: "#6A0572",
        marginBottom: "10px",
    },
    predictionText: {
        fontSize: "1.2rem",
        marginBottom: "15px",
    },
    navButtons: {
        display: "flex",
        justifyContent: "space-between",
        marginTop: "10px",
    },
    navButton: {
        padding: "8px 12px",
        fontSize: "1rem",
        borderRadius: "8px",
        border: "none",
        backgroundColor: "#6A0572",
        color: "white",
        cursor: "pointer",
    },
    backButton: {
        marginTop: "20px",
        padding: "10px 15px",
        fontSize: "1rem",
        borderRadius: "8px",
        border: "none",
        backgroundColor: "#6A0572",
        color: "white",
        cursor: "pointer",
        transition: "background 0.3s",
    },
};

export default PredictionForm;





