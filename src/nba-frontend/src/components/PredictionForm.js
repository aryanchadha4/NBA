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

        if (result.error) {
            alert("Prediction failed: " + result.error);
            return;
        }

        setPrediction(result);
        setModelIndex(0); // Reset model index on new prediction
    };


    const handleNextModel = () => {
        setModelIndex((prevIndex) => (prevIndex + 1) % modelNames.length);
    };

    const handlePrevModel = () => {
        setModelIndex((prevIndex) => (prevIndex - 1 + modelNames.length) % modelNames.length);
    };
    console.log("Full prediction object:", prediction);

    // Create an array of predictions corresponding to our model names
    const predictionData = prediction ? [
        prediction.logistic_winner,
        prediction.random_forest_winner,
        prediction.neural_network_winner
    ] : [];

    console.log("Prediction data:", predictionData);

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
                    <h3>📊 Prediction Results</h3>
                    <h4 style={styles.modelHeading}>{modelNames[modelIndex]} Model</h4>
                    <div style={styles.winnerCard}>
                        <div style={styles.winnerTitle}>Predicted Winner</div>
                        <div style={styles.winnerName}>{predictionData[modelIndex]}</div>
                    </div> 
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
    statsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: "15px",
    marginTop: "15px",
    },
    statBox: {
        padding: "10px",
        borderRadius: "8px",
        backgroundColor: "#FFE5EC",
        boxShadow: "0px 2px 5px rgba(0, 0, 0, 0.1)",
        fontSize: "1.1rem",
        textAlign: "center",
    },

    winnerCard: {
    marginTop: "20px",
    padding: "20px",
    borderRadius: "10px",
    backgroundColor: "#FFE5EC",
    boxShadow: "0px 2px 5px rgba(0, 0, 0, 0.1)",
    textAlign: "center",
    display: "inline-block",
    minWidth: "250px",
    },

    winnerTitle: {
        fontSize: "1.2rem",
        fontWeight: "bold",
        color: "#6A0572",
        marginBottom: "8px",
    },

    winnerName: {
        fontSize: "1.6rem",
        fontWeight: "bold",
        color: "#000",
    },

};



export default PredictionForm;





