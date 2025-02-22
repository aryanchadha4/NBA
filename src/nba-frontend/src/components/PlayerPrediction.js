import React, { useState } from "react";
import { getPlayerPrediction } from "../api";
import { useNavigate } from "react-router-dom";

const PlayerPrediction = () => {
    const [player, setPlayer] = useState("");
    const [prediction, setPrediction] = useState(null);
    const [modelIndex, setModelIndex] = useState(0); // Index for cycling through models
    const navigate = useNavigate();

    const modelNames = ["Linear", "Random Forest", "Neural Network"];

    const handlePredict = async () => {
        if (!player) {
            alert("Please enter a player name!");
            return;
        }
        const result = await getPlayerPrediction(player);
        setPrediction(result);
        setModelIndex(0); // Reset to first model on new prediction
    };

    const handleNextModel = () => {
        setModelIndex((prevIndex) => (prevIndex + 1) % modelNames.length);
    };

    const handlePrevModel = () => {
        setModelIndex((prevIndex) => (prevIndex - 1 + modelNames.length) % modelNames.length);
    };

    return (
        <div style={styles.container}>
            <h2 style={styles.heading}> Player Stats Predictor</h2>
            <div style={styles.inputContainer}>
                <input 
                    type="text" 
                    placeholder="Enter Player Name" 
                    value={player} 
                    onChange={(e) => setPlayer(e.target.value)} 
                    style={styles.input}
                />
                <button onClick={handlePredict} style={styles.button}>Predict Stats</button>
            </div>

            {prediction && (
                <div style={styles.resultContainer}>
                    <h3>📊 Predicted Stats for {prediction.player}</h3>
                    <h4 style={styles.modelHeading}>{modelNames[modelIndex]} Model</h4>

                    <div style={styles.statsGrid}>
                        {Object.entries(prediction.predictions).map(([stat, models]) => (
                            <div key={stat} style={styles.statBox}>
                                <strong>{stat}:</strong> {Object.values(models)[modelIndex].toFixed(2)}
                            </div>
                        ))}
                    </div>

                    <div style={styles.navButtons}>
                        <button onClick={handlePrevModel} style={styles.navButton}>⬅ Previous</button>
                        <button onClick={handleNextModel} style={styles.navButton}>Next ➡</button>
                    </div>
                </div>
            )}

            <button onClick={() => navigate("/")} style={styles.backButton}>⬅ Back to Home</button>
        </div>
    );
};

// Styling for UI elements
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
    },
    heading: {
        fontSize: "2rem",
        fontWeight: "bold",
        color: "#822659",
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

export default PlayerPrediction;




