import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import PredictionForm from "./components/PredictionForm";
import PlayerPrediction from "./components/PlayerPrediction";

const Home = () => {
  return (
      <div style={styles.container}>  {/* Apply styles.container */}
          <h1 style={styles.heading}>Welcome to NBA Predictions</h1>
          <p style={styles.description}>Select a feature:</p>
          <Link to="/predict">
              <button style={styles.button}>Game Predictor</button>
          </Link>
          <Link to="/player-predict">
              <button style={styles.button}>Player Stats Predictor</button>
          </Link>
      </div>
  );
};


const App = () => {
    return (
        <Router>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/predict" element={<PredictionForm />} />
              <Route path="/player-predict" element={<PlayerPrediction />} />
            </Routes>
        </Router>
    );
};

// Basic inline styles
const styles = {
    container: {
        textAlign: "center",
        marginTop: "100px",
        backgroundColor: "#FAE3D9",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
    },
    heading: { fontSize: "2.5rem", color: "#822659" },
    description: { fontSize: "1.2rem", color: "#555" },
    button: {
        marginTop: "20px",
        padding: "10px 20px",
        fontSize: "1.2rem",
        borderRadius: "10px",
        border: "none",
        backgroundColor: "#C06C84",
        color: "white",
        cursor: "pointer",
    },
};

export default App;
