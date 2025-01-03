import React, { useState } from 'react';
import Button from '../components/Button';
import Display from '../components/Display';
import BannerImage from '../assets/banner.jpg';

function HomePage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [started, setStarted] = useState(false);

  const handleStartProcess = async () => {
    setLoading(true);
    setResult(null); // Clear previous result
    setStarted(true); // Hide the "Click the button to start the process" text

    // Simulate backend process
    setTimeout(() => {
      setResult({
        product: 'Carrots',
        quantity: '20 kg'
      });
      setLoading(false);
    }, 2000);
  };

  return (
    <div style={styles.container}>
      {/* Header Section */}
      <div style={styles.header}>
        <img 
          src={BannerImage} 
          alt="AI Banner" 
          style={styles.bannerImage} 
        />
        <h1 style={styles.title}>Restaurant Order System</h1>
        <p style={styles.subtitle}>
          Easily track and manage your product requirements with AI assistance.
        </p>
      </div>

      {/* Button Section */}
      <div style={styles.buttonContainer}>
        <Button onClick={handleStartProcess} disabled={loading} />
        {!started && !loading && (
          <p style={styles.instructions}>
            Click the button to start the process.
          </p>
        )}
      </div>

      {/* Result Section */}
      <div style={styles.resultContainer}>
        <Display result={result} />
      </div>
    </div>
  );
}

// Styles for the page
const styles = {
  container: {
    fontFamily: 'Arial, sans-serif',
    backgroundColor: '#f9f9f9',
    padding: '20px',
    minHeight: '100vh',
  },
  header: {
    textAlign: 'center',
    marginBottom: '30px',
  },
  bannerImage: {
    width: '100%',
    maxHeight: '300px',
    objectFit: 'cover',
    borderRadius: '10px',
  },
  title: {
    fontSize: '2.5rem',
    fontWeight: 'bold',
    margin: '20px 0 10px',
  },
  subtitle: {
    fontSize: '1.2rem',
    color: '#666',
  },
  buttonContainer: {
    textAlign: 'center',
    marginTop: '20px',
  },
  instructions: {
    fontSize: '1rem',
    color: '#555',
    marginTop: '10px',
  },
  resultContainer: {
    marginTop: '30px',
    textAlign: 'center',
  },
};

export default HomePage;
