import React from 'react';

// Function to calculate tomorrow's date
function getTomorrowDate() {
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);

  // Format the date as `YYYY-MM-DD`
  const formattedDate = tomorrow.toISOString().split('T')[0];
  return formattedDate;
}

function Display({ result }) {
  if (!result) {
    return null; // Don't render anything if there's no result
  }

  const predictionDate = getTomorrowDate();

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Product Requirement</h2>
      <table style={styles.table}>
        <thead>
          <tr style={styles.headerRow}>
            <th style={styles.th}>Product</th>
            <th style={styles.th}>Prediction Quantity</th>
            <th style={styles.th}>Prediction Date</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={styles.td}>{result.product}</td>
            <td style={styles.td}>{result.quantity}</td>
            <td style={styles.td}>{predictionDate}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

// Styles for the table
const styles = {
  container: {
    textAlign: 'center',
    marginTop: '20px',
    padding: '10px',
    backgroundColor: '#fff',
    borderRadius: '10px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  title: {
    fontSize: '1.5rem',
    marginBottom: '20px',
  },
  table: {
    width: '80%',
    margin: '0 auto',
    borderCollapse: 'collapse',
  },
  headerRow: {
    backgroundColor: '#4CAF50', // Header row color (greenish)
  },
  th: {
    borderBottom: '2px solid #ddd',
    padding: '10px',
    backgroundColor: '#4CAF50', // Column header background color
    color: '#fff', // Column header text color
    fontSize: '1rem',
  },
  td: {
    padding: '10px',
    borderBottom: '1px solid #ddd',
    fontSize: '1rem',
    textAlign: 'center',
  },
};

export default Display;
