import  Data  from '../models/predictionModel.js'; // Importing the Data model

// Simulate fetching RNN prediction
export const getRnnPrediction = async (req, res) => {
  try {
    // Simulating fetching the latest RNN prediction from the database (example query)
    const latestPrediction = await Data.findOne().sort({ createdAt: -1 }).exec(); // Assuming Data model has a timestamp field
    
    if (!latestPrediction) {
      return res.status(404).json({ message: "No prediction data found." });
    }

    // Simulated RNN response
    const rnnResponse = {
      day: latestPrediction.day || 1, // Example day field from the model
      prediction: latestPrediction.prediction || 824.7135009765625, // Example prediction field
    };

    return res.status(200).json(rnnResponse);
  } catch (error) {
    console.error('Error fetching RNN prediction:', error);
    return res.status(500).json({ message: "Error fetching RNN prediction", error: error.message });
  }
};

// Simulate processing data with LLM
export const processWithLlm = async (req, res) => {
  try {
    const updatedData = req.body;

    // Check if input data exists
    if (!updatedData || Object.keys(updatedData).length === 0) {
      return res.status(400).json({ message: "No data provided for processing." });
    }

    // Assuming we need to save or update data using the Data model
    const newData = new Data(updatedData);
    await newData.save(); // Save the new data to the database

    // Simulated LLM response
    const llmResponse = {
      message: "LLM successfully processed the data",
      input: updatedData,
    };

    return res.status(200).json(llmResponse);
  } catch (error) {
    console.error('Error processing data with LLM:', error);
    return res.status(500).json({ message: "Error processing data with LLM", error: error.message });
  }
};


