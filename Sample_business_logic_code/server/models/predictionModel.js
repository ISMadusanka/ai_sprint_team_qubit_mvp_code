import mongoose from "mongoose";

const predictionSchema = new mongoose.Schema({
  day: { type: Number, required: true },
  prediction: { type: Number, required: true },
  createdAt: { type: Date, default: Date.now },
});

const Prediction = mongoose.model("Prediction", predictionSchema);

export default Prediction;
