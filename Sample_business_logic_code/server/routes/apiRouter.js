import express from "express";
import { getRnnPrediction, processWithLlm } from "../controllers/dataController.js";

const router = express.Router();

// RNN API Route
router.get("/rnn", getRnnPrediction);

// LLM API Route
router.post("/llm", processWithLlm);

export default router;

