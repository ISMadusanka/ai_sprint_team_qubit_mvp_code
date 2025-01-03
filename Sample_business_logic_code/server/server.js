import express from "express";
import dotenv from "dotenv";
dotenv.config();
import connectDB from "./config/db.js";
import apiRoutes from "./routes/apiRouter.js";

const port = 4000;
connectDB();

const app = express();
app.use(express.json());

// Use the API routes
app.use("/api", apiRoutes);

app.get("/", (req, res) => {
    res.send("API is running");
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});




 