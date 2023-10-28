<template>
  <div>
    <button @click="preprocessData">Preprocess Data</button>
    <button @click="trainModel">Train Model</button>
    <button @click="evaluateModel">Evaluate Model</button>
    <button @click="predict">Predict</button>
  </div>
</template>

<script>
import axios from "axios";

export default {
  methods: {
    async preprocessData() {
      await axios.post("http://localhost:8081/api/handwriting/preprocess");
      alert("Data Preprocessed Successfully");
    },
    async trainModel() {
      await axios.post("http://localhost:8081/api/handwriting/train");
      alert("Model Trained Successfully");
    },
    async evaluateModel() {
      const response = await axios.get("http://localhost:8081/api/handwriting/evaluate");
      alert(`Model Accuracy: ${response.data}`);
    },
    async predict() {
      const id = 1; // Replace with the ID you want to predict
      const response = await axios.get(`http://localhost:8081/api/handwriting/predict/${id}`);
      alert(`Predicted Value: ${response.data}`);
    },
  },
};
</script>
