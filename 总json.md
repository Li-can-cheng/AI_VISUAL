```json
{
  "task": "ImageClassification",
  "import_data": {
    "file_path": "/upload/data.csv",
    "file_type": "import_csv_data"
  },
  "data_preprocessing": [
    {
      "name": "normalize_images",
      "arguments": {
        "mean": ""
      }
    },
    {
      "name": "standardize_images",
      "arguments": {
        "mean": ""
      }
    }
  ],
  "model_selection": {
    "name": "MLP",
    "model_evaluation": [
      "Accuracy",
      "F1_score"
    ],
    "arguments": {
      "epoch": 10,
      "layer": [
        {
          "linear": 256,
          "activate_function": "ReLU"
        },
        {
          "linear": 128,
          "activate_function": "ReLU"
        }
      ]
    }
  }
  
}
```