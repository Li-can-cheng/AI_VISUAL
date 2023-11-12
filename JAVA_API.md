### 发送文件
- 请求地址
  ```
  http://localhost:8080/model/sendFile?task=ImageClassification
  ```
- 请求头
  ```
  Content-Type:form-data
  ```
- 请求参数
  ```
  file:file.xsml
  username:balabala
  ```
### 发送文件预处理
- 请求地址
  ```
  http://localhost:8080/model/send_data_processing
  ```
- 请求头
  ```
  Content-Type:json
  ```
- 请求参数
  ```json
  [
    {
      "name": "Normalize",
      "arguments": {
         "mean": ""
      }
    },
    {
      "name": "Standardize",
      "arguments": {
         "mean": ""
      }
    }
  ]
  ```
### 发送模型
- 请求地址
  ```
  http://localhost:8080/model/MLP
  ```
- 请求头
  ```
  Content-Type:json
  ```
- 请求参数
  ```json
  {
    "name":"MLP",
    "arguments":{
      "epoch":-1,
      "layer":{
        "linear1":256,
        "sigmoid":-1,
        "ReLU1":-1,
        "linear2":128,
        "ReLU2":-1,
        "linear3":10
      }
    }
  }
  ```
- 没有选择的神经元的层数，默认值传-1

### 发送模型
- 请求地址
  ```
  http://localhost:8080/model/send_model_evaluation
  ```
- 请求头
  ```
  Content-Type:json
  ```
- 请求参数
  ```json
  ["Accuracy", "F1"]
  ```
