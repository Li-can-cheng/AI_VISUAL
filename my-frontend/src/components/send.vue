<script setup>
    import axios from 'axios'
    const data = {
        "name":"MLP",
        "arguments":{
        "epoch":-1,
        "layer":{
        "linear1":256,
        "sigmoid":-1,
        "ReLU1":-1,
        "linear2":128,
        "ReLU2":-1,
        "linear3":-1
      }
    }
    }
    const processData =   [
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
  ]
    const evaData = ["Accuracy", "F1_score"]
    function say(message) {
        axios.post('http://localhost:8080/model/MLP',data,{
            headers:{
                'Content-Type':'application/json'
            }
        }).then(response=>{
            // console.log(response)
            console.log(response)
        }).catch(error=>{
            console.log(error)
        })
    } 
    function pre_process(){
        axios.post('http://localhost:8080/model/send_data_processing',processData,{
            headers:{
                'Content-Type':'application/json'
            }
        }).then(response=>{
            console.log(response)
        }).catch(error=>{
            console.log(error)
        })
    }
    function evaluation(){
        axios.post('http://localhost:8080/model/send_model_evaluation',evaData,{
            headers:{
                'Content-Type':'application/json'
            }
        }).then(response=>{
            console.log(response)
        }).catch(error=>{
            console.log(error)
        })
    }
</script>
<template>
    <button @click="pre_process">文件预处理</button>
    <br/>
    <br/>
	<button @click="say">模型参数上传</button>
    <br/>
    <br/>
	<button @click="evaluation">模型评估</button>
</template>