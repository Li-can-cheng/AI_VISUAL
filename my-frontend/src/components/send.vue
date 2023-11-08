<script>
    import axios from 'axios'
    var data = {
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
    function click1button(){
        axios.post('http://localhost:8080/model/send_model_selection',data).then(response=>{
        console.log(response)
        }).catch(error=>{
            console.log(error)
        })
    }
    var clicks = document.getElementById("button");
    clicks.onclick = click1button;
</script>
<template>
    <div id="button">23kdjfbutton</div>
</template>
<style>

</style>