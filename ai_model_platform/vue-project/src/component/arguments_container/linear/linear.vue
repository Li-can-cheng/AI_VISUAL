<script setup>



</script>

<template>

<div class="linear_container">
    神经元数:<input type="number" class="input"><br/>
    激活函数:
    <select class="select">
        <option value="">tanh</option>
        <option value="">ReLU</option>
        <option value="">sigmoid</option>
        <option value="">softmax</option>
    </select>

</div>

</template>


<style>

.arguments_container .linear_container{
    height: 80%;
    padding:20px;
    font-size: 18px;
    flex: 5;
    line-height: 32px;
    background-color: rgba(234, 234, 234,0.2);
    /* background-color: rgb(207, 230, 250); */
    box-shadow: 2px 3px 3px rgba(0,0,0,0.2);
}
.input{
    font-size: 16px;
}
.select{
    font-size: 18px;
}
</style>