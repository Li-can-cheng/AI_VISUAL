<script setup>



</script>

<template>

<div class="linear_container">
    神经元数:<input type="number"><br/>
    激活函数:
    <select>
        <option value="">tanh</option>
        <option value="">ReLU</option>
        <option value="">sigmoid</option>
        <option value="">softmax</option>

    </select>

</div>

</template>


<style>

.arguments_container .linear_container{
    height: 100%;
    flex: 5;
    background-color: thistle;
}

</style>