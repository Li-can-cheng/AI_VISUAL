<script setup>

</script>

<template>
    <div class="model_container">
        <div class="inner_container">
            epoch:<input type="number" name="">
            <br/>
            model_evaluation:
            <br />
            Accurancy:<input type="checkbox" name="model_evaluation" value="Accurancy">
            <br/>
            F1_score:<input type="checkbox" name="model_evaluation" value="F1_score">
        </div>
    </div>
</template>

<style>
.arguments_container .model_container{
    height: 100%;
    flex: 5;
    background-color: turquoise;
}
</style>