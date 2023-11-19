<script setup>

</script>

<template>
    <div class="model_container">
        <div class="inner_container">
            epochs:<input type="number" name="">
            <br/>
            model_evaluation:
            <br />
            Accuracy:<input type="checkbox" name="model_evaluation" value="Accuracy">
            <br/>
            F1:<input type="checkbox" name="model_evaluation" value="F1">
            <br/>
            Recall:<input type="checkbox" name="model_evaluation" value="Recall">
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