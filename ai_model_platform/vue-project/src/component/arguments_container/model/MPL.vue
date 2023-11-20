<script setup>

</script>

<template>
    <div class="model_container">
        <div class="inner_container" >
            epochs:<input type="number" name="">
            <br/>
            model_evaluation:
            <br />
            Accuracy:<input class="s1" type="checkbox" name="model_evaluation" value="Accuracy">
            <br/>
            F1:<input class="s1"  type="checkbox" name="model_evaluation" value="F1">
            <br/>
            Recall:<input class="s1"  type="checkbox" name="model_evaluation" value="Recall">
        </div>
    </div>
</template>

<style>
.inner_container{
    font-size: 20px;
}
.s1{
    /* padding-top: 4px; */
    width:26px;
    height:20px;
}
.arguments_container .model_container{
    height: 80%;
    flex: 5;
    padding:20px;
    font-size: 20px;
    flex: 5;
    line-height: 32px;
    border-radius: 5px;
    margin-right: 3px;
    /* background-color: turquoise; */
    background-color: rgba(234, 234, 234,0.2);
    box-shadow: 0px 3px 3px rgba(0,0,0,0.2);
}
</style>