<script setup>
import {useDataStore} from "../store/data";


const data_store = useDataStore();

/**
 * 中枢拖拽函数
 */
function dragstart_hander_all(e){
    let category = e.target.getAttribute("category");
    if(category === "data_processing"){
        dragstart_hander_data_progressing(e);
    }else if(category === "model"){
        // dragstart_hander_model(category);
        data_store.set_current_select(e.target.getAttribute("name"));
    }else if(category === "linear"){
        // dragstart_hander_linear(category);
        data_store.set_current_select(category);
    }
}
/**
 * 数据预处理拖拽函数
 * @param {*} e 
 */
function dragstart_hander_data_progressing(e){
    let data_processing_unit = e.target;
    e.effectAllowed = "copyMove";
    console.log(data_processing_unit.getAttribute("name"));
    //设置当前current_data_processing
    data_store.set_current_data_processing({
        name:data_processing_unit.getAttribute("name"),
        arguments:{
            mean:""
        }
    })
}

function dragstart_hander_model(e){

}

function dragstart_hander_linear(e){

}

function dragend_hander(e){
    data_store.set_current_data_processing({
        name:"",
        arguments:{
            mean:""
        }
    });
    data_store.set_current_select("");
}

</script>
<template>
    <div class="module_select_container">
        <div class="item" category="data_processing" name="normalize_image" draggable="true"
        @dragstart="dragstart_hander_all" @dragend="dragend_hander">normalize</div>
        <div class="item" category="data_processing" name="standardize_image" draggable="true"
        @dragstart="dragstart_hander_all" @dragend="dragend_hander">standardize</div>
        <div class="item" category="model" name="MLP" draggable="true"
        @dragstart="dragstart_hander_all">MLP</div>
        <div class="item" category="linear" draggable="true"
        @dragstart="dragstart_hander_all">Linear</div>
    </div>
</template>
<style>
    .module_select_container{
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: flex-start;
        flex-wrap: wrap;
    }
    .module_select_container .item{
        width: 80%;
        height: 80px;
        background-color: antiquewhite;
        margin: 0 auto;
        margin-top: 20px;
        text-align: center;
        line-height: 80px;
    }

</style>