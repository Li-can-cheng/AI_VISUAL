<script setup>
import { useDataStore } from '../../store/data';
import {ref} from "vue";

const data_store = useDataStore();
const data_processing_container = ref();

function drop_hander_data_processing(e){
    //判断放入的是否是data_processing
    let name = data_store.get_current_data_processing().name;
    if(name === "") return;
    console.log(111);
    //创建新的div
    let data_processing_unit = document.createElement("div");
    data_processing_unit.setAttribute("class","data_processing_unit");
    data_processing_unit.setAttribute("name",name);
    data_processing_unit.addEventListener("dblclick",delete_data_processing_unit);
    data_processing_unit.innerHTML = name;
    //添加到容器中
    data_processing_container.value.appendChild(data_processing_unit);
    //并且将current_data_processing添加到data_processing
    data_store.add_data_processing();
    // console.log(data_store.get_data_processing());
    // console.log(data_store.get_current_data_processing());
}

function dragover_hander(e){
}


function delete_data_processing_unit(e){
    //并且删除data_processing
    data_store.delete_data_processing(e.target.getAttribute("name"));
    data_processing_container.value.removeChild(e.target);
    console.log(data_store.get_data_processing());
}
</script>

<template>
    <div class="data_processing_container" ref="data_processing_container"
    @drop.prevent="drop_hander_data_processing"
    @dragover.prevent="dragover_hander">
    </div>
</template>

<style>
.net_display_container .data_processing_container {
    height: 100%;
    flex: 1;
    background-color: azure;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-evenly;
}
.net_display_container .data_processing_container .data_processing_unit{
    width: 80px;
    height: 80px;
    background-color: thistle;
    text-align: center;
    line-height: 80px;
    border-radius: 20px;
    /* 禁止用户选择文字，为了美观 */
    user-select: none;
    overflow: hidden;
    /*超出部分隐藏*/
    white-space: nowrap;
    /*禁止换行*/
    text-overflow: ellipsis;
    /*省略号*/
}
</style>