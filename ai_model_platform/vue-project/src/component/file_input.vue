<script setup>
import { ref } from "vue";
import { useFileStore } from "../store/file";

const file_store = useFileStore();

const file = ref();
const file_container = ref();




/**
 * 上传文件的按钮
 * @param {*} e 
 */
function click_button(e) {
    file.value.click();
}
/**
 * 当文件改变的时候
 * @param {*} e 
 */
function file_change(e) {
    let f = file.value.files[0];

    console.log(f);
    let input_file = document.createElement("div");
    input_file.setAttribute("class", "file");
    input_file.setAttribute("name", f.name);
    input_file.setAttribute("draggable", "true");
    input_file.innerText = f.type;
    file_store.set_file(f.name, f);

    // let store_file = file_store.get_file(input_file.getAttribute("file"));
    // console.log(store_file);

    //给新建的文件预览添加函数
    input_file.addEventListener("dragstart",dragstart_hander);
    input_file.addEventListener("dragend",dragend_hander);


    file_container.value.appendChild(input_file);
}
/**
 * 拖拽开始，然后存储当前文件
 * @param {*} e 
 */
function dragstart_hander(e) {
    console.log("dragstart");
    e.effectAllowed = "copyMove";
    //获取文件的dom元素
    let current_file = e.target;
    //获取文件的名称
    let name = current_file.getAttribute("name");
    //在pinia中存储当前文件的信息
    file_store.set_current_file({
        name,
        dom:current_file,
        file:file_store.get_file(name),
        type:file_store.get_file(name).type
    })
}
function dragend_hander(e){
    console.log("dragend");
}

</script>
<template>
    <div class="file_input_container">
        <div class="file_container" ref="file_container">
        </div>
        <div class="button_container">
            <input type="file" @change="file_change(e)" name="file" ref="file">
            <button class="button" @click="click_button(e)">上传文件</button>
        </div>
    </div>
</template>
<style>
.file_input_container {
    width: 100%;
    height: 100%;
    /* background-color: #eee; */
}

.file_input_container .file_container {
    width: 100%;
    height: 80%;
    border:3px dashed rgba(0,0,0,0.4);
    display: flex;
    justify-content: flex-start;
    align-items: start;
    /* 使得子元素自动换行 */
    flex-wrap: wrap;
}

.file_input_container .file_container .file {
    width: 20%;
    height: 60px;
    margin: 0 2%;
    border-radius: 20px;
    margin-top: 20px;
    background-color: #fff;
    text-align: center;
    line-height: 60px;
    overflow: hidden;
    /*超出部分隐藏*/
    white-space: nowrap;
    /*禁止换行*/
    text-overflow: ellipsis;
    /*省略号*/
}

.file_input_container .button_container {
    width: 100%;
    margin-top: 2%;
    height: 20%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.file_input_container .button_container input[type="file"] {
    display: none;
}

.file_input_container .button_container .button {
    border: 0;
    outline: 0;
    width: 100px;
    height: 40px;
    border-radius: 50px;
}
</style>