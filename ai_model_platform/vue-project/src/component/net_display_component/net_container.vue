<script setup>
import { useDataStore } from "../../store/data";
import { ref, toRaw } from "vue";
import axios from "axios";
import qs from "querystring";


const data_store = useDataStore();
const model_title = ref();

function drop_hander_net_module(e) {

    let current_select = data_store.get_current_select();
    if (current_select == "linear") {
        let target = e.target;
        let linear = document.createElement("div");
        linear.setAttribute("class", "linear");
        target.appendChild(linear);
    } else {
        model_title.value.innerHTML = current_select;
    }
}

function dragover_hander() {

}

function run_model() {
    let file_input = data_store.get_file_input();
    // axios.post("http://localhost:8080/model/sendFile?task=ImageClassification", qs.stringify({
    //     "file": file_input.file,
    //     "username": file_input.username
    // }, {
    //     header: {
    //         "Content-Type": "form-data"
    //     }
    // })).then(res => {
    //     if (toRefs.data.code == 200) {
    //         console.log(res.data);
    //     }
    // }).catch(err => {
    //     console.log(err);
    // });
    let fd = new FormData();
    fd.append("file", file_input.file);
    fd.append("username", file_input.username);
    axios({
        url: "http://localhost:8080/model/sendFile?task=ImageClassification",
        method: "post",
        headers: { 'Content-Type': 'multipart/form-data' },
        data: fd
    }).then(function (response) {
        // console.log(response.data)
    })
    // console.log(data_store.get_data_processing());
    axios.post('http://localhost:8080/model/send_data_processing', toRaw(data_store.get_data_processing()), {
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => {
        // console.log(response)
    }).catch(error => {
        console.log(error)
    })


    // console.log(data_store.get_model_selection());
    axios.post('http://localhost:8080/model/MLP', data_store.get_model_selection(), {
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => {
        // console.log(response)
        console.log(response.data.result)
    }).catch(error => {
        console.log(error)
    })



}

</script>

<template>
    <div class="net_container">
        <div class="header_container">
            <div class="name item" ref="model_title"></div>
            <div class="btn item" @click="run_model">运行</div>
        </div>
        <div class="net_module_container" @dragover.prevent="dragover_hander" @drop.prevent="drop_hander_net_module">
        </div>
    </div>
</template>

<style>
.net_display_container .net_container {
    height: 100%;
    flex: 7;
}

.net_display_container .net_container .header_container {
    width: 100%;
    height: 10%;
    background-color: cadetblue;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.net_display_container .net_container .header_container .item {
    margin: 0 30px;
    width: 100px;
    height: 70%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.net_display_container .net_container .header_container .name {}

.net_display_container .net_container .header_container .btn {
    background-color: wheat;
    border-radius: 30px;
    cursor: pointer;
}

.net_display_container .net_container .net_module_container {
    width: 100%;
    height: 90%;
    background-color: yellowgreen;
    display: flex;
    align-items: center;
    justify-content: flex-start;
}

.net_display_container .net_container .net_module_container .linear {
    width: 80px;
    height: 80%;
    background-color: violet;
    margin: 0 40px;
}
</style>