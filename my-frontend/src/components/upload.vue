<!-- <script>
    var uploadResult=document.getElementById("uploadResult")
    function uploadFile() {
        // console.log("nihao")
        alert("nihaoya")
        // var fileInput = document.getElementById('fileInput');
        // var file = fileInput.files[0];
        // if (!file) return; // 没有选择文件
        // var xhr = new XMLHttpRequest();
        // // 处理上传进度
        // xhr.upload.onprogress = function(event) {
        //     var percent = 100 * event.loaded / event.total;
        //     uploadResult.innerHTML='上传进度：' + percent + '%';
        // };
        // // 当上传完成时调用
        // xhr.onload = function() {
        //     if (xhr.status === 200) {
        //         uploadResult.innerHTML='上传成功'+ xhr.responseText;
        //     }
        // }
        // xhr.onerror = function() {
        //     uploadResult.innerHTML='上传失败';
        // }
        // // 发送请求
        // xhr.open('POST', 'http://localhost:8080/model/sendFile?task=ImageClassification', true);
        // var formData = new FormData();
        // formData.append('file', file);
        // xhr.send(formData);
    }
    function says(message) {
        // axios.post('http://localhost:8080/model/send_model_selection',data,{
        //     headers:{
        //         'Content-Type':'application/json'
        //     }
        // }).then(response=>{
        //     // console.log(response)
        //     console.log(response)
        // }).catch(error=>{
        //     console.log(error)
        // })
        alert("nihao")
    } 
</script>
<template>
  <h3>文件上传</h3>
  <input type="file" id="fileInput" name="fileInput" >
  <span id="uploadResult"></span>
  <button @click="uploadFile">点击上传</button>
  <button @click="says">模型参数上传</button>
</template>
<style>

</style> -->

<!-- <template>
  <input type="file" ref="upload" >
</template>
<script>
    function upload(){
       var objfile = this.$refs.upload.files[0];
       var formData = new FormData();
       formData.append('file',this.$refs.upload.files[0]);
       var flag;
       fileupload(formData).then(res=>{
           if (res.code!=200){
             console.log(res)
             flag = true;
           }else {
             flag = false;
           }
        })
    }
    import axios from 'axios'

export function request(config) {
  // 1.创建axios的实例
  const instance = axios.create({
    baseURL: 'http://localhost:8888',
    //timeout: 10000,
    headers: {
      'Content-Type': "application/json;charset=utf-8",
      'Authorization': window.sessionStorage.getItem("token")
    },
    changeOrigin: true
  })
    // 2.axios的拦截器
    // 2.1.请求拦截的作用
    instance.interceptors.request.use(config => {
        return config
    }, err => {
        // console.log(err);
    })

    // 2.2.响应拦截
    instance.interceptors.response.use(res => {
        return res.data
    }, err => {
        console.log(err);
    })
    // 3.发送真正的网络请求
    return instance(config)
    }

</script>
<style>

</style> -->
<template>
    <!-- <form id="myForm" action="http://localhost:8080/model/sendFile?task=ImageClassification">
    文件上传：<input type="file" id="fileInput" name="fileInput" ><br>
    <input type="button" onclick="myFunction()" value="提交表单">
    </form> -->
</template>
<script>
    function myFunction() {
        var myform = document.getElementById("myForm");
        console.log("ih")
    }
</script>