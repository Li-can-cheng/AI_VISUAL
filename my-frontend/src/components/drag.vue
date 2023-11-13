<script setup>
    function submit(){
        let fd = new FormData();
        fd.append("img",this.$refs.picx.files[0]);
        fd.append("username","balabala");
        axios({
            url:"/f",
            method:"post",
            headers:{'Content-Type':'multipart/form-data'},
            data:fd
        }).then(function(response){
            console.log(response.data)
        })
    }
</script>
<template>
    <form>
        图片：<input ref='picx' type="file"  />
    </form>
    <button @click="submit()">Post利用FormData上传文件</button><br>
</template>