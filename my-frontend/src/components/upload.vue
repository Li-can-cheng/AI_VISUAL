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
    <div class="submits">
      <form> 
        <input class="file-input" ref='picx' type="file"/>
      </form>
        <button @click="submit" style="margin-top:1.6%;width:50%;height:46px;font-size: 16px;">删除</button>
        <button @click="submit" style="margin-top:1.6%;width:50%;height:46px;font-size: 16px;">上传</button>
    </div>
</template>
<style scoped>
.submits{
  width:100%;
  height:100%;
  left:60px;
}
.submits .file-input{
  outline-style: none;
  border:1px solid #ccc;
  border-radius: 3px;
  padding:64px 30px;
  font-size:14px;
  font-family: "Microsoft soft";
}
.file-input:focus{
  border-color: #66af39;
  outline: 0;
  -webkit-box-shadow:input 0 1px 1px rgba(0,0,0,0.075),0 0 8px rgba(102,175,233,0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,0.075),0 0 8px rgba(102,175,233,0.6);
}
</style>