<template>
  <div id="app">
    <h1>欢迎来到文件上传示例！🎉</h1>
    <FileUpload />
  </div>
</template>

<script>
import FileUpload from './components/UploadFile.vue';

export default {
  components: {
    FileUpload
  }
}
</script>

<style>
#app {
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

h1 {
  font-size: 2em;
  margin-bottom: 20px;
}
</style>
