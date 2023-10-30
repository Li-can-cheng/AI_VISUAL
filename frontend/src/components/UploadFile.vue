<template>
  <div class="file-upload">
    <h1>文件上传组件 📤</h1>
    <input type="file" ref="fileInput" />
    <button @click="handleUpload">上传文件</button>
    <p v-if="uploadStatus">{{ uploadStatus }}</p>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';

const fileInput = ref(null);
const uploadStatus = ref('');

const handleUpload = async () => {
  if (!fileInput.value?.files?.[0]) {
    uploadStatus.value = '请选择一个文件！🚨';
    return;
  }

  const file = fileInput.value.files[0];
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await axios.post('http://localhost:8080/upload', formData);
    uploadStatus.value = `上传成功！🎉 文件名：${response.data.fileName}`;
  } catch (error) {
    uploadStatus.value = '上传失败：😞 ' + error;
  }
};
</script>
