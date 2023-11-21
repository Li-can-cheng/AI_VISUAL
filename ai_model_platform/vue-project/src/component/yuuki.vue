<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

const logs = ref("");  // 用于保存从 WebSocket 接收到的日志

let socket = null;
let reconnectInterval = 5000; // 设置重连间隔为 5000 毫秒（5 秒）

const connectToWebSocket = () => {
  socket = new WebSocket("ws://localhost:8000/ws/train_mlp_model");

  socket.onmessage = (event) => {
    logs.value += event.data + "\n";  // 将新日志追加到现有日志
  };

  socket.onclose = () => {
    console.log("WebSocket disconnected. Attempting to reconnect...");
    setTimeout(connectToWebSocket, reconnectInterval);  // 重试机制
  };

  socket.onerror = (error) => {
    console.error('WebSocket Error:', error);
    socket.close(); // 在错误时关闭并触发重连
  };
};

onMounted(() => {
  connectToWebSocket();
});

onUnmounted(() => {
  if (socket) {
    socket.close();
  }
});
</script>

<template>
  <div class="log-display">
    <pre>{{ logs }}</pre>  <!-- 使用 pre 标签来保留日志的格式 -->
  </div>
</template>

<style scoped>
.log-display {
  position: relative; /* 从 fixed 改为 relative */
  max-width: 100%; /* 限制最大宽度 */
  max-height: 100%; /* 限制最大高度 */
  bottom: 10px;
  right: 10px;
  width: 100%;  /* 稍微增加宽度 */
  height: 250px; /* 增加高度 */
  overflow: auto;  /* 自动滚动条 */
  background-color: #f5f5f5;  /* 淡灰色背景，更柔和的视觉效果 */
  border: 1px solid #ccc;  /* 边框颜色更柔和 */
  padding: 15px; /* 内边距增加 */
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);  /* 添加阴影以提升层次感 */
  border-radius: 8px;  /* 边框圆角 */
  font-family: 'Courier New', Courier, monospace; /* 使用等宽字体，适合代码和日志显示 */
  color: #333;  /* 字体颜色更深，增强可读性 */
  font-size: 0.9rem; /* 调整字体大小 */
  white-space: pre-wrap;  /* 保证换行和空格被正确显示 */
}

</style>
