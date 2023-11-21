<template>
  <div class="user-func-container">
    <br>
    <button @click="predictModel" class="btn-predict">模型预测</button>
    <br>
    <button @click="logout" class="btn-logout"  >用户登出</button>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
export default {
  name: 'UserFunc',
  setup() {
    const router = useRouter(); // 获取router实例

    function predictModel() {
      // 模型预测的逻辑
      console.log('进行模型预测');
      window.location.href = 'http://localhost:8080/model-prediction';

    }
    function logout() {
      // 用户登出的逻辑
      console.log('用户登出');
      //router.push('http://127.0.0.1:8080/login');
      window.location.href = 'http://localhost:8080/login';

    }
    return {
      predictModel,
      logout

    };
  }

}
</script>

<style scoped>
.user-func-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px; /* 按钮之间的间距 */
  margin-top: 50px;
  margin-left: 15px;
  margin-right: 15px;
  width: 11%; /* 调整宽度 */
  height: 180px; /* 调整高度 */
  padding: 50px; /* 调整内边距 */
  color: blue; /* 白色文本 */
  border: 6px solid rgba(148, 238, 243, 0.5);
  background-color: rgba(118, 224, 243, 0.1);
  border-radius: 10px;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s; /* 添加过渡效果 */

}
.user-func-container:hover {
  transform: scale(1.02);
}

.btn-predict, .btn-logout {
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.btn-predict:hover {
  background-color: #4caf50; /* 鼠标悬停时的颜色 */
}

.btn-logout:hover {
  background-color: #f44336; /* 鼠标悬停时的颜色 */
}
</style>
