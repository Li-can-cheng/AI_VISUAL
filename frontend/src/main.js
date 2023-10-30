// main.js
import { createApp } from 'vue';
import App from './App.vue';
import axios from 'axios';

const app = createApp(App);

// 将 axios 添加到 app 的原型链上，方便在组件里使用
app.config.globalProperties.$axios = axios;

app.mount('#app');
