import './assets/main.css'
import { createApp } from 'vue'
import { createPinia } from "pinia";
import App from './App.vue'
import * as echarts from 'echarts';

let app = createApp(App)
let pinia = createPinia();

app.use(pinia);
app.provide('echarts', echarts);
app.mount('#app');
