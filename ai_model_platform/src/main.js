import './assets/main.css'

import { createApp } from 'vue'
import {createPinia} from "pinia";
import App from './App.vue'

let app = createApp(App)
let pinia = createPinia();

app.use(pinia);
app.mount('#app')
