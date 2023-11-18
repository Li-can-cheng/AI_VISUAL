
import { createApp } from 'vue'
import App from './App.vue'
import draggable from 'vuedraggable'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
// 引入Echarts
import echarts from "echarts"
Vue.prototype.$echarts = echarts
const app = createApp(App)
app.use(ElementPlus)
app.mount('#app')
export default{
    components:{
        draggable
    }
}
