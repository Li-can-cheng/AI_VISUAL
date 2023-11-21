<script setup>
import {provide} from "vue";
import * as echarts from "echarts";
import file_input from './component/file_input.vue';
import net_display from './component/net_display.vue';
import module_select from './component/module_select.vue';
import arguments_container from './component/arguments_container.vue';
import echarts_container from './component/echarts.vue';
import LogDisplay from './component/yuuki.vue';
import user_func from './component/yuuki_userfunc.vue';

provide("echarts", echarts);
</script>

<template>
  <div class="container">
    <div class="left-container">
      <div class="top">
        <div class="left">
          <module_select/>
        </div>
        <div class="center">
          <net_display/>
        </div>
      </div>
      <div class="bottom">
        <div class="left">
          <file_input/>

        </div>
        <div class="center">
          <arguments_container/>
        </div>
      </div>
    </div>
    <div class="right">
      <echarts_container/>
    </div>
  </div>

  <div class="super-bottom">
    <div class="log-display-container">
      <LogDisplay/>
    </div>
      <user_func/>
  </div>

</template>


<style scoped>
.container {
  display: flex;
  width: 100%;
  height: 100vh;
  /* background-color: #f5f5f5; */
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.left-container {
  display: flex;
  flex-direction: column;
  width: 78%;
  height: 85%;
  padding: 15px;
//box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
}

.container .top {
  height: 65%;
  /* background-color: #63ccc2; */
  border: 6px solid rgba(148, 238, 243, 0.5);
  display: flex;
  border-radius: 10px;
  color: #fff;
  padding: 10px;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s;
}

.container .top:hover {
  transform: scale(1.02);
}

.container .bottom {
  height: 35%;
  border: 6px solid rgba(148, 238, 243, 0.5);
  /* background-color: rgb(223, 253, 255); */
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
  display: flex;
  border-radius: 10px;
  color: #333;
  margin-top: 15px;
  padding: 15px;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s;
}

.container .bottom:hover {
  transform: scale(1.02);
}

.container .left, .container .center {
  height: 100%;
  border-radius: 10px;
}

.container .left {
  flex: 2;
  /* background-color: #bae9ef; */
  /* background-color: rgba(159, 193, 173,0.5); */
}

.container .center {
  flex: 6;
  margin-left: 10px;
}

.container .right {
  height: 90%;
  margin-top: 0.9%;
  margin-right: 0.4%;
  overflow-y: auto;
  flex: 1;
  border: 6px solid rgba(148, 238, 243, 0.5);
  background-color: rgba(118, 224, 243, 0.1);
  padding: 10px;
  border-radius: 10px;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s; /* 添加过渡效果 */
}

.container .right:hover {
  transform: scale(1.02);
}

.log-display-container {
  /* 水平居中，左边距自动 */
  /* 水平居中，右边距自动 */
  margin: 50px;
  margin-left: 15px;
  margin-right:15px;
  width: 71%; /* 调整宽度 */
  height: 200px; /* 调整高度 */
  padding: 40px; /* 调整内边距 */
  color: blue; /* 白色文本 */
  border: 6px solid rgba(148, 238, 243, 0.5);
  background-color: rgba(118, 224, 243, 0.1);
  border-radius: 10px;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s; /* 添加过渡效果 */
}


.super-bottom {
  display: flex;

}


</style>


