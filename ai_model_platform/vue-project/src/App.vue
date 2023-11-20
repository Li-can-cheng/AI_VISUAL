<script setup>
import { provide } from "vue";
import * as echarts from "echarts";
provide("echarts", echarts);
import file_input from './component/file_input.vue';
import net_display from './component/net_display.vue';
import module_select from './component/module_select.vue';
import arguments_container from './component/arguments_container.vue';
import echarts_container from './component/echarts.vue';

</script>

<template>
  <div class="container">
    <div class="left-container">
      <div class="top">
        <div class="left">
          <module_select />
        </div>
        <div class="center">
          <net_display />
        </div>
      </div>
      <div class="bottom">
        <div class="left">
          <file_input />
        </div>
        <div class="center">
          <arguments_container />
        </div>
      </div>
    </div>
    <div class="right">
      <echarts_container />
    </div>
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
  height: 100%;
  padding: 15px;
  box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
}

.container .top {
  height: 65%;
  /* background-color: #63ccc2; */
  border:6px solid rgba(148, 238, 243,0.5);
  display: flex;
  border-radius: 10px;
  color: #fff;
  padding: 10px;
  box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
  transition: transform 0.3s;
}

.container .top:hover {
  transform: scale(1.02);
}

.container .bottom {
  height: 35%;
  border:6px solid rgba(148, 238, 243,0.5);
  /* background-color: rgb(223, 253, 255); */
  box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
  display: flex;
  border-radius: 10px;
  color: #333;
  margin-top: 15px;
  padding: 15px;
  box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
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
  height: 97%;
  margin-top:0.9%;
  margin-right: 0.4%;
  width:20%;
  width:50px;
  overflow-y: auto;
  flex: 1;
  border:6px solid rgba(148, 238, 243,0.5);
  background-color: rgba(118, 224, 243,0.1);  
  padding: 10px;
  border-radius: 10px;
  box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
}
</style>


