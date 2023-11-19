<script setup>
import { provide } from "vue";
import * as echarts from "echarts";
provide("echarts", echarts);
import file_input from './file_input.vue';
import net_display from './net_display.vue';
import module_select from './module_select.vue';
import arguments_container from './arguments_container.vue';
import echarts_container from './echarts.vue';

</script>

<template>
  <div class="container">
    <div class="top">
      <div class="left">
        <module_select />
      </div>
      <div class="center">
        <net_display />
      </div>
      <div class="right">
        <echarts_container />
      </div>
    </div>
    <div class="bottom">
      <div class="left">
        <file_input />
      </div>
      <div class="center">
        <arguments_container />
      </div>
      <div class="right">

      </div>
    </div>
  </div>
</template>

<style scoped>
.container {
  width: 100%;
  height: 100vh;
  background-color: #eee;
}

.container .top {
  width: 100%;
  height: 65%;
  background-color: aquamarine;
  display: flex;
}

.container .bottom {
  width: 100%;
  height: 35%;
  background-color: bisque;
  display: flex;
}

.container .left {
  height: 100%;
  flex: 2;
  background-color: azure;
}

.container .center {
  height: 100%;
  flex: 6;
  background-color: antiquewhite;
}

.container .right {
  height: 100%;
  flex: 2;
  background-color: aqua;
}
</style>
