<script setup>
  import file_input from './component/file_input.vue';
  import net_display from './component/net_display.vue';
  import module_select from './component/module_select.vue';
  import arguments_container from './component/arguments_container.vue';
</script>

<template>
  <div class="container">
    <div class="top">
      <div class="left">
        <module_select/>
      </div>
      <div class="center">
        <net_display/>
      </div>
      <div class="right"></div>
    </div>
    <div class="bottom">
      <div class="left">
        <file_input/>
      </div>
      <div class="center">
        <arguments_container/>
      </div>
      <div class="right"></div>
    </div>
  </div>
</template>

<style scoped>
  .container{
    width: 100%;
    height: 100vh;
    background-color: #eee;
  }
  .container .top{
    width: 100%;
    height: 65%;
    background-color: aquamarine;
    display: flex;
  }
  .container .bottom{
    width: 100%;
    height: 35%;
    background-color: bisque;
    display: flex;
  }
  .container .left
  {
    height: 100%;
    flex: 2;
    background-color: azure;
  }
  .container .center
  {
    height: 100%;
    flex: 6;
    background-color: antiquewhite;
  }
  .container .right
  {
    height: 100%;
    flex: 2;
    background-color: aqua;
  }
</style>
