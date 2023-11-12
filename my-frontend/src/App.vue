<script setup>
import Page from './components/page.vue'
import Draggable from './components/draggable.vue';
import upload from './components/upload.vue'
import Header from './components/Header.vue'
import send from './components/send.vue'
import drag from './components/drag.vue'
</script>

<template>
  <drag></drag>
  <!-- <Page/> -->
  <!-- <Draggable></Draggable> -->
  
  <upload></upload>
  <br/>
  <br/>
  <send></send>
  <!-- <Header></Header> -->
</template>
<style>
*{
  margin:0px;
  padding:0px;
}
</style>