<script setup>
import {ref} from "vue";
import {useFileStore} from "../../store/file";
import {useDataStore} from "../../store/data";

const file_container = ref();
const file_display = ref(false);
const file = ref();//file_container中的子元素

const file_store = useFileStore();
const data_store = useDataStore();


function drop_hander_file(e) {
  console.log("drop_file");
  //获取当前文件
  let current_file = file_store.get_current_file();
  //如果当前文件不存在，就返回
  if (!current_file) return;
  //当file_container拥有一个子元素的时候替换掉原来的文件 TODO:

  //如果存在就显示文件容器的子元素
  file_display.value = true;
  file.value.innerHTML = current_file.type;
  //并且把数据存放到json文件中去
  data_store.set_file_input({
    file: current_file.file
  });
}

function dragover_hander(e) {
  // console.log("dragover");
  // e.preventDefault();
}

/**
 * 用于文件节点的双击清除
 */
function delete_file() {
  //删除文件节点
  file_display.value = false;
  //清除json文件中的file
  data_store.set_file_input({
    file: null
  });
}

</script>
<template>
  <div class="file_container arrow" @dragover.prevent="dragover_hander" @drop.prevent="drop_hander_file">
    <div class="file" ref="file" v-show="file_display" @dblclick="delete_file"></div>
  </div>
</template>
<style>
.net_display_container .file_container {
  height: 100%;
  border-radius: 5px;
  flex: 1;
  background-color: rgba(148, 238, 243, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  color: white;
}

.net_display_container .file_container .file {
  width: 80px;
  height: 80px;
  text-align: center;
  line-height: 80px;
  border-radius: 20px;
  /* background-color: chartreuse; */
  /* 禁止用户选择文字，为了美观 */
  user-select: none;
  overflow: hidden;
  /*超出部分隐藏*/
  white-space: nowrap;
  /*禁止换行*/
  text-overflow: ellipsis;
  /*省略号*/
}
</style>