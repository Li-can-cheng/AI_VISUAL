<script lang="ts">
    import { ref } from 'vue'
    import type { UploadProps, UploadUserFile } from 'element-plus'
    import draggable from 'vuedraggable'
    export default{
        name:'two-lists',
        display:'Two Lists',
        order:1,
        components:{
            draggable
        },
        data(){
            return{
                list1:[
                    {name:"John",id:1},
                    {name:"Joao",id:2},
                    {name:"Jean",id:3},
                    {name:"Gerard",id:4},
                    {name:"Edgard",id:6},
                    {name:"Johnson",id:7},
                ],
                list2:[
                    {name:"Juan",id:5},
                ],
            };
        },
        methods:{
            add:function(){
                this.list1.push({name:"Juan"});
            },
            replace:function(){
                this.list = [{name:"Edgard"}];
            },
            clone:function(el){
                return{
                    name:el.name+"cloned"
                };
            },
            log:function(evt){
                window.console.log(evt); //什么意思？？不懂
            }
        },
        setup(){
            const fileList = ref<UploadUserFile[]>([
            {
            name: 'food.jpeg',
            url: 'https://fuss10.elemecdn.com/3/63/4e7f3a15429bfda99bce42a18cdd1jpeg.jpeg?imageMogr2/thumbnail/360x360/format/webp/quality/100',
            },
            {
            name: 'food2.jpeg',
            url: 'https://fuss10.elemecdn.com/3/63/4e7f3a15429bfda99bce42a18cdd1jpeg.jpeg?imageMogr2/thumbnail/360x360/format/webp/quality/100',
            },
            ])
            const handleChange: UploadProps['onChange'] = (uploadFile, uploadFiles) => {
                fileList.value = fileList.value.slice(-3)
            }
            const active = ref(0)
            const next = ()=>{
                if(active.value++>4) active.value=0
            }
            const activeIndex = ref('1')
            const handleSelect = (key: string, keyPath: string[]) => {
                console.log(key, keyPath)
            }
            return{
                fileList,
                handleChange,
                next,
                active,
                activeIndex,
                handleSelect
            }
        }
    }
</script>
<template>
<div class="front-page">
    <div class="header">
        <el-menu
            :default-active="activeIndex"
            class="el-menu-demo"
            mode="horizontal"
            :ellipsis="false"
            @select="handleSelect"
        >
      <el-menu-item index="0">
        <img
          style="width: 30px;height:30px;"
          src="../assets/image/人工智能.png"
          alt="Element logo"
        />
      </el-menu-item>
      <div class="flex-grow" />
      <el-menu-item index="1">AI智能化模型训练平台</el-menu-item>
      <el-sub-menu index="2">
        <template #title>工作区</template>
        <el-menu-item index="2-1">item one</el-menu-item>
        <el-menu-item index="2-2">item two</el-menu-item>
        <el-menu-item index="2-3">item three</el-menu-item>
        <el-sub-menu index="2-4">
          <template #title>item four</template>
          <el-menu-item index="2-4-1">item one</el-menu-item>
          <el-menu-item index="2-4-2">item two</el-menu-item>
          <el-menu-item index="2-4-3">item three</el-menu-item>
        </el-sub-menu>
      </el-sub-menu>
    </el-menu>
    </div>
    <div class="designer">
        <div class="AI-model">
            <div class="panel-tab">
                <div class="panel-tab-title">AI模型列表</div>
            </div>
            <div class="panel-list">
                <div class="col-2">
                    <draggable class="list-group" :list="list1" group="people" @change="log" itemKey="name">
                        <template #item="{element,index}">
                            <div class="list-group-item">
                                {{ element.name }}{{ index }}
                            </div>
                        </template>
                    </draggable>
                </div>
            </div>
        </div>
        <div class="process-interface">
            <!-- 进度条 -->
            <div class="progress-bar" style="height:36px;">
                <el-steps :active="active" finish-status="success" align-center>
                    <el-step title="Step1" />
                    <el-step title="Step2" />
                    <el-step title="Step3" />
                    <el-button style="margin-top: 0px;width:80px;height:30px;margin-left:-36px;margin-right:20px;" @click="next">Next step</el-button>
                </el-steps>
            </div>
          <div class="col-3">
            <draggable class="list-group" :list="list2" group="people" @change="log" itemKey="name">
                <template #item="{element,index}">
                    <div class="list-group-item">
                        {{ element.name }}{{ index }}
                    </div>
                </template>
            </draggable>
          </div>
        </div>
        <div class="visual-result">
            <p>ni</p>
        </div>
    </div>
    <div class="result">
        <div class="upload">
            <div class="upload-title">上传文件</div>
            <el-upload
                v-model:file-list="fileList"
                class="upload-demo"
                action="https://run.mocky.io/v3/9d059bf9-4660-45f2-925d-ce80ad6c4d15"
                :on-change="handleChange"
            >
                <el-button type="primary">Click to upload</el-button>
                <template #tip>
                    <div class="el-upload__tip">
                    jpg/png files with a size less than 500kb
                    </div>
                </template>
            </el-upload>
        </div>
        <div class="model-parameter">

        </div>
        <div class="result-parameter">
            
        </div>
    </div>
</div>
</template>
<style scoped>
*{
    margin:0px;
    width:100%;
    padding:0px;
}
.front-page{
    width:100%;
    background-color:antiquewhite;
    height:730px;
    border-radius: 10px;
    box-shadow: 0px 1px 2px rgb(121, 121, 121);
}
.header{
    width:100%;
    height:8%;
    background-color: antiquewhite;
    /* box-shadow: 3px 3px 3px rgb(83, 82, 82); */
}
.el-menu-demo{
    box-shadow: 1px 1px 1px rgba(197, 197, 197,0.8);
}
.designer{
    margin-top: 2px;
    width:100%;
    height:60%;
    background-color: aqua;
    display: flex;
    flex-direction: row;
}
.designer .AI-model{
    width:70%;
    height: 100%;
    background-color: rgb(255, 255, 255);
}
.AI-model .panel-tab .panel-tab-title{
    height: 30px;
    width:100%;
    line-height: 30px;
    text-align: center;
    font-size: 16px;
    color:rgb(87, 87, 87);
    border-radius: 2px;
    box-shadow: 0px 1px 2px rgb(121, 121, 121);
    margin-bottom: 2px;       
    background-color: rgb(193, 231, 219);  
}
.designer .process-interface{
    width:180%;
    height: 100%;
    background-color: rgb(216, 251, 239);
}
.designer .visual-result{
    /* width:180%; */
    height: 100%;
    background-color: rgb(114, 165, 148);
}
.result{
    width:100%;
    height:32.3%;
    display: flex;
    flex-direction: row;
}
.result .upload{
    width:70%;
    height: 100%;
    background-color: rgb(225, 250, 242);
}
.result .upload .upload-title{
    height: 30px;
    width:100%;
    line-height: 30px;
    text-align: center;
    font-size: 16px;
    color:rgb(87, 87, 87);
    border-radius: 2px;
    background-color: rgb(193, 231, 219);  
    box-shadow: 0px 1px 2px rgb(121, 121, 121);
}
.model-parameter{
    width:180%;
    height: 100%;
    background-color: rgb(157, 206, 190);
}
.result-parameter{
    height: 100%;
    background-color: rgb(82, 143, 122);
}
.col-3{
    margin-top: 28px;
    width:100%;
    /* height:80%; */
    margin-bottom: 10px;
    /* height:400px; */
    /* background-color: aqua; */
}
.col-2{
    width:100%;
    /* background-color: rgb(175, 226, 226); */
}
.list-group{
    overflow-y: scroll;
    overflow-x: hidden;
    height:419px;
}
.list-group-item{
    width:100%;
    background-color: rgb(196, 218, 218);
    height:60px;
    margin-bottom: 3px;
    /* border: 1px solid black; */
    box-shadow: 2px 2px 2px rgb(222, 222, 222);
}
.progress-bar{
    width:100%;
    height:62px;
    margin-top: 15px;
}
.flex-grow {
    flex-grow: 1;
  }
</style>