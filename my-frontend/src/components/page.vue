<script lang="ts">
    import { ref } from 'vue'
    import type { UploadProps, UploadUserFile } from 'element-plus'
    import draggable from 'vuedraggable'
    import upload from './upload.vue'
    import axios from 'axios'
    export default{
        name:'two-lists',
        display:'Two Lists',
        order:1,
        components:{
            draggable,
            upload
        },
        data(){
            return{
                list1:[
                    {name:"CNN模型",id:"cnn"},
                    {name:"神经网络",id:"MLP"},
                    {name:"数据预处理",id:"pre-process"},
                    {name:"模型评估",id:"evaluation"},
                    {name:"支持向量机",id:"svm"},
                ],
                list2:[

                ],
            };
        },
        methods:{
            getId:function(itemid){
                return itemid;
            },
            // add:function(){
            //     this.list1.push({name:"Juan"});
            // },
            // replace:function(){
            //     this.list = [{name:"Edgard"}];
            // },
            // clone:function(el){
            //     return{
            //         name:el.name+"cloned"
            //     };
            // },
            log:function(evt){
                window.console.log(evt.added.element.id);
                // window.console.log(evt.added);
                var name = evt.added.element.id;
                if(name == 'MLP')
                {
                    const data = {
                        "name":"MLP",
                        "arguments":{
                        "epochs":-1,
                        "layers":{
                        "linear1":256,
                        "sigmoid":-1,
                        "ReLU1":-1,
                        "linear2":128,
                        "ReLU2":-1,
                        "linear3":10
                    }
                    }
                    }
                    axios.post('http://localhost:8080/model/MLP',data,{
                        headers:{
                            'Content-Type':'application/json'
                        }
                    }).then(response=>{
                        // console.log(response)
                        console.log(response)
                    }).catch(error=>{
                        console.log(error)
                    })
                }
                // var element = document.getElementById(evt.added.element.id)
                // var ee = document.getElementsByClassName("list-group-item")
                // var ee = evt.added.element.id
                // alert(ee)
                // var ele = document.getElementsByClassName("header")
                // console.log(ee)
                var ee =document.getElementById(" ")
            }
        },
        setup(){
            const active = ref(0)
            const next = ()=>{
                if(active.value++>4) active.value=0
            }
            const activeIndex = ref('1')
            const handleSelect = (key: string, keyPath: string[]) => {
                console.log(key, keyPath)
            }
            return{
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
                            <div class="list-group-item" :id="getId(element.id)">
                                {{ element.name }}
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
                    <el-step title="Step4" />
                    <el-button style="margin-top: 0px;width:80px;height:30px;margin-left:-36px;margin-right:20px;" @click="next">Next step</el-button>
                </el-steps>
            </div>
          <div class="col-3">
            <draggable class="list-group viewable" :list="list2" group="people" @change="log" itemKey="name">
                <template #item="{element,index}">
                    <div class="list-group-item">
                        {{ element.name }}
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
            <div class="upload-content">
                <upload></upload>
            </div>
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
    width:20%;
    height: 100%;
    background-color: rgb(225, 250, 242);
}
.result .upload .upload-title{
    height: 30px;
    /* width:100%; */
    line-height: 30px;
    text-align: center;
    font-size: 16px;
    color:rgb(87, 87, 87);
    border-radius: 2px;
    background-color: rgb(193, 231, 219);  
    box-shadow: 0px 1px 2px rgb(121, 121, 121);
}
.result .model-parameter{
    width:180%;
    height: 100%;
    background-color: rgb(157, 206, 190);
}
.result .result-parameter{
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