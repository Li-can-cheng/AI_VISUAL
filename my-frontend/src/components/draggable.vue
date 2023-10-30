<script>
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
                ],
                list2:[
                    {name:"Juan",id:5},
                    {name:"Edgard",id:6},
                    {name:"Johnson",id:7},
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
        }
    }
</script>
<template>
    <!-- <div>
        <h1>nihao</h1>
    </div> -->
<div class="row">
        <div class="col-2">
            <h3>Draggleable 1</h3>
            <draggable class="list-group" :list="list1" group="people" @change="log" itemKey="name">
                <template #item="{element,index}">
                    <div class="list-group-item">
                        {{ element.name }}{{ index }}
                    </div>
                </template>
            </draggable>
        </div>
        <div class="col-3">
            <h3>Draggleable 2</h3>
            <draggable class="list-group" :list="list2" group="people" @change="log" itemKey="name">
                <template #item="{element,index}">
                    <div class="list-group-item">
                        {{ element.name }}{{ index }}
                    </div>
                </template>
            </draggable>
        </div>
        <!-- <rawDisplayer class="col-2" :value="list1" title="List 1"/> -->
        <!-- <rawDisplayer class="col-3" :value="list2" title="List 2"/> -->
    </div>
</template>
<style scoped>
.row{
    display: flex;
    flex-direction: row;
    width:600px;

}
.col-3{
    width:300px;
    height:600px;
    background-color: aqua;
}
.col-2{
    width:300px;
    height:600px;
    background-color: rgb(175, 226, 226);
}
.list-group-item{
    width:100px;
    background-color: aqua;
    height:60px;
    border: 1px solid black;
}
</style>