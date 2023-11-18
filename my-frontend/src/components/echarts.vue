<template>
    <div ref="myRef" id="demoDiv">

    </div>
</template>
<script>
import {ref,onMounted} from 'vue'
import *as echarts from "echarts";
export default{
    props:{
        option:Object
    },
    setup(props){
        const myRef = ref<any>(null)
        onMounted(()=>{
            setTimeout(()=>{
                drawChart()
            },20)
        })
        const drawChart=()=>{
            const Chart = echarts.init(myRef.value);
            Chart.setOption(props.option)
            window.addEventListener("resize",()=>{
                Chart.resize();
            })
        }
        return{
            myRef,
            drawChart
        }
    },
}
</script>


<style>
 #demoDiv{
    width:500px;
    height:500px;
    border:1px solid red;
 }
</style>