<template>
    <div ref="myChart" id="myChart">

    </div>
</template>
<script>
import * as echarts from "echarts"
export default{
    mounted(){
        // echarts
        let myEcharts = echarts.init(this.$refs.myChart)
        let xData = ['m','t','w','t','fd','as','su']
        let data =[13,2,3,25,44,3,6]
        let option={
            xAxis:{
                type:"category",
                data:xData
            },
            yAxis:{
                type:"value"
            },
            series:[
                {
                    data,
                    type:"line"
                }
            ]
        };
        myEcharts.setOption()
    }
}
</script>
<style>
#myChart{
    width:500px;
    height:500px;
    border:1px solid red;
}
</style>