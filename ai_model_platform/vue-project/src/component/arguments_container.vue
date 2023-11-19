<script setup>

import MPL from './arguments_container/model/MPL.vue';
import linear from './arguments_container/linear/linear.vue';

import {ref} from "vue";
import {useDataStore} from "../store/data";

const data_store = useDataStore();

</script>

<template>
    <div class="arguments_container">
        <MPL/>
        <linear/>
    </div>
</template>

<style>
.arguments_container{
    width: 100%;
    height: 100%;
    display: flex;
}






</style>