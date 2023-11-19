<script setup>
import file_container from "./net_display_component/file_container.vue";
import data_processing_container from "./net_display_component/data_processing_container.vue";
import net_container from "./net_display_component/net_container.vue";

</script>
<template>
    <div class="net_display_container">
        <file_container/>
        <data_processing_container/>
        <net_container/>
    </div>
</template>
<style>
.net_display_container {
    width: 100%;
    height: 100%;
    display: flex;
}
</style>