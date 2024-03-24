<template>
  <div class="dialog">
    <h2>Predictions for {{ station }}</h2>
    <ul v-if="predictions.length">
      <li v-for="(prediction, index) in predictions" :key="`prediction-${index}`">
        Prediction {{ index + 1 }}: {{ Math.round(prediction) }}
      </li>
    </ul>
    <button @click="$emit('close')">Close</button>
  </div>
</template>

<script lang="ts" setup>
import axios from "axios";

const apiUrl = "http://localhost:5000/mbajk";

import { ref, watch, defineProps } from 'vue';

const props = defineProps({
  station: String
});

const stationData = ref<number[]>([]);
const predictions = ref<number[]>([]);

watch(() => props.station, async (newStation) => {
  if (newStation) {
    stationData.value = await fetchStationData(newStation,7);
    if (stationData.value && stationData.value.length > 0) {
      predictions.value = await predictStationData(newStation, stationData.value);
    }
  }
}, { immediate: true });

async function fetchStationData(stationName: string, limit: number) {
  try {
    const response = await axios.get(`${apiUrl}/${encodeURIComponent(stationName)}/${limit}`);
    return response.data.data;
  } catch (error) {
    console.error("There was an error fetching the station data:", error);
  }
}

async function predictStationData(stationName: string, data: number[]) {
  try {
    const response = await axios.post(`${apiUrl}/predict/${encodeURIComponent(stationName)}`, {
      data: data
    });
    return response.data.prediction;
  } catch (error) {
    console.error("There was an error making a prediction:", error);
  }
}
</script>

<style scoped>

</style>