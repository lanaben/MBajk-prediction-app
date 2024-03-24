<template>
  <div id="app">
    <img alt="Vue logo" src="./assets/logo.png">
    <h1>Welcome to Your Vue.js App</h1>
    <ul>
      <li v-for="station in stations" :key="station" @click="selectStation(station)">
        {{ station }}
      </li>
    </ul>
    <PredictionsDialog v-if="selectedStation" :station="selectedStation" @close="selectedStation = null"/>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue';
import axios from "axios";
import PredictionsDialog from './components/PredictionsDialog.vue';

const apiUrl = "http://localhost:5000/mbajk";
const stations = ref<string[]>([]);
const selectedStation = ref<string | null>(null);

const fetchStations = async () => {
  try {
    const response = await axios.get(`${apiUrl}/stations`);
    stations.value = response.data.Stations.map((station: any) => station.name);
  } catch (error) {
    console.error("There was an error fetching the stations:", error);
  }
};

const selectStation = (stationName: string) => {
  selectedStation.value = stationName;
};

onMounted(fetchStations);
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
