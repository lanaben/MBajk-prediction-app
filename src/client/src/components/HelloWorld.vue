<template>
  <div>
    <h1>Stations</h1>
    <ul>
      <li v-for="station in stations" :key="station.id">{{ station.name }}</li>
    </ul>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return {
      stations: []
    }
  },
  created() {
    this.fetchStations();
  },
  methods: {
    async fetchStations() {
      try {
        const response = await axios.get('http://localhost:5000/mbajk/stations');
        this.stations = response.data.Stations;
      } catch (error) {
        console.error("There was an error fetching the stations:", error);
      }
    }
  }
}
</script>


<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
