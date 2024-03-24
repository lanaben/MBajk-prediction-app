# Bike Stands Monitoring System

This project offers a real-time monitoring solution for bike stands, featuring a Vue.js frontend and a Python Flask backend. Designed for ease of use and efficient data handling, this system provides valuable insights into bike stand availability and usage patterns.

## Getting Started

Follow these instructions to set up the project on your local machine for development and testing purposes.

## Prerequisites

Ensure you have Docker, Docker Compose, and Git installed on your system.

## Installation

### Clone the Repository

git clone https://github.com/lanaben/MBajk-prediction-app.git

### Build and Run with Docker Compose

This command builds Docker images for the frontend and backend and starts the containers:

docker-compose up --build 

Access the frontend at http://localhost:8080 and the backend API at http://localhost:5000.

### Backend API Endpoints

Fetch stations: GET /mbajk/stations

Get last values of data for specific station: GET /mbajk/<station_name>/<limit>

Predict availability: POST /mbajk/predict/<station_name>
