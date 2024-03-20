import requests

latitude = 46.234818
longitude = 15.267305
start_date = '2023-02-15'
end_date = '2023-12-11'
api_url = 'https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current=temperature_2m,relative_humidity_2m,apparent_temperature'



response = requests.get(api_url)
api_data = response.json()
print(api_data)
