import requests

url = "http://localhost:80/predict"
data = {"Input": [5.1, 3.5, 1.4, 0.2]}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Prediction:", response.json())
