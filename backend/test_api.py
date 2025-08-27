
import requests

# URL of Flask backend
url = "http://127.0.0.1:5000/classify"

# Replace with any sample image path from your dataset
files = {"file": open("dataset/Jewelery/Ring/ring_003.jpg", "rb")}

response = requests.post(url, files=files)

print("API Response:")
print(response.json())
