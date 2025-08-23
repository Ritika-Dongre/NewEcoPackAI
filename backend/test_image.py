import requests

# Flask server URL
URL = "http://127.0.0.1:5000/classify"

# Path to test image
IMAGE_PATH = IMAGE_PATH = "dataset/books_stationery/crayons.jpg"

# Send request
with open(IMAGE_PATH, "rb") as f:
    files = {"image": f}
    response = requests.post(URL, files=files)

# Print result
print(" Response from API:")
print(response.json())
