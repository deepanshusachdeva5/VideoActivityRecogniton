import requests
response = requests.get(
    "http://127.0.0.1:5000/recipe?ingredients= pasta tomato onion")
print(response.text)
