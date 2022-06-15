import requests

response = requests.get("http://127.0.0.1:5000/hello")
print(response)
requests.post("http://127.0.0.1:5000/hello",data ={'sentence':'Hello there'})
# print(response.content())
# print(response.text())
#print(response.json())