import requests

# Define the URL of your Flask application
url = 'http://127.0.0.1:80/predict'  # Update the URL if necessary

# Sample input data
data = {'input': ['the movie was one of the worst movies i have ever seen. Couldnt believe it was all so bad.']}

# Make the POST request
response = requests.post(url, json=data)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the prediction from the response
    prediction = response.json()['prediction']
    print('Prediction:', prediction)
else:
    print('Error:', response.text)