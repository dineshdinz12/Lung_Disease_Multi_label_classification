import requests
import json
import os

def test_backend_connection():
    # Create a test image path (use a real image if available)
    test_image = "test_image.jpg"
    
    # Check if the test image exists
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found. Please provide a valid image path.")
        return
    
    # Prepare form data
    files = {'image': open(test_image, 'rb')}
    
    try:
        # Send request to the backend
        response = requests.post('http://localhost:5000/analyze', files=files)
        
        # Print the response status code
        print(f"Response status code: {response.status_code}")
        
        # Print the response headers
        print(f"Response headers: {response.headers}")
        
        # Try to parse the JSON response
        try:
            data = response.json()
            print("Successfully parsed JSON response:")
            print(json.dumps(data, indent=2))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: '{response.text}'")
    except Exception as e:
        print(f"Error sending request: {e}")

if __name__ == "__main__":
    test_backend_connection() 