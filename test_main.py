import subprocess
import json
import os

def test_main_output():
    # Create a test image path (use a real image if available)
    test_image = "test_image.jpg"
    
    # Check if the test image exists
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found. Please provide a valid image path.")
        return
    
    # Run main.py with the test image
    result = subprocess.run(['python', 'main.py', test_image], 
                          capture_output=True, 
                          text=True)
    
    # Print the raw output
    print(f"Return code: {result.returncode}")
    print(f"Raw stdout: '{result.stdout}'")
    print(f"Raw stderr: '{result.stderr}'")
    
    # Try to parse the JSON output
    try:
        stdout = result.stdout.strip()
        if not stdout:
            print("No output from main.py")
            return
        
        analysis_results = json.loads(stdout)
        print("Successfully parsed JSON output:")
        print(json.dumps(analysis_results, indent=2))
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw output: '{stdout}'")

if __name__ == "__main__":
    test_main_output() 