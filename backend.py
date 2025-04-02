from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import subprocess
import json
import traceback

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided. Please upload an image file.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file. Please choose an image to analyze.'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Run the Python script with the image path
            result = subprocess.run(['python', 'main.py', filepath], 
                                 capture_output=True, 
                                 text=True)
            
            # Debug: Print the raw output
            print(f"Raw stdout: '{result.stdout}'")
            print(f"Raw stderr: '{result.stderr}'")
            
            # Check if the script execution was successful
            if result.returncode != 0:
                error_message = result.stderr or 'Error running analysis script'
                return jsonify({'error': f'Analysis failed: {error_message}'}), 500
            
            # Parse the JSON output
            try:
                # Clean the output to ensure it's valid JSON
                stdout = result.stdout.strip()
                if not stdout:
                    return jsonify({'error': 'No output from analysis script'}), 500
                
                # Try to parse the JSON
                analysis_results = json.loads(stdout)
                
                # Check if the results contain an error
                if 'error' in analysis_results:
                    return jsonify(analysis_results), 500
                
                # Return the results
                return jsonify(analysis_results)
            except json.JSONDecodeError as e:
                # Log the raw output for debugging
                print(f"Raw output: '{stdout}'")
                error_details = str(e)
                return jsonify({
                    'error': 'Error parsing analysis results. The model output was not in the expected format.',
                    'details': error_details
                }), 500
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            return jsonify({
                'error': 'An unexpected error occurred during image analysis.',
                'details': str(e)
            }), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True) 