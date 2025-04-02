import os
import sys
from backend import app

if __name__ == "__main__":
    # Set environment variables for debugging
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 