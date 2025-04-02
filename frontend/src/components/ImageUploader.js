import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, CircularProgress, Alert } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import axios from 'axios';

const ImageUploader = ({ onResults, onError, onLoading }) => {
  const [preview, setPreview] = useState(null);
  const [uploadError, setUploadError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Reset error state
    setUploadError(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result);
    reader.readAsDataURL(file);

    // Prepare form data
    const formData = new FormData();
    formData.append('image', file);

    try {
      onLoading(true);
      const response = await axios.post('http://localhost:5000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Process the results
      const results = response.data;
      
      // Validate the results format
      if (typeof results !== 'object' || Object.keys(results).length === 0) {
        throw new Error('Invalid response format from the server');
      }

      // Check if the response contains an error
      if (results.error) {
        throw new Error(results.error + (results.details ? `\nDetails: ${results.details}` : ''));
      }

      // Convert string values to numbers if needed and ensure all values are between 0 and 1
      const processedResults = Object.entries(results).reduce((acc, [key, value]) => {
        const numValue = typeof value === 'string' ? parseFloat(value) : value;
        acc[key] = isNaN(numValue) ? 0 : Math.min(Math.max(numValue, 0), 1);
        return acc;
      }, {});

      onResults(processedResults);
    } catch (error) {
      let errorMessage = 'Error analyzing image. Please try again.';
      let details = '';

      if (error.response) {
        // Server responded with an error
        errorMessage = error.response.data?.error || errorMessage;
        details = error.response.data?.details || '';
      } else if (error.request) {
        // Request was made but no response received
        errorMessage = 'No response from server. Please check if the server is running.';
      } else {
        // Error in request setup
        errorMessage = error.message || errorMessage;
      }

      const fullError = details ? `${errorMessage}\nDetails: ${details}` : errorMessage;
      setUploadError(fullError);
      onError(fullError);
    } finally {
      onLoading(false);
    }
  }, [onResults, onError, onLoading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false,
    maxSize: 5242880, // 5MB
  });

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      gap: 3,
      height: '100%',
      overflow: 'auto'
    }}>
      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          borderRadius: 2,
          p: 4,
          cursor: 'pointer',
          '&:hover': {
            borderColor: 'primary.main',
            bgcolor: 'action.hover',
          },
          bgcolor: isDragActive ? 'action.hover' : 'background.paper',
          transition: 'all 0.3s ease',
          flexShrink: 0
        }}
      >
        <input {...getInputProps()} />
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            {isDragActive
              ? 'Drop the image here'
              : 'Drag and drop an image here'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            or click to select
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
            Supported formats: JPEG, JPG, PNG (max 5MB)
          </Typography>
        </Box>
      </Box>

      {uploadError && (
        <Alert 
          severity="error" 
          sx={{ 
            width: '100%',
            '& .MuiAlert-message': { whiteSpace: 'pre-line' },
            flexShrink: 0
          }}
        >
          {uploadError}
        </Alert>
      )}

      {preview && (
        <Box sx={{ 
          width: '100%', 
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          <Typography variant="h6" gutterBottom sx={{ flexShrink: 0 }}>
            Preview
          </Typography>
          <Box
            component="img"
            src={preview}
            alt="Preview"
            sx={{
              width: '100%',
              height: 'auto',
              maxHeight: '100%',
              objectFit: 'contain',
              borderRadius: 1,
              boxShadow: 1,
              display: 'block',
              margin: '0 auto',
              flexGrow: 1
            }}
          />
        </Box>
      )}
    </Box>
  );
};

export default ImageUploader; 