import React, { useState } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  ThemeProvider, 
  createTheme,
  CssBaseline,
  Paper,
  Alert,
  Grid,
  Divider,
  Tabs,
  Tab
} from '@mui/material';
import ImageUploader from './components/ImageUploader';
import ResultsDisplay from './components/ResultsDisplay';
import Header from './components/Header';
import ModelMetrics from './components/ModelMetrics';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
  },
});

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [rightTabValue, setRightTabValue] = useState(0);

  const handleResults = (data) => {
    if (typeof data === 'string') {
      try {
        // Try to parse if the data is a JSON string
        const parsedData = JSON.parse(data);
        setResults(parsedData);
        setError(null);
      } catch (e) {
        setError('Invalid results format received');
        setResults(null);
      }
    } else {
      setResults(data);
      setError(null);
    }
  };

  const handleError = (err) => {
    setError(err);
    setResults(null);
  };

  const handleLoading = (isLoading) => {
    setLoading(isLoading);
    if (isLoading) {
      setError(null);
    }
  };

  const handleRightTabChange = (event, newValue) => {
    setRightTabValue(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', display: 'flex', flexDirection: 'column' }}>
        <Header />
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ 
            display: 'flex', 
            flexDirection: { xs: 'column', md: 'row' },
            flexGrow: 1,
            height: { md: 'calc(100vh - 120px)' }
          }}>
            {/* Left side - Upload and Preview (40%) */}
            <Box sx={{ 
              width: { md: '40%' }, 
              p: 2,
              height: { xs: 'auto', md: '100%' }
            }}>
              <Paper 
                elevation={3} 
                sx={{ 
                  p: 3,
                  borderRadius: 2,
                  bgcolor: 'background.paper',
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column'
                }}
              >
                <ImageUploader 
                  onResults={handleResults}
                  onError={handleError}
                  onLoading={handleLoading}
                />
                
                {error && (
                  <Box sx={{ mt: 3 }}>
                    <Alert 
                      severity="error" 
                      sx={{ 
                        borderRadius: 1,
                        '& .MuiAlert-message': {
                          width: '100%'
                        }
                      }}
                    >
                      <Typography variant="body1" gutterBottom>
                        {error}
                      </Typography>
                      <Typography variant="body2">
                        Please try uploading a different image or contact support if the problem persists.
                      </Typography>
                    </Alert>
                  </Box>
                )}
              </Paper>
            </Box>
            
            {/* Vertical divider */}
            <Divider orientation="vertical" flexItem sx={{ display: { xs: 'none', md: 'block' } }} />
            
            {/* Right side - Results (60%) */}
            <Box sx={{ 
              width: { md: '60%' }, 
              p: 2,
              height: { xs: 'auto', md: '100%' }
            }}>
              <Paper 
                elevation={3} 
                sx={{ 
                  p: 3,
                  borderRadius: 2,
                  bgcolor: 'background.paper',
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'auto'
                }}
              >
                <Tabs 
                  value={rightTabValue} 
                  onChange={handleRightTabChange} 
                  sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
                >
                  <Tab label="Model Performance" />
                  <Tab label="Analysis Results" disabled={!results && !loading} />
                </Tabs>

                {rightTabValue === 0 ? (
                  <ModelMetrics />
                ) : (
                  loading ? (
                    <Box sx={{ 
                      display: 'flex', 
                      justifyContent: 'center', 
                      alignItems: 'center',
                      height: '100%'
                    }}>
                      <Typography variant="h6">Analyzing image...</Typography>
                    </Box>
                  ) : results ? (
                    <ResultsDisplay results={results} />
                  ) : (
                    <Box sx={{ 
                      display: 'flex', 
                      justifyContent: 'center', 
                      alignItems: 'center',
                      height: '100%'
                    }}>
                      <Typography variant="h6" color="text.secondary">
                        Upload an image to see analysis results
                      </Typography>
                    </Box>
                  )
                )}
              </Paper>
            </Box>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
