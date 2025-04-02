import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  LinearProgress,
  Chip,
  Divider,
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ResultsDisplay = ({ results }) => {
  // Extract the predictions from the results
  // Handle both direct predictions object and wrapped predictions
  const predictions = results.predictions || results;
  
  // Ensure we have valid data
  if (!predictions || typeof predictions !== 'object' || Object.keys(predictions).length === 0) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="error">
          No valid prediction data received
        </Typography>
      </Box>
    );
  }
  
  const chartData = {
    labels: Object.keys(predictions),
    datasets: [
      {
        label: 'Confidence Score',
        data: Object.values(predictions),
        backgroundColor: Object.values(predictions).map(value => 
          value > 0.3 ? 'rgba(75, 192, 192, 0.6)' : 'rgba(255, 99, 132, 0.6)'
        ),
        borderColor: Object.values(predictions).map(value => 
          value > 0.3 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'
        ),
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Disease Detection Confidence Scores',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        title: {
          display: true,
          text: 'Confidence Score'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Conditions'
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45
        }
      }
    },
  };

  const detectedConditions = Object.entries(predictions)
    .filter(([_, value]) => value > 0.3)
    .map(([key, value]) => ({
      name: key,
      confidence: value,
    }));

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%',
      overflow: 'auto'
    }}>
      <Grid container spacing={3}>
        {/* Summary Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Analysis Summary
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
              {detectedConditions.length > 0 ? (
                detectedConditions.map((condition) => (
                  <Chip
                    key={condition.name}
                    label={`${condition.name} (${(condition.confidence * 100).toFixed(1)}%)`}
                    color={condition.confidence > 0.5 ? "success" : "warning"}
                    variant="outlined"
                  />
                ))
              ) : (
                <Typography color="text.secondary">
                  No significant conditions detected
                </Typography>
              )}
            </Box>
            <Divider sx={{ my: 2 }} />
            <Typography variant="body2" color="text.secondary">
              Note: Confidence scores above 30% are considered for detection. Higher scores indicate stronger confidence in the diagnosis.
            </Typography>
          </Paper>
        </Grid>

        {/* Chart Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Confidence Score Distribution
            </Typography>
            <Box sx={{ height: 300 }}>
              <Bar data={chartData} options={chartOptions} />
            </Box>
          </Paper>
        </Grid>

        {/* Detailed Analysis Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Detailed Analysis
            </Typography>
            <Grid container spacing={2}>
              {Object.entries(predictions).map(([condition, confidence]) => (
                <Grid item xs={12} key={condition}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Typography sx={{ minWidth: 150, fontWeight: confidence > 0.3 ? 600 : 400 }}>
                      {condition}
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={confidence * 100}
                      sx={{
                        flexGrow: 1,
                        height: 10,
                        borderRadius: 5,
                        backgroundColor: 'grey.200',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: confidence > 0.5 ? 'success.main' : 
                                         confidence > 0.3 ? 'warning.main' : 'error.main',
                        },
                      }}
                    />
                    <Typography 
                      sx={{ 
                        minWidth: 60,
                        color: confidence > 0.5 ? 'success.main' : 
                               confidence > 0.3 ? 'warning.main' : 'text.secondary'
                      }}
                    >
                      {(confidence * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ResultsDisplay; 