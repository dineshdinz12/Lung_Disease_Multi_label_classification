import React from 'react';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';

const Header = () => {
  return (
    <AppBar position="static" color="primary" elevation={0}>
      <Toolbar sx={{ pl: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <LocalHospitalIcon sx={{ mr: 1 }} />
          <Typography variant="h6" component="div">
            Chest X-Ray Analysis
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 