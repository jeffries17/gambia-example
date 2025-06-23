import React from 'react';
import { Box, Container, Grid, Typography } from '@mui/material';
import ForgotPasswordForm from '../components/auth/ForgotPasswordForm';

const ForgotPasswordPage = () => {
  return (
    <Box 
      sx={{ 
        minHeight: '100vh', 
        display: 'flex', 
        alignItems: 'center',
        py: 5,
        background: 'linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%)'
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4} alignItems="center" justifyContent="center">
          <Grid item xs={12} md={6}>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <Typography variant="h3" fontWeight="bold" gutterBottom>
                Reset Your Password
              </Typography>
              <Typography variant="h6" color="text.secondary">
                Don't worry, it happens to the best of us.
              </Typography>
            </Box>
            <ForgotPasswordForm />
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default ForgotPasswordPage; 