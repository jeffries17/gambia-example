import React from 'react';
import { Box, Container, Grid, Typography } from '@mui/material';
import LoginForm from '../components/auth/LoginForm';

const LoginPage = () => {
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
        <Grid container spacing={4} alignItems="center">
          <Grid item xs={12} md={6}>
            <Box sx={{ pr: { md: 6 }, mb: { xs: 4, md: 0 } }}>
              <Typography variant="h3" fontWeight="bold" gutterBottom>
                Welcome Back
              </Typography>
              <Typography variant="h6" color="text.secondary" paragraph>
                Sign in to access your dashboard and continue analyzing customer feedback.
              </Typography>
              <Typography variant="body1" paragraph>
                Transform guest reviews into actionable insights that help you improve your tourism business.
                Discover what your customers love and identify areas for improvement.
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <LoginForm />
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default LoginPage; 