import React from 'react';
import { Box, Container, Grid, Typography } from '@mui/material';
import SignupForm from '../components/auth/SignupForm';

const SignupPage = () => {
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
                Join Us Today
              </Typography>
              <Typography variant="h6" color="text.secondary" paragraph>
                Create your account to start unlocking insights from your customer reviews.
              </Typography>
              <Typography variant="body1" paragraph>
                Our sentiment analysis tool helps tourism businesses understand what guests love 
                and where improvements can be made. Start with our free tier today and see the power 
                of AI-driven feedback analysis.
              </Typography>
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  With your account, you can:
                </Typography>
                <ul>
                  <li>
                    <Typography>Analyze customer reviews for sentiment and themes</Typography>
                  </li>
                  <li>
                    <Typography>Receive actionable recommendations</Typography>
                  </li>
                  <li>
                    <Typography>Generate visual reports to share with your team</Typography>
                  </li>
                  <li>
                    <Typography>Track improvements over time</Typography>
                  </li>
                </ul>
              </Box>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <SignupForm />
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default SignupPage; 