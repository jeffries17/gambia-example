import React from 'react';
import { 
  Box, 
  Container, 
  Grid, 
  Typography, 
  Card, 
  CardContent,
  Button
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import Header from '../components/layout/Header';
import ReviewUpload from '../components/analyze/ReviewUpload';
import { useAuth } from '../components/auth/AuthContext';

const DashboardPage = ({ activeTab }) => {
  const navigate = useNavigate();
  const { userProfile } = useAuth();
  
  // This function will render the appropriate content based on the active tab
  const renderContent = () => {
    switch (activeTab) {
      case 'analyze':
        return (
          <Box>
            <Typography variant="h4" fontWeight="bold" gutterBottom>
              Analyze Reviews
            </Typography>
            <Typography color="text.secondary" paragraph>
              Upload your customer reviews to get insights and actionable recommendations.
            </Typography>
            <ReviewUpload />
          </Box>
        );
      case 'reports':
        return (
          <Box>
            <Typography variant="h4" fontWeight="bold" gutterBottom>
              My Reports
            </Typography>
            <Typography color="text.secondary" paragraph>
              View, download, and share your previous analysis reports.
            </Typography>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  No reports found
                </Typography>
                <Typography color="text.secondary" paragraph>
                  You haven't generated any reports yet. Start by analyzing some reviews.
                </Typography>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={() => navigate('/analyze')}
                >
                  Analyze Reviews
                </Button>
              </CardContent>
            </Card>
          </Box>
        );
      case 'profile':
        return (
          <Box>
            <Typography variant="h4" fontWeight="bold" gutterBottom>
              My Profile
            </Typography>
            <Typography color="text.secondary" paragraph>
              Manage your account settings and subscription.
            </Typography>
            <Card>
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" fontWeight="bold">
                      Account Information
                    </Typography>
                    <Box sx={{ mt: 2 }}>
                      <Typography>
                        <strong>Name:</strong> {userProfile?.name || 'Not specified'}
                      </Typography>
                      <Typography>
                        <strong>Email:</strong> {userProfile?.email || 'Not specified'}
                      </Typography>
                      <Typography>
                        <strong>Current Plan:</strong> {userProfile?.plan?.charAt(0).toUpperCase() + userProfile?.plan?.slice(1) || 'Free'}
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" fontWeight="bold">
                      Usage Statistics
                    </Typography>
                    <Box sx={{ mt: 2 }}>
                      <Typography>
                        <strong>Reviews Analyzed:</strong> {userProfile?.reviewsAnalyzed || 0}
                      </Typography>
                      <Typography>
                        <strong>Reports Generated:</strong> 0
                      </Typography>
                      <Typography>
                        <strong>Usage This Month:</strong> {userProfile?.usageThisMonth || 0} reviews
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Box>
        );
      default:
        // Default dashboard overview
        return (
          <Box>
            <Typography variant="h4" fontWeight="bold" gutterBottom>
              Dashboard
            </Typography>
            <Typography color="text.secondary" paragraph>
              Welcome to your Sentiment Analyzer dashboard. Get started by analyzing customer reviews or view your past reports.
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={4}>
                <Card sx={{ height: '100%' }}>
                  <CardContent sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="h5" gutterBottom>
                      Analyze New Reviews
                    </Typography>
                    <Typography color="text.secondary" paragraph>
                      Upload reviews from your customers to get actionable insights.
                    </Typography>
                    <Button 
                      variant="contained" 
                      color="primary"
                      onClick={() => navigate('/analyze')}
                    >
                      Start Analysis
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={4}>
                <Card sx={{ height: '100%' }}>
                  <CardContent sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="h5" gutterBottom>
                      View Reports
                    </Typography>
                    <Typography color="text.secondary" paragraph>
                      Access your previously generated analysis reports.
                    </Typography>
                    <Button 
                      variant="outlined" 
                      color="primary"
                      onClick={() => navigate('/reports')}
                    >
                      My Reports
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={4}>
                <Card sx={{ height: '100%' }}>
                  <CardContent sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="h5" gutterBottom>
                      Account Information
                    </Typography>
                    <Typography color="text.secondary" paragraph>
                      Current Plan: {userProfile?.plan?.charAt(0).toUpperCase() + userProfile?.plan?.slice(1) || 'Free'}
                    </Typography>
                    <Button 
                      variant="outlined" 
                      color="primary"
                      onClick={() => navigate('/profile')}
                    >
                      View Profile
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        );
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f5f7fa' }}>
      <Header activeTab={activeTab} />
      <Container sx={{ py: 4 }}>
        {renderContent()}
      </Container>
    </Box>
  );
};

export default DashboardPage; 