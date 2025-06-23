import React from 'react';
import { Box, Button, Card, CardContent, Container, Grid, Typography, List, ListItem, ListItemIcon, ListItemText, Divider } from '@mui/material';
import { styled } from '@mui/material/styles';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import HotelIcon from '@mui/icons-material/Hotel';
import DirectionsBoatIcon from '@mui/icons-material/DirectionsBoat';
import TerrainIcon from '@mui/icons-material/Terrain';
import LooksOneIcon from '@mui/icons-material/LooksOne';
import LooksTwoIcon from '@mui/icons-material/LooksTwo';
import Looks3Icon from '@mui/icons-material/Looks3';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';

const HeroSection = styled(Box)(({ theme }) => ({
  background: 'linear-gradient(135deg, #1a237e 0%, #4a148c 100%)',
  color: 'white',
  padding: theme.spacing(10, 0, 15),
  position: 'relative',
  overflow: 'hidden',
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: '100%',
    height: '30%',
    background: 'linear-gradient(to top, white, transparent)',
  }
}));

const StepCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  borderRadius: theme.spacing(2),
  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.1)',
  transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-10px)',
    boxShadow: '0 15px 35px rgba(0, 0, 0, 0.15)',
  }
}));

const SectionHeading = styled(Typography)(({ theme }) => ({
  position: 'relative',
  marginBottom: theme.spacing(6),
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: -10,
    left: 0,
    width: 60,
    height: 4,
    backgroundColor: theme.palette.primary.main,
  }
}));

const PricingCard = styled(Card)(({ theme, featured }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  borderRadius: theme.spacing(2),
  boxShadow: featured ? '0 10px 30px rgba(74, 20, 140, 0.2)' : '0 5px 15px rgba(0, 0, 0, 0.08)',
  border: featured ? `2px solid ${theme.palette.primary.main}` : 'none',
  transition: 'transform 0.3s ease-in-out',
  '&:hover': {
    transform: 'scale(1.03)',
  }
}));

const ActionButton = styled(Button)(({ theme }) => ({
  borderRadius: theme.spacing(3),
  padding: theme.spacing(1, 4),
  fontWeight: 'bold',
}));

const LandingPage = () => {
  return (
    <>
      {/* Hero Section - Above the fold */}
      <HeroSection>
        <Container maxWidth="lg">
          <Grid container spacing={6} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography variant="h2" fontWeight="bold" gutterBottom>
                Unlock the Voice of Your Customers
              </Typography>
              <Typography variant="h5" sx={{ mb: 4, opacity: 0.9 }}>
                Transform reviews into actionable insights to improve your tourism business
              </Typography>
              <ActionButton 
                variant="contained" 
                color="secondary" 
                size="large"
                endIcon={<ArrowForwardIcon />}
              >
                Start Free Analysis
              </ActionButton>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <StepCard>
                    <CardContent sx={{ display: 'flex', alignItems: 'center' }}>
                      <LooksOneIcon color="primary" sx={{ fontSize: 50, mr: 2 }} />
                      <Box>
                        <Typography variant="h6" fontWeight="bold">Upload Your Reviews</Typography>
                        <Typography color="textSecondary">
                          Paste text, upload a spreadsheet, or connect to TripAdvisor
                        </Typography>
                      </Box>
                    </CardContent>
                  </StepCard>
                </Grid>
                
                <Grid item xs={12}>
                  <StepCard>
                    <CardContent sx={{ display: 'flex', alignItems: 'center' }}>
                      <LooksTwoIcon color="primary" sx={{ fontSize: 50, mr: 2 }} />
                      <Box>
                        <Typography variant="h6" fontWeight="bold">Our AI Analyzes Sentiment</Typography>
                        <Typography color="textSecondary">
                          Advanced algorithms identify themes and sentiment patterns
                        </Typography>
                      </Box>
                    </CardContent>
                  </StepCard>
                </Grid>
                
                <Grid item xs={12}>
                  <StepCard>
                    <CardContent sx={{ display: 'flex', alignItems: 'center' }}>
                      <Looks3Icon color="primary" sx={{ fontSize: 50, mr: 2 }} />
                      <Box>
                        <Typography variant="h6" fontWeight="bold">Get Actionable Insights</Typography>
                        <Typography color="textSecondary">
                          Beautiful reports and strategic recommendations for your business
                        </Typography>
                      </Box>
                    </CardContent>
                  </StepCard>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </Container>
      </HeroSection>

      {/* Industry-specific Sections */}
      <Box sx={{ py: 10 }}>
        <Container maxWidth="lg">
          <SectionHeading variant="h4" fontWeight="bold">
            Tailored Solutions for Tourism Businesses
          </SectionHeading>
          
          <Grid container spacing={5}>
            {/* Hotels Section */}
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', p: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <HotelIcon color="primary" sx={{ fontSize: 40, mr: 2 }} />
                    <Typography variant="h5" fontWeight="bold">For Hotels</Typography>
                  </Box>
                  <Typography paragraph>
                    Understand what guests love and where you can improve across all aspects of their stay.
                  </Typography>
                  <List>
                    <ListItem sx={{ p: 0, mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Room comfort & cleanliness metrics" />
                    </ListItem>
                    <ListItem sx={{ p: 0, mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Staff service analysis" />
                    </ListItem>
                    <ListItem sx={{ p: 0 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Facility & amenity feedback" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            
            {/* Tour Operators Section */}
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', p: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <DirectionsBoatIcon color="primary" sx={{ fontSize: 40, mr: 2 }} />
                    <Typography variant="h5" fontWeight="bold">For Tour Operators</Typography>
                  </Box>
                  <Typography paragraph>
                    Enhance your experiences and highlight what makes your tours special to guests.
                  </Typography>
                  <List>
                    <ListItem sx={{ p: 0, mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Guide performance insights" />
                    </ListItem>
                    <ListItem sx={{ p: 0, mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Experience highlight analysis" />
                    </ListItem>
                    <ListItem sx={{ p: 0 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Value perception metrics" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            
            {/* Destinations Section */}
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', p: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <TerrainIcon color="primary" sx={{ fontSize: 40, mr: 2 }} />
                    <Typography variant="h5" fontWeight="bold">For Destinations</Typography>
                  </Box>
                  <Typography paragraph>
                    Comprehensive analysis to improve your destination's reputation and visitor experience.
                  </Typography>
                  <List>
                    <ListItem sx={{ p: 0, mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Cross-sector insights" />
                    </ListItem>
                    <ListItem sx={{ p: 0, mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Visitor satisfaction mapping" />
                    </ListItem>
                    <ListItem sx={{ p: 0 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Competitive benchmarking" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Pricing Section */}
      <Box sx={{ py: 10, bgcolor: '#f5f5f7' }}>
        <Container maxWidth="lg">
          <SectionHeading variant="h4" fontWeight="bold">
            Simple, Transparent Pricing
          </SectionHeading>
          
          <Grid container spacing={4}>
            {/* Free Tier */}
            <Grid item xs={12} sm={6} md={3}>
              <PricingCard>
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Starter
                  </Typography>
                  <Typography variant="h3" fontWeight="bold" gutterBottom>
                    Free
                  </Typography>
                  <Typography color="textSecondary" gutterBottom>
                    Perfect for first-time users
                  </Typography>
                  <Divider sx={{ my: 2 }} />
                  <List dense>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Up to 10 reviews" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Basic sentiment analysis" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Simple dashboard" />
                    </ListItem>
                  </List>
                </CardContent>
                <Box sx={{ p: 2 }}>
                  <Button variant="outlined" color="primary" fullWidth>
                    Try for Free
                  </Button>
                </Box>
              </PricingCard>
            </Grid>
            
            {/* $49 Tier */}
            <Grid item xs={12} sm={6} md={3}>
              <PricingCard featured={true}>
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Business
                  </Typography>
                  <Typography variant="h3" fontWeight="bold" gutterBottom>
                    $49
                  </Typography>
                  <Typography color="textSecondary" gutterBottom>
                    Most popular for small businesses
                  </Typography>
                  <Divider sx={{ my: 2 }} />
                  <List dense>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Up to 50 reviews" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Advanced sentiment analysis" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Strategic recommendations" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Brand identity suggestions" />
                    </ListItem>
                  </List>
                </CardContent>
                <Box sx={{ p: 2 }}>
                  <Button variant="contained" color="primary" fullWidth>
                    Get Started
                  </Button>
                </Box>
              </PricingCard>
            </Grid>
            
            {/* $199 Tier */}
            <Grid item xs={12} sm={6} md={3}>
              <PricingCard>
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Premium
                  </Typography>
                  <Typography variant="h3" fontWeight="bold" gutterBottom>
                    $199
                  </Typography>
                  <Typography color="textSecondary" gutterBottom>
                    For established businesses
                  </Typography>
                  <Divider sx={{ my: 2 }} />
                  <List dense>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Up to 500 reviews" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Comprehensive analysis" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Detailed strategy report" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Competitor benchmarking" />
                    </ListItem>
                  </List>
                </CardContent>
                <Box sx={{ p: 2 }}>
                  <Button variant="outlined" color="primary" fullWidth>
                    Upgrade
                  </Button>
                </Box>
              </PricingCard>
            </Grid>
            
            {/* Enterprise Tier */}
            <Grid item xs={12} sm={6} md={3}>
              <PricingCard>
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Destination
                  </Typography>
                  <Typography variant="h3" fontWeight="bold" gutterBottom>
                    Custom
                  </Typography>
                  <Typography color="textSecondary" gutterBottom>
                    For DMOs and large organizations
                  </Typography>
                  <Divider sx={{ my: 2 }} />
                  <List dense>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Unlimited reviews" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Multi-sector analysis" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Regional benchmarking" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Custom implementation" />
                    </ListItem>
                  </List>
                </CardContent>
                <Box sx={{ p: 2 }}>
                  <Button variant="outlined" color="primary" fullWidth>
                    Contact Us
                  </Button>
                </Box>
              </PricingCard>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Case Studies Section */}
      <Box sx={{ py: 10 }}>
        <Container maxWidth="lg">
          <SectionHeading variant="h4" fontWeight="bold">
            Success Stories
          </SectionHeading>
          
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h5" fontWeight="bold" gutterBottom>
                    Kingdom of Tonga
                  </Typography>
                  <Typography paragraph>
                    We helped Tonga Tourism understand visitor sentiment across multiple islands, 
                    accommodations, and attractions, identifying key improvement areas and competitive 
                    advantages against regional competitors.
                  </Typography>
                  <Typography variant="subtitle1" fontWeight="bold" color="primary">
                    Results:
                  </Typography>
                  <List>
                    <ListItem sx={{ p: 0, mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="15% improvement in visitor satisfaction" />
                    </ListItem>
                    <ListItem sx={{ p: 0, mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Targeted training for service improvements" />
                    </ListItem>
                    <ListItem sx={{ p: 0 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="New marketing strategy based on strengths" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h5" fontWeight="bold" gutterBottom>
                    Eswatini Tourism
                  </Typography>
                  <Typography paragraph>
                    Our analysis helped Eswatini identify key differentiators from neighboring 
                    destinations, uncovering hidden gems in visitor feedback and opportunities 
                    to enhance the visitor experience.
                  </Typography>
                  <Typography variant="subtitle1" fontWeight="bold" color="primary">
                    Results:
                  </Typography>
                  <List>
                    <ListItem sx={{ p: 0, mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Identification of unique cultural strengths" />
                    </ListItem>
                    <ListItem sx={{ p: 0, mb: 1 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Creation of targeted improvement plan" />
                    </ListItem>
                    <ListItem sx={{ p: 0 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        <CheckCircleIcon color="primary" fontSize="small" />
                      </ListItemIcon>
                      <ListItemText primary="Data-driven promotional campaign strategy" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Call to Action */}
      <Box sx={{ py: 8, bgcolor: 'primary.main', color: 'white' }}>
        <Container maxWidth="md" sx={{ textAlign: 'center' }}>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Ready to Transform Your Customer Feedback?
          </Typography>
          <Typography variant="h6" sx={{ mb: 4, opacity: 0.9 }}>
            Get started today with our free analysis and see the power of sentiment insights
          </Typography>
          <Button 
            variant="contained" 
            color="secondary" 
            size="large"
            sx={{ px: 5, py: 1.5, borderRadius: 3 }}
          >
            Start Your Free Analysis
          </Button>
        </Container>
      </Box>
    </>
  );
};

export default LandingPage; 