import React, { useState, useEffect } from 'react';
import { Box, Container, CircularProgress, Typography, Button } from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import Header from '../components/layout/Header';
import ResultsDisplay from '../components/analyze/ResultsDisplay';
import { getAuth } from 'firebase/auth';
import axios from 'axios';

const ResultsPage = () => {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const auth = getAuth();
  
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const fetchResults = async () => {
      if (!jobId) {
        setError('No job ID provided');
        setIsLoading(false);
        return;
      }
      
      try {
        const idToken = await auth.currentUser.getIdToken();
        
        // First check job status
        const statusResponse = await axios.get(
          `${process.env.REACT_APP_API_URL}/jobs/${jobId}`,
          {
            headers: {
              Authorization: `Bearer ${idToken}`,
            },
          }
        );
        
        const jobData = statusResponse.data.job || statusResponse.data;
        
        if (jobData.status === 'completed') {
          // Job is complete, get the results
          if (jobData.results) {
            // Results are included in job data
            setResults(jobData.results);
            setIsLoading(false);
          } else {
            // Need to fetch results separately
            const resultsResponse = await axios.get(
              `${process.env.REACT_APP_API_URL}/results/${jobId}`,
              {
                headers: {
                  Authorization: `Bearer ${idToken}`,
                },
              }
            );
            
            setResults(resultsResponse.data.results || resultsResponse.data);
            setIsLoading(false);
          }
        } else if (jobData.status === 'error') {
          setError(jobData.error?.message || 'An error occurred during analysis');
          setIsLoading(false);
        } else {
          // Job is still processing, poll again in a few seconds
          setTimeout(fetchResults, 3000);
        }
      } catch (err) {
        console.error('Error fetching results:', err);
        setError('Failed to fetch analysis results. Please try again.');
        setIsLoading(false);
      }
    };
    
    fetchResults();
  }, [jobId, auth]);
  
  const handleNewAnalysis = () => {
    navigate('/analyze');
  };
  
  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f5f7fa' }}>
      <Header activeTab="analyze" />
      <Container sx={{ py: 4 }}>
        {isLoading ? (
          <Box sx={{ textAlign: 'center', py: 6 }}>
            <CircularProgress sx={{ mb: 2 }} />
            <Typography variant="h6">Loading Results</Typography>
            <Typography color="text.secondary">
              Please wait while we load your analysis results...
            </Typography>
          </Box>
        ) : error ? (
          <Box sx={{ textAlign: 'center', py: 6 }}>
            <Typography variant="h6" color="error" gutterBottom>
              Error Loading Results
            </Typography>
            <Typography paragraph>{error}</Typography>
            <Button variant="contained" onClick={handleNewAnalysis}>
              Start New Analysis
            </Button>
          </Box>
        ) : (
          <>
            <ResultsDisplay results={results} />
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleNewAnalysis}
                sx={{ mr: 2 }}
              >
                Analyze More Reviews
              </Button>
              <Button 
                variant="outlined" 
                onClick={() => navigate('/reports')}
              >
                View All Reports
              </Button>
            </Box>
          </>
        )}
      </Container>
    </Box>
  );
};

export default ResultsPage; 