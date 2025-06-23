import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Alert,
  CircularProgress,
  Paper,
  Tab,
  Tabs,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { CloudUpload as UploadIcon, Description as FileIcon } from '@mui/icons-material';
import axios from 'axios';
import { getAuth } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';

const ReviewUpload = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState(0);
  const [reviewText, setReviewText] = useState('');
  const [businessType, setBusinessType] = useState('accommodations');
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const auth = getAuth();

  // Handle file drop
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setError('');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls', '.xlsx'],
      'application/json': ['.json'],
      'text/plain': ['.txt'],
    },
    maxFiles: 1,
  });

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    setError('');
    setSuccess(false);
  };

  // Handle text area change
  const handleTextChange = (event) => {
    setReviewText(event.target.value);
    setError('');
  };

  // Handle business type change
  const handleBusinessTypeChange = (event) => {
    setBusinessType(event.target.value);
  };

  // Handle submit for text input
  const handleTextSubmit = async () => {
    if (!reviewText.trim()) {
      setError('Please enter some review text to analyze');
      return;
    }

    setIsLoading(true);
    setError('');
    setSuccess(false);

    try {
      const idToken = await auth.currentUser.getIdToken();
      
      // Call to backend API
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/analyze`,
        {
          input_type: 'text',
          data: reviewText,
          category: businessType,
        },
        {
          headers: {
            Authorization: `Bearer ${idToken}`,
          },
        }
      );

      setSuccess(true);
      console.log('Analysis response:', response.data);
      
      // Check if we have results immediately or need to wait
      if (response.data.job_id) {
        // Redirect to results page after a short delay
        setTimeout(() => {
          navigate(`/results/${response.data.job_id}`);
        }, 1000);
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError('Failed to analyze reviews. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle submit for file upload
  const handleFileSubmit = async () => {
    if (!file) {
      setError('Please upload a file to analyze');
      return;
    }

    setIsLoading(true);
    setError('');
    setSuccess(false);

    try {
      const idToken = await auth.currentUser.getIdToken();
      
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      formData.append('category', businessType);
      formData.append('input_type', 'file');

      // Call to backend API
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/analyze`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            Authorization: `Bearer ${idToken}`,
          },
        }
      );

      setSuccess(true);
      console.log('Analysis response:', response.data);
      
      // Redirect to results page
      if (response.data.job_id) {
        setTimeout(() => {
          navigate(`/results/${response.data.job_id}`);
        }, 1000);
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError('Failed to analyze file. Please check file format and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h5" fontWeight="bold" gutterBottom>
          Analyze Reviews
        </Typography>
        <Typography color="text.secondary" paragraph>
          Upload your customer reviews for sentiment analysis and actionable insights.
        </Typography>

        <Box sx={{ mb: 3 }}>
          <Tabs value={activeTab} onChange={handleTabChange} centered>
            <Tab label="Paste Text" />
            <Tab label="Upload File" />
          </Tabs>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mb: 3 }}>
            Analysis submitted successfully! Redirecting to results...
          </Alert>
        )}

        <FormControl fullWidth variant="outlined" sx={{ mb: 3 }}>
          <InputLabel>Business Type</InputLabel>
          <Select
            value={businessType}
            onChange={handleBusinessTypeChange}
            label="Business Type"
            disabled={isLoading}
          >
            <MenuItem value="accommodations">Hotel / Accommodation</MenuItem>
            <MenuItem value="restaurants">Restaurant / Dining</MenuItem>
            <MenuItem value="attractions">Attraction / Activity</MenuItem>
            <MenuItem value="general">General / Other</MenuItem>
          </Select>
        </FormControl>

        {activeTab === 0 ? (
          // Text input tab
          <Box>
            <TextField
              fullWidth
              multiline
              rows={10}
              placeholder="Paste your reviews here... (one review per line or in paragraph form)"
              value={reviewText}
              onChange={handleTextChange}
              disabled={isLoading}
              variant="outlined"
              sx={{ mb: 3 }}
            />
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Pro tip: The more reviews you provide, the more accurate our analysis will be. For best results, include at least 5-10 reviews.
            </Typography>
            <Button
              variant="contained"
              color="primary"
              disabled={isLoading || !reviewText.trim()}
              onClick={handleTextSubmit}
              sx={{ py: 1.2 }}
              fullWidth
            >
              {isLoading ? <CircularProgress size={24} /> : 'Analyze Reviews'}
            </Button>
          </Box>
        ) : (
          // File upload tab
          <Box>
            <Paper
              {...getRootProps()}
              variant="outlined"
              sx={{
                p: 3,
                mb: 3,
                borderStyle: 'dashed',
                borderColor: isDragActive ? 'primary.main' : 'divider',
                bgcolor: isDragActive ? 'rgba(74, 20, 140, 0.04)' : 'background.paper',
                cursor: 'pointer',
                textAlign: 'center',
              }}
            >
              <input {...getInputProps()} />
              <UploadIcon color="primary" sx={{ fontSize: 48, mb: 2 }} />
              {file ? (
                <Box>
                  <FileIcon color="primary" sx={{ mr: 1 }} />
                  <Typography display="inline">{file.name}</Typography>
                </Box>
              ) : isDragActive ? (
                <Typography>Drop your file here...</Typography>
              ) : (
                <Typography>
                  Drag and drop your file here, or click to select
                </Typography>
              )}
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Supported formats: CSV, Excel, JSON, Text
              </Typography>
            </Paper>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Your file should contain reviews with at least one column for review text. Headers are recommended.
            </Typography>

            <Button
              variant="contained"
              color="primary"
              disabled={isLoading || !file}
              onClick={handleFileSubmit}
              sx={{ py: 1.2 }}
              fullWidth
            >
              {isLoading ? <CircularProgress size={24} /> : 'Upload & Analyze'}
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ReviewUpload; 