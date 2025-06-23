import React, { useState } from 'react';
import { 
  TextField, 
  Button, 
  Typography, 
  Paper, 
  Alert,
  Box,
  Link,
  CircularProgress
} from '@mui/material';
import { useAuth } from './AuthContext';
import { Link as RouterLink } from 'react-router-dom';

const ForgotPasswordForm = () => {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  
  const { resetPassword } = useAuth();
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!email) {
      setError('Please enter your email address');
      return;
    }
    
    try {
      setIsLoading(true);
      setError('');
      setSuccess(false);
      await resetPassword(email);
      setSuccess(true);
    } catch (err) {
      console.error('Password reset error:', err);
      setError('Failed to send password reset email. Please check if the email is correct.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 4, maxWidth: 400, width: '100%', mx: 'auto' }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Reset Password
      </Typography>
      <Typography color="text.secondary" gutterBottom>
        Enter your email address and we'll send you a link to reset your password.
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mt: 2, mb: 2 }}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" sx={{ mt: 2, mb: 2 }}>
          Password reset link has been sent to your email.
        </Alert>
      )}
      
      <form onSubmit={handleSubmit}>
        <TextField
          margin="normal"
          required
          fullWidth
          label="Email Address"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={isLoading || success}
          sx={{ mb: 2 }}
        />
        
        <Button
          type="submit"
          fullWidth
          variant="contained"
          sx={{ mt: 1, mb: 2, py: 1.2 }}
          disabled={isLoading || success}
        >
          {isLoading ? <CircularProgress size={24} /> : 'Send Reset Link'}
        </Button>
        
        <Box sx={{ mt: 2, textAlign: 'center' }}>
          <Typography variant="body2">
            <Link 
              component={RouterLink} 
              to="/login"
              color="primary"
              underline="hover"
            >
              Back to Sign In
            </Link>
          </Typography>
        </Box>
      </form>
    </Paper>
  );
};

export default ForgotPasswordForm; 