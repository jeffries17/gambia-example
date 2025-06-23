import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Paper, 
  Divider,
  Link,
  IconButton,
  InputAdornment,
  Alert,
  CircularProgress
} from '@mui/material';
import { Google as GoogleIcon, Visibility, VisibilityOff } from '@mui/icons-material';
import { useAuth } from './AuthContext';
import { useNavigate, Link as RouterLink } from 'react-router-dom';

const SignupForm = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const { signUp, signInWithGoogle } = useAuth();
  const navigate = useNavigate();
  
  const handleSignup = async (e) => {
    e.preventDefault();
    
    // Validation
    if (!name || !email || !password || !confirmPassword) {
      setError('Please fill in all fields');
      return;
    }
    
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }
    
    try {
      setIsLoading(true);
      setError('');
      await signUp(email, password, name);
      navigate('/dashboard');
    } catch (err) {
      console.error('Signup error:', err);
      if (err.code === 'auth/email-already-in-use') {
        setError('Email is already in use. Please use a different email or sign in.');
      } else {
        setError('Failed to create account. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleGoogleSignup = async () => {
    try {
      setIsLoading(true);
      setError('');
      await signInWithGoogle();
      navigate('/dashboard');
    } catch (err) {
      console.error('Google signup error:', err);
      setError('Failed to sign up with Google. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 4, maxWidth: 400, width: '100%', mx: 'auto' }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Create an Account
      </Typography>
      <Typography color="text.secondary" gutterBottom>
        Start analyzing your customer reviews today.
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mt: 2, mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <form onSubmit={handleSignup}>
        <TextField
          margin="normal"
          required
          fullWidth
          label="Full Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          disabled={isLoading}
        />
        <TextField
          margin="normal"
          required
          fullWidth
          label="Email Address"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={isLoading}
        />
        <TextField
          margin="normal"
          required
          fullWidth
          label="Password"
          type={showPassword ? 'text' : 'password'}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={isLoading}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  onClick={() => setShowPassword(!showPassword)}
                  edge="end"
                >
                  {showPassword ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
        <TextField
          margin="normal"
          required
          fullWidth
          label="Confirm Password"
          type={showPassword ? 'text' : 'password'}
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          disabled={isLoading}
        />
        
        <Button
          type="submit"
          fullWidth
          variant="contained"
          sx={{ mt: 3, mb: 2, py: 1.2 }}
          disabled={isLoading}
        >
          {isLoading ? <CircularProgress size={24} /> : 'Sign Up'}
        </Button>
      </form>
      
      <Divider sx={{ my: 3 }}>or</Divider>
      
      <Button
        fullWidth
        variant="outlined"
        startIcon={<GoogleIcon />}
        onClick={handleGoogleSignup}
        disabled={isLoading}
        sx={{ py: 1.2 }}
      >
        Sign up with Google
      </Button>
      
      <Box sx={{ mt: 3, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          Already have an account?{' '}
          <Link 
            component={RouterLink} 
            to="/login"
            color="primary"
            underline="hover"
            fontWeight="medium"
          >
            Sign In
          </Link>
        </Typography>
      </Box>
    </Paper>
  );
};

export default SignupForm; 