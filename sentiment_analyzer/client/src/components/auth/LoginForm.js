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

const LoginForm = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const { signIn, signInWithGoogle } = useAuth();
  const navigate = useNavigate();
  
  const handleEmailLogin = async (e) => {
    e.preventDefault();
    
    if (!email || !password) {
      setError('Please enter both email and password');
      return;
    }
    
    try {
      setIsLoading(true);
      setError('');
      await signIn(email, password);
      navigate('/dashboard');
    } catch (err) {
      console.error('Login error:', err);
      setError(
        err.code === 'auth/invalid-credential'
          ? 'Invalid email or password'
          : 'Failed to sign in. Please try again.'
      );
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleGoogleLogin = async () => {
    try {
      setIsLoading(true);
      setError('');
      await signInWithGoogle();
      navigate('/dashboard');
    } catch (err) {
      console.error('Google login error:', err);
      setError('Failed to sign in with Google. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 4, maxWidth: 400, width: '100%', mx: 'auto' }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Sign In
      </Typography>
      <Typography color="text.secondary" gutterBottom>
        Welcome back! Please sign in to continue.
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mt: 2, mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <form onSubmit={handleEmailLogin}>
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
        
        <Box sx={{ mt: 1, textAlign: 'right' }}>
          <Link 
            component={RouterLink} 
            to="/forgot-password"
            color="primary"
            underline="hover"
            variant="body2"
          >
            Forgot password?
          </Link>
        </Box>
        
        <Button
          type="submit"
          fullWidth
          variant="contained"
          sx={{ mt: 3, mb: 2, py: 1.2 }}
          disabled={isLoading}
        >
          {isLoading ? <CircularProgress size={24} /> : 'Sign In'}
        </Button>
      </form>
      
      <Divider sx={{ my: 3 }}>or</Divider>
      
      <Button
        fullWidth
        variant="outlined"
        startIcon={<GoogleIcon />}
        onClick={handleGoogleLogin}
        disabled={isLoading}
        sx={{ py: 1.2 }}
      >
        Sign in with Google
      </Button>
      
      <Box sx={{ mt: 3, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          Don't have an account?{' '}
          <Link 
            component={RouterLink} 
            to="/signup"
            color="primary"
            underline="hover"
            fontWeight="medium"
          >
            Sign Up
          </Link>
        </Typography>
      </Box>
    </Paper>
  );
};

export default LoginForm; 