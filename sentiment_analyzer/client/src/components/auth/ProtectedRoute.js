import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { Box, CircularProgress } from '@mui/material';

/**
 * ProtectedRoute component for handling authenticated routes
 * Redirects to /login if user is not authenticated
 */
const ProtectedRoute = ({ children }) => {
  const { currentUser, loading } = useAuth();
  const location = useLocation();

  // Show loading spinner while checking auth state
  if (loading) {
    return (
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh' 
        }}
      >
        <CircularProgress size={60} />
      </Box>
    );
  }

  // Redirect to login if not authenticated
  if (!currentUser) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Render children if authenticated
  return children;
};

export default ProtectedRoute; 