import React, { useState } from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  IconButton, 
  Avatar, 
  Box, 
  Menu, 
  MenuItem, 
  Divider,
  useMediaQuery,
  useTheme,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton
} from '@mui/material';
import { 
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  InsertChartOutlined as AnalyticsIcon,
  Article as ReportsIcon,
  Person as ProfileIcon,
  ExitToApp as LogoutIcon
} from '@mui/icons-material';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

const Header = ({ activeTab }) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { currentUser, userProfile, signOut } = useAuth();
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleUserMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setAnchorEl(null);
  };

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleLogout = async () => {
    try {
      await signOut();
      navigate('/');
    } catch (err) {
      console.error('Logout error:', err);
    }
  };

  const handleNavigation = (path) => {
    navigate(path);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { text: 'Analyze Reviews', icon: <AnalyticsIcon />, path: '/analyze' },
    { text: 'My Reports', icon: <ReportsIcon />, path: '/reports' },
  ];

  const drawer = (
    <Box sx={{ width: 250 }}>
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="h6" color="primary" fontWeight="bold">
          Sentiment Analyzer
        </Typography>
      </Box>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={activeTab === item.text.toLowerCase() || 
                (activeTab === undefined && item.text === 'Dashboard')}
              onClick={() => handleNavigation(item.path)}
            >
              <ListItemIcon sx={{ color: 'primary.main' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <>
      <AppBar position="static" color="default" elevation={1}>
        <Toolbar>
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          
          <Typography 
            variant="h6" 
            component={RouterLink}
            to="/"
            sx={{ 
              textDecoration: 'none', 
              color: 'primary.main', 
              fontWeight: 'bold',
              flexGrow: { xs: 1, md: 0 }
            }}
          >
            Sentiment Analyzer
          </Typography>
          
          {!isMobile && (
            <Box sx={{ display: 'flex', ml: 4, flexGrow: 1 }}>
              {menuItems.map((item) => (
                <Button
                  key={item.text}
                  color="inherit"
                  component={RouterLink}
                  to={item.path}
                  sx={{ 
                    mx: 1,
                    fontWeight: activeTab === item.text.toLowerCase() || 
                      (activeTab === undefined && item.text === 'Dashboard') 
                      ? 'bold' : 'normal',
                    color: activeTab === item.text.toLowerCase() || 
                      (activeTab === undefined && item.text === 'Dashboard')
                      ? 'primary.main' : 'inherit'
                  }}
                >
                  {item.text}
                </Button>
              ))}
            </Box>
          )}
          
          <Box sx={{ ml: 'auto', display: 'flex', alignItems: 'center' }}>
            <Button 
              variant="contained" 
              color="primary"
              size="small"
              component={RouterLink}
              to="/analyze"
              sx={{ mr: 2 }}
            >
              New Analysis
            </Button>
            
            <IconButton 
              onClick={handleUserMenuOpen}
              aria-controls="user-menu"
              aria-haspopup="true"
            >
              <Avatar 
                src={currentUser?.photoURL || ''} 
                alt={userProfile?.name || currentUser?.email || 'User'}
                sx={{ width: 40, height: 40 }}
              >
                {userProfile?.name ? userProfile.name.charAt(0).toUpperCase() : 
                  currentUser?.email ? currentUser.email.charAt(0).toUpperCase() : 'U'}
              </Avatar>
            </IconButton>
          </Box>
          
          <Menu
            id="user-menu"
            anchorEl={anchorEl}
            keepMounted
            open={Boolean(anchorEl)}
            onClose={handleUserMenuClose}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
          >
            <MenuItem disabled>
              <Typography variant="body2" color="textSecondary">
                {userProfile?.name || currentUser?.email}
              </Typography>
            </MenuItem>
            <Divider />
            <MenuItem onClick={() => { handleUserMenuClose(); navigate('/profile'); }}>
              <ListItemIcon>
                <ProfileIcon fontSize="small" />
              </ListItemIcon>
              <Typography variant="body2">My Profile</Typography>
            </MenuItem>
            <MenuItem onClick={() => { handleUserMenuClose(); handleLogout(); }}>
              <ListItemIcon>
                <LogoutIcon fontSize="small" />
              </ListItemIcon>
              <Typography variant="body2">Logout</Typography>
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>
      
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={handleDrawerToggle}
        ModalProps={{
          keepMounted: true, // Better open performance on mobile
        }}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': { boxSizing: 'border-box', width: 250 },
        }}
      >
        {drawer}
      </Drawer>
    </>
  );
};

export default Header; 