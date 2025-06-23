/**
 * Firebase Cloud Functions for Sentiment Analyzer
 * 
 * These functions provide an interface between the frontend
 * and the Flask backend API. They handle authentication,
 * file uploads, and API proxying.
 */

const functions = require('firebase-functions');
const admin = require('firebase-admin');
const express = require('express');
const cors = require('cors');
const axios = require('axios');

// Initialize Firebase Admin
admin.initializeApp();

// Create Express app
const app = express();
app.use(cors({ origin: true }));

// API endpoint configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://api.sentiment-analyzer.example.com'  // Replace with actual production URL
  : 'http://localhost:5000';

/**
 * Middleware to validate Firebase Authentication
 */
const validateAuth = async (req, res, next) => {
  if (!req.headers.authorization || !req.headers.authorization.startsWith('Bearer ')) {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const idToken = req.headers.authorization.split('Bearer ')[1];
  
  try {
    const decodedToken = await admin.auth().verifyIdToken(idToken);
    req.user = decodedToken;
    return next();
  } catch (error) {
    console.error('Error verifying auth token:', error);
    return res.status(403).json({ error: 'Unauthorized' });
  }
};

/**
 * API proxy endpoints
 */

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Analyze reviews endpoint (requires auth)
app.post('/analyze', validateAuth, async (req, res) => {
  try {
    // Add user info to request
    const requestData = {
      ...req.body,
      user_id: req.user.uid,
      email: req.user.email
    };
    
    // Forward to Flask API
    const response = await axios.post(`${API_BASE_URL}/api/analyze`, requestData, {
      headers: {
        'Authorization': `Bearer ${req.headers.authorization.split('Bearer ')[1]}`
      }
    });
    
    // Record usage in Firestore
    await admin.firestore().collection('usage').add({
      user_id: req.user.uid,
      email: req.user.email,
      operation: 'analyze',
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
      request_size: JSON.stringify(requestData).length,
      job_id: response.data.job_id || null
    });
    
    return res.json(response.data);
  } catch (error) {
    console.error('Error calling analyze API:', error);
    return res.status(500).json({ 
      error: 'Failed to process analysis request',
      details: error.message
    });
  }
});

// Get job status (requires auth)
app.get('/jobs/:jobId', validateAuth, async (req, res) => {
  try {
    const { jobId } = req.params;
    const response = await axios.get(`${API_BASE_URL}/api/jobs/${jobId}`, {
      headers: {
        'Authorization': `Bearer ${req.headers.authorization.split('Bearer ')[1]}`
      }
    });
    return res.json(response.data);
  } catch (error) {
    console.error('Error fetching job status:', error);
    return res.status(500).json({ 
      error: 'Failed to fetch job status',
      details: error.message
    });
  }
});

// Get analysis results (requires auth)
app.get('/results/:resultId', validateAuth, async (req, res) => {
  try {
    const { resultId } = req.params;
    const response = await axios.get(`${API_BASE_URL}/api/results/${resultId}`, {
      headers: {
        'Authorization': `Bearer ${req.headers.authorization.split('Bearer ')[1]}`
      }
    });
    return res.json(response.data);
  } catch (error) {
    console.error('Error fetching results:', error);
    return res.status(500).json({ 
      error: 'Failed to fetch analysis results',
      details: error.message
    });
  }
});

// Get user reports (requires auth)
app.get('/user/reports', validateAuth, async (req, res) => {
  try {
    // Get user reports from Flask API
    const response = await axios.get(`${API_BASE_URL}/api/user/reports`, {
      headers: {
        'Authorization': `Bearer ${req.headers.authorization.split('Bearer ')[1]}`
      }
    });
    
    return res.json(response.data);
  } catch (error) {
    console.error('Error fetching user reports:', error);
    return res.status(500).json({ 
      error: 'Failed to fetch user reports',
      details: error.message
    });
  }
});

// Get user usage stats (requires auth)
app.get('/user/usage', validateAuth, async (req, res) => {
  try {
    // Get user usage stats from Flask API
    const response = await axios.get(`${API_BASE_URL}/api/user/usage`, {
      headers: {
        'Authorization': `Bearer ${req.headers.authorization.split('Bearer ')[1]}`
      }
    });
    
    return res.json(response.data);
  } catch (error) {
    console.error('Error fetching user usage stats:', error);
    return res.status(500).json({ 
      error: 'Failed to fetch user usage stats',
      details: error.message
    });
  }
});

/**
 * Job processing function
 */

// Process sentiment analysis jobs from the queue
exports.processAnalysisJob = functions.firestore
  .document('jobs/{jobId}')
  .onCreate(async (snapshot, context) => {
    const jobData = snapshot.data();
    const jobId = context.params.jobId;
    
    // Only process jobs with 'pending' status
    if (jobData.status !== 'pending') {
      console.log(`Job ${jobId} is not pending, skipping processing`);
      return null;
    }
    
    console.log(`Processing sentiment analysis job ${jobId} for user ${jobData.user_id}`);
    
    try {
      // Update job status to 'processing'
      await snapshot.ref.update({
        status: 'processing',
        processing_started_at: admin.firestore.FieldValue.serverTimestamp()
      });
      
      // Check if this is a file upload or text input
      let results;
      if (jobData.file_url) {
        // Get user auth token for API request
        const user = await admin.auth().getUser(jobData.user_id);
        const customToken = await admin.auth().createCustomToken(jobData.user_id);
        
        // Process file from Cloud Storage
        // Option 1: If using Flask API to process
        const response = await axios.post(`${API_BASE_URL}/api/process_file`, {
          file_url: jobData.file_url,
          category: jobData.category || 'general',
          job_id: jobId
        }, {
          headers: {
            'Authorization': `Bearer ${customToken}`
          }
        });
        results = response.data;
      } else if (jobData.text_content) {
        // Process text content
        // For larger text content we're handling here instead of in the API
        const analysisData = {
          text: jobData.text_content,
          category: jobData.category || 'general'
        };
        
        // Call your Flask backend with the text to analyze
        const response = await axios.post(`${API_BASE_URL}/api/analyze_text`, analysisData);
        results = response.data;
      } else {
        throw new Error('No content to analyze');
      }
      
      // Update job with results
      await snapshot.ref.update({
        status: 'completed',
        completed_at: admin.firestore.FieldValue.serverTimestamp(),
        results: results,
        error: null
      });
      
      // Update user usage counts
      const reviewCount = results.overall?.review_count || 0;
      await admin.firestore().collection('users').doc(jobData.user_id).update({
        'usage.total_analyses': admin.firestore.FieldValue.increment(1),
        'usage.total_reviews_processed': admin.firestore.FieldValue.increment(reviewCount),
        'usage.last_analysis': admin.firestore.FieldValue.serverTimestamp()
      });
      
      console.log(`Successfully processed job ${jobId}`);
      return null;
    } catch (error) {
      console.error(`Error processing job ${jobId}:`, error);
      
      // Update job with error
      await snapshot.ref.update({
        status: 'error',
        error: {
          message: error.message,
          timestamp: admin.firestore.FieldValue.serverTimestamp()
        }
      });
      
      return null;
    }
  });

/**
 * User management functions
 */

// Create user record in Firestore when a new user signs up
exports.createUserRecord = functions.auth.user().onCreate(async (user) => {
  try {
    const { uid, email, displayName, photoURL } = user;
    
    // Create user document in Firestore
    await admin.firestore().collection('users').doc(uid).set({
      email,
      displayName: displayName || '',
      photoURL: photoURL || '',
      createdAt: admin.firestore.FieldValue.serverTimestamp(),
      plan: {
        name: 'Free',
        review_limit: 10,
        start_date: admin.firestore.FieldValue.serverTimestamp()
      },
      usage: {
        total_analyses: 0,
        total_reviews_processed: 0
      },
      lastLoginAt: admin.firestore.FieldValue.serverTimestamp()
    });
    
    console.log(`Created new user record for ${email}`);
    return null;
  } catch (error) {
    console.error('Error creating user record:', error);
    return null;
  }
});

// Update user's last login timestamp
exports.updateUserLastLogin = functions.auth.user().onLogin(async (user) => {
  try {
    await admin.firestore().collection('users').doc(user.uid).update({
      lastLoginAt: admin.firestore.FieldValue.serverTimestamp()
    });
    return null;
  } catch (error) {
    console.error('Error updating user last login:', error);
    return null;
  }
});

// Check and enforce usage limits
exports.checkUsageLimits = functions.firestore
  .document('jobs/{jobId}')
  .onUpdate(async (change, context) => {
    const afterData = change.after.data();
    const beforeData = change.before.data();
    
    // Only check completed jobs that were previously processing
    if (beforeData.status !== 'completed' && afterData.status === 'completed') {
      const userId = afterData.user_id;
      
      try {
        // Get user data
        const userDoc = await admin.firestore().collection('users').doc(userId).get();
        if (!userDoc.exists) {
          console.log(`User ${userId} not found`);
          return null;
        }
        
        const userData = userDoc.data();
        const usage = userData.usage || {};
        const plan = userData.plan || {};
        
        // Check if user has exceeded their plan limits
        if (usage.total_reviews_processed > plan.review_limit) {
          console.log(`User ${userId} has exceeded their plan limit`);
          
          // Add notification for the user
          await admin.firestore().collection('notifications').add({
            user_id: userId,
            type: 'limit_exceeded',
            message: `You've reached your plan limit of ${plan.review_limit} reviews. Please upgrade to continue analyzing reviews.`,
            created_at: admin.firestore.FieldValue.serverTimestamp(),
            read: false
          });
        }
        
        return null;
      } catch (error) {
        console.error(`Error checking usage limits for user ${userId}:`, error);
        return null;
      }
    }
    
    return null;
  });

// Export the API as a Firebase Function
exports.api = functions.https.onRequest(app); 