#!/usr/bin/env python3
"""
Flask API for Sentiment Analyzer

This module provides the REST API endpoints for the sentiment analysis
web application. It handles file uploads, user authentication, and
dispatches analysis requests to the processing modules.
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, auth
from functools import wraps

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Firebase if credentials exist
try:
    if os.getenv('FIREBASE_CREDENTIALS'):
        cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS'))
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase initialized successfully")
    else:
        logger.warning("No Firebase credentials found. Running in development mode.")
        db = None
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {str(e)}")
    db = None

def verify_firebase_token(f):
    """
    Middleware to verify Firebase authentication tokens
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip token verification in development mode if Firebase isn't initialized
        if db is None and os.getenv('FLASK_ENV') == 'development':
            logger.warning("Skipping token verification in development mode")
            return f(*args, **kwargs)
        
        # Get the authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("No valid authorization header found")
            return jsonify({
                "status": "error",
                "message": "Unauthorized - No valid token provided"
            }), 401
        
        # Extract the token
        token = auth_header.split('Bearer ')[1]
        
        try:
            # Verify the token
            decoded_token = auth.verify_id_token(token)
            # Add user_id to kwargs for the route function
            kwargs['user_id'] = decoded_token['uid']
            logger.info(f"Authenticated user: {decoded_token['uid']}")
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Unauthorized - Invalid token"
            }), 401
    
    return decorated_function

@app.route('/')
def index():
    """API health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "Sentiment Analyzer API is running"
    })

@app.route('/api/analyze', methods=['POST'])
@verify_firebase_token
def analyze_reviews(user_id=None):
    """
    Analyze reviews from uploaded content
    
    Accepts:
    - Text content
    - CSV/JSON files
    - URLs (in Phase 2)
    
    Returns analysis results or job ID for async processing
    """
    try:
        # Get the request data
        data = request.json
        
        # Extract the review text or file data
        text_content = data.get('text')
        file_url = data.get('file_url')
        category = data.get('category', 'general')
        
        # Create a new job document in Firestore
        if db:
            job_ref = db.collection('jobs').document()
            job_id = job_ref.id
            job_data = {
                'user_id': user_id,
                'status': 'pending',
                'created_at': firestore.SERVER_TIMESTAMP,
                'category': category,
                'text_content': text_content,
                'file_url': file_url
            }
            job_ref.set(job_data)
            logger.info(f"Created job {job_id} for user {user_id}")
        else:
            # For dev mode without Firebase
            job_id = "local_dev_job_id"
            
        # If text content is provided, do simple analysis immediately
        if text_content and len(text_content) < 10000:  # Process small text inputs synchronously
            from analyzer import analyze
            results = analyze(text_content, input_type='text', category=category)
            
            # Update the job status in Firestore
            if db:
                job_ref.update({
                    'status': 'completed',
                    'completed_at': firestore.SERVER_TIMESTAMP,
                    'results': results
                })
                
            return jsonify({
                "status": "success",
                "message": "Analysis completed",
                "job_id": job_id,
                "results": results
            })
        else:
            # For larger datasets or file uploads, use async processing
            # TODO: Implement queue processing with Firebase Functions
            return jsonify({
                "status": "success",
                "message": "Analysis request received and queued for processing",
                "job_id": job_id
            })
    except Exception as e:
        logger.error(f"Error in analyze_reviews: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/jobs/<job_id>', methods=['GET'])
@verify_firebase_token
def get_job_status(job_id, user_id=None):
    """Get status and results of an analysis job"""
    try:
        # Fetch job status from Firestore
        if db:
            job_ref = db.collection('jobs').document(job_id)
            job = job_ref.get()
            
            if not job.exists:
                return jsonify({
                    "status": "error",
                    "message": "Job not found"
                }), 404
            
            job_data = job.to_dict()
            
            # Check if the job belongs to the authenticated user
            if job_data.get('user_id') != user_id:
                return jsonify({
                    "status": "error",
                    "message": "Unauthorized access to this job"
                }), 403
            
            return jsonify({
                "status": "success",
                "job": {
                    "id": job_id,
                    "status": job_data.get('status'),
                    "created_at": job_data.get('created_at'),
                    "completed_at": job_data.get('completed_at'),
                    "results": job_data.get('results')
                }
            })
        else:
            # Demo response for development mode
            return jsonify({
                "status": "completed",
                "job_id": job_id,
                "progress": 100,
                "results_url": f"/api/results/{job_id}"
            })
    except Exception as e:
        logger.error(f"Error in get_job_status: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/results/<result_id>', methods=['GET'])
@verify_firebase_token
def get_results(result_id, user_id=None):
    """Get detailed results for an analysis job"""
    try:
        # Fetch results from Firestore
        if db:
            job_ref = db.collection('jobs').document(result_id)
            job = job_ref.get()
            
            if not job.exists:
                return jsonify({
                    "status": "error",
                    "message": "Results not found"
                }), 404
            
            job_data = job.to_dict()
            
            # Check if the job belongs to the authenticated user
            if job_data.get('user_id') != user_id:
                return jsonify({
                    "status": "error",
                    "message": "Unauthorized access to these results"
                }), 403
            
            # Track usage for the user
            user_ref = db.collection('users').document(user_id)
            user_ref.update({
                'usage.last_analysis': firestore.SERVER_TIMESTAMP,
                'usage.total_analyses': firestore.Increment(1),
                'usage.total_reviews_processed': firestore.Increment(job_data.get('results', {}).get('overall', {}).get('review_count', 0))
            })
            
            return jsonify({
                "status": "success",
                "results": job_data.get('results')
            })
        else:
            # Demo response for development mode
            return jsonify({
                "result_id": result_id,
                "created_at": "2023-05-12T10:30:00Z",
                "analysis": {
                    "overview": {
                        "review_count": 50,
                        "average_sentiment": 0.68,
                        "rating_distribution": [2, 5, 8, 15, 20]
                    },
                    "sentiment_breakdown": {
                        "positive": 65,
                        "neutral": 20,
                        "negative": 15
                    },
                    "themes": [
                        {"name": "service", "sentiment": 0.85, "count": 30},
                        {"name": "cleanliness", "sentiment": 0.75, "count": 25},
                        {"name": "location", "sentiment": 0.90, "count": 28},
                        {"name": "value", "sentiment": 0.45, "count": 15},
                        {"name": "facilities", "sentiment": 0.62, "count": 20}
                    ],
                    "top_phrases": [
                        {"text": "friendly staff", "sentiment": 0.92, "count": 12},
                        {"text": "great location", "sentiment": 0.88, "count": 10},
                        {"text": "clean rooms", "sentiment": 0.78, "count": 9},
                        {"text": "poor value", "sentiment": -0.65, "count": 5},
                        {"text": "beautiful views", "sentiment": 0.95, "count": 8}
                    ]
                }
            })
    except Exception as e:
        logger.error(f"Error in get_results: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/user/reports', methods=['GET'])
@verify_firebase_token
def get_user_reports(user_id=None):
    """Get a list of reports for the authenticated user"""
    try:
        # Fetch user reports from Firestore
        if db:
            # Query Firestore for all completed jobs for this user
            jobs_ref = db.collection('jobs').where('user_id', '==', user_id).where('status', '==', 'completed').order_by('created_at', direction=firestore.Query.DESCENDING).limit(20)
            jobs = jobs_ref.stream()
            
            reports = []
            for job in jobs:
                job_data = job.to_dict()
                reports.append({
                    "id": job.id,
                    "name": job_data.get('name', f"Analysis {job.id}"),
                    "created_at": job_data.get('created_at'),
                    "review_count": job_data.get('results', {}).get('overall', {}).get('review_count', 0),
                    "average_sentiment": job_data.get('results', {}).get('overall', {}).get('average_sentiment', 0)
                })
            
            return jsonify({
                "status": "success",
                "reports": reports
            })
        else:
            # Demo response for development mode
            return jsonify({
                "reports": [
                    {
                        "id": "report1",
                        "name": "Beach Resort Analysis",
                        "created_at": "2023-05-10T14:30:00Z",
                        "review_count": 120,
                        "average_sentiment": 0.72
                    },
                    {
                        "id": "report2",
                        "name": "Restaurant Reviews",
                        "created_at": "2023-05-05T09:15:00Z",
                        "review_count": 85,
                        "average_sentiment": 0.68
                    }
                ]
            })
    except Exception as e:
        logger.error(f"Error in get_user_reports: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/user/usage', methods=['GET'])
@verify_firebase_token
def get_user_usage(user_id=None):
    """Get usage statistics for the authenticated user"""
    try:
        if db:
            # Fetch user data from Firestore
            user_ref = db.collection('users').document(user_id)
            user = user_ref.get()
            
            if not user.exists:
                return jsonify({
                    "status": "error",
                    "message": "User not found"
                }), 404
            
            user_data = user.to_dict()
            usage_data = user_data.get('usage', {})
            plan_data = user_data.get('plan', {})
            
            return jsonify({
                "status": "success",
                "usage": {
                    "total_analyses": usage_data.get('total_analyses', 0),
                    "total_reviews_processed": usage_data.get('total_reviews_processed', 0),
                    "last_analysis": usage_data.get('last_analysis'),
                },
                "plan": {
                    "name": plan_data.get('name', 'Free'),
                    "review_limit": plan_data.get('review_limit', 10),
                    "remaining_reviews": plan_data.get('review_limit', 10) - usage_data.get('total_reviews_processed', 0),
                    "start_date": plan_data.get('start_date'),
                    "end_date": plan_data.get('end_date')
                }
            })
        else:
            # Demo response for development mode
            return jsonify({
                "status": "success",
                "usage": {
                    "total_analyses": 5,
                    "total_reviews_processed": 205,
                    "last_analysis": "2023-05-10T14:30:00Z"
                },
                "plan": {
                    "name": "Business",
                    "review_limit": 500,
                    "remaining_reviews": 295,
                    "start_date": "2023-05-01T00:00:00Z",
                    "end_date": "2023-06-01T00:00:00Z"
                }
            })
    except Exception as e:
        logger.error(f"Error in get_user_usage: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/analyze_text', methods=['POST'])
def analyze_text():
    """
    Process text directly without authentication for internal use by Cloud Functions
    """
    try:
        # Get the request data
        data = request.json
        
        # Extract the text content and category
        text_content = data.get('text')
        category = data.get('category', 'general')
        
        if not text_content:
            return jsonify({
                "status": "error",
                "message": "No text content provided"
            }), 400
        
        # Analyze the text using the analyzer module
        from analyzer import analyze
        results = analyze(text_content, input_type='text', category=category)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/process_file', methods=['POST'])
@verify_firebase_token
def process_file(user_id=None):
    """
    Process a file from Cloud Storage
    """
    try:
        # Get the request data
        data = request.json
        
        # Extract the file URL and category
        file_url = data.get('file_url')
        category = data.get('category', 'general')
        job_id = data.get('job_id')
        
        if not file_url:
            return jsonify({
                "status": "error",
                "message": "No file URL provided"
            }), 400
        
        # Download the file from Cloud Storage if it's a GCS URL
        if file_url.startswith('gs://'):
            # Initialize Firebase Storage if available
            if db:
                from firebase_admin import storage
                bucket = storage.bucket(app=firebase_admin.get_app())
                
                # Extract the blob name from the URL (gs://bucket-name/blob-name)
                blob_name = file_url.split('gs://')[1].split('/', 1)[1]
                blob = bucket.blob(blob_name)
                
                # Create a temporary file to download to
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                blob.download_to_filename(temp_file.name)
                
                # Close and re-open the file to ensure it's ready for reading
                temp_file.close()
                
                # Now temp_file.name contains the path to the downloaded file
                file_path = temp_file.name
            else:
                return jsonify({
                    "status": "error",
                    "message": "Firebase Storage not available"
                }), 500
        else:
            # It's a local file path or URL
            file_path = file_url
        
        # Process the file based on its type
        import os
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Analyze the file using the analyzer module
        from analyzer import analyze
        
        if file_ext in ['.csv', '.xls', '.xlsx']:
            results = analyze(file_path, input_type='file', category=category)
        else:
            # Default to text file handling
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            results = analyze(text_content, input_type='text', category=category)
        
        # Clean up the temporary file if we created one
        if file_url.startswith('gs://'):
            os.unlink(file_path)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in process_file: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Run the app
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_ENV') == 'development') 