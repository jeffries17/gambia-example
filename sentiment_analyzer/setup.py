#!/usr/bin/env python3
"""
Setup script for TripAdvisor Sentiment Analysis System
"""

import os
import sys
import subprocess
import platform

def install_chrome_driver():
    """Install ChromeDriver for Selenium."""
    system = platform.system().lower()
    print("Setting up ChromeDriver...")
    
    if system == "darwin":  # macOS
        try:
            # Check if Homebrew is available
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
            print("Installing ChromeDriver via Homebrew...")
            subprocess.run(["brew", "install", "chromedriver"], check=True)
            print("✓ ChromeDriver installed successfully")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Homebrew not found. Please install ChromeDriver manually:")
            print("1. Download from: https://chromedriver.chromium.org/")
            print("2. Add to PATH or place in /usr/local/bin/")
            
    elif system == "linux":
        print("For Linux, please install ChromeDriver manually:")
        print("1. sudo apt-get update")
        print("2. sudo apt-get install chromium-chromedriver")
        print("Or download from: https://chromedriver.chromium.org/")
        
    elif system == "windows":
        print("For Windows, please:")
        print("1. Download ChromeDriver from: https://chromedriver.chromium.org/")
        print("2. Add to PATH or place in same directory as script")
    
    print()

def install_requirements():
    """Install Python requirements."""
    print("Installing Python requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Python requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def setup_nltk_data():
    """Download required NLTK data."""
    print("Setting up NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✓ NLTK data downloaded successfully")
    except Exception as e:
        print(f"✗ Error setting up NLTK: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = [
        'results',
        'results/comparisons',
        'uploads',
        'outputs',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_basic_templates():
    """Create basic HTML templates."""
    print("Creating basic templates...")
    
    templates_dir = 'templates'
    
    # Extract page template
    extract_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2>Extract TripAdvisor Reviews</h2>
        <p>Paste a TripAdvisor URL below to extract and analyze reviews.</p>
    </div>
</div>

<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-body">
                <form id="extractForm">
                    <div class="mb-3">
                        <label for="url" class="form-label">TripAdvisor URL</label>
                        <input type="url" class="form-control" id="url" required 
                               placeholder="https://www.tripadvisor.com/...">
                        <div class="form-text">Paste the URL of a hotel, restaurant, or attraction from TripAdvisor</div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="maxReviews" class="form-label">Maximum Reviews</label>
                        <select class="form-control" id="maxReviews">
                            <option value="25">25 reviews</option>
                            <option value="50" selected>50 reviews</option>
                            <option value="100">100 reviews</option>
                            <option value="200">200 reviews</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">
                        <span id="submitText">Extract Reviews</span>
                        <span id="loadingSpinner" class="spinner-border spinner-border-sm d-none" role="status"></span>
                    </button>
                </form>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-body">
                <h6 class="card-title">Supported URLs</h6>
                <ul class="small">
                    <li>Hotels & Accommodations</li>
                    <li>Restaurants & Cafes</li>
                    <li>Attractions & Activities</li>
                    <li>Tours & Experiences</li>
                </ul>
                
                <h6 class="card-title mt-3">Privacy</h6>
                <p class="small">Reviewer names are anonymized to initials only.</p>
            </div>
        </div>
    </div>
</div>

<div class="row mt-4 d-none" id="resultsSection">
    <div class="col-12">
        <div class="alert alert-success" role="alert" id="successAlert"></div>
        <div class="alert alert-danger" role="alert" id="errorAlert"></div>
    </div>
</div>

<script>
$(document).ready(function() {
    $('#extractForm').on('submit', function(e) {
        e.preventDefault();
        
        const url = $('#url').val();
        const maxReviews = $('#maxReviews').val();
        
        // Show loading state
        $('#submitText').text('Extracting...');
        $('#loadingSpinner').removeClass('d-none');
        $('button[type="submit"]').prop('disabled', true);
        
        // Make API call
        $.ajax({
            url: '/api/extract',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                url: url,
                max_reviews: parseInt(maxReviews)
            }),
            success: function(response) {
                if (response.success) {
                    $('#successAlert').html(`
                        <strong>Extraction Complete!</strong><br>
                        Business: ${response.business_name}<br>
                        Category: ${response.category}<br>
                        Location: ${response.location}<br>
                        Reviews: ${response.review_count}<br>
                        <a href="/api/download/${response.filename}" class="btn btn-sm btn-success mt-2">Download JSON</a>
                        <a href="/compare" class="btn btn-sm btn-primary mt-2">Compare with Other Destinations</a>
                    `).show();
                    $('#errorAlert').hide();
                    $('#resultsSection').removeClass('d-none');
                }
            },
            error: function(xhr) {
                const error = xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error occurred';
                $('#errorAlert').text('Error: ' + error).show();
                $('#successAlert').hide();
                $('#resultsSection').removeClass('d-none');
            },
            complete: function() {
                // Reset loading state
                $('#submitText').text('Extract Reviews');
                $('#loadingSpinner').addClass('d-none');
                $('button[type="submit"]').prop('disabled', false);
            }
        });
    });
});
</script>
{% endblock %}'''

    # Compare page template
    compare_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2>Compare Destinations</h2>
        <p>Select 2 or more extracted destinations to compare their sentiment analysis.</p>
    </div>
</div>

{% if results_files %}
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-body">
                <form id="compareForm">
                    <div class="mb-3">
                        <label class="form-label">Select Destinations to Compare</label>
                        {% for file in results_files %}
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" value="{{ file.filename }}" 
                                   id="file_{{ loop.index }}">
                            <label class="form-check-label" for="file_{{ loop.index }}">
                                <strong>{{ file.business_name }}</strong><br>
                                <small class="text-muted">{{ file.review_count }} reviews • {{ file.extraction_date[:10] }}</small>
                            </label>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <button type="submit" class="btn btn-success">
                        <span id="compareText">Compare Destinations</span>
                        <span id="compareSpinner" class="spinner-border spinner-border-sm d-none" role="status"></span>
                    </button>
                </form>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-body">
                <h6 class="card-title">Comparison Features</h6>
                <ul class="small">
                    <li>Overall sentiment scores</li>
                    <li>Rating comparisons</li>
                    <li>Aspect-based analysis</li>
                    <li>Visual charts & graphs</li>
                </ul>
            </div>
        </div>
    </div>
</div>

<div class="row mt-4 d-none" id="comparisonResults">
    <div class="col-12">
        <div class="alert alert-success" role="alert" id="comparisonSuccess"></div>
        <div class="alert alert-danger" role="alert" id="comparisonError"></div>
    </div>
</div>

{% else %}
<div class="row">
    <div class="col-12">
        <div class="alert alert-info" role="alert">
            No extracted destinations found. <a href="{{ url_for('extract_page') }}">Extract some reviews first</a> to compare destinations.
        </div>
    </div>
</div>
{% endif %}

<script>
$(document).ready(function() {
    $('#compareForm').on('submit', function(e) {
        e.preventDefault();
        
        const selectedFiles = [];
        $('.form-check-input:checked').each(function() {
            selectedFiles.push($(this).val());
        });
        
        if (selectedFiles.length < 2) {
            alert('Please select at least 2 destinations to compare.');
            return;
        }
        
        // Show loading state
        $('#compareText').text('Comparing...');
        $('#compareSpinner').removeClass('d-none');
        $('button[type="submit"]').prop('disabled', true);
        
        // Make API call
        $.ajax({
            url: '/api/compare',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                files: selectedFiles
            }),
            success: function(response) {
                if (response.success) {
                    let resultsHtml = `
                        <strong>Comparison Complete!</strong><br>
                        Destinations: ${response.destinations.join(' vs ')}<br>
                        Total Reviews: ${response.total_reviews}<br><br>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Destination</th>
                                        <th>Avg Rating</th>
                                        <th>Avg Sentiment</th>
                                        <th>Reviews</th>
                                    </tr>
                                </thead>
                                <tbody>
                    `;
                    
                    for (const [dest, data] of Object.entries(response.results_summary)) {
                        resultsHtml += `
                            <tr>
                                <td>${dest}</td>
                                <td>${data.avg_rating ? data.avg_rating.toFixed(1) : 'N/A'}</td>
                                <td>${data.avg_sentiment ? data.avg_sentiment.toFixed(2) : 'N/A'}</td>
                                <td>${data.review_count || 0}</td>
                            </tr>
                        `;
                    }
                    
                    resultsHtml += `
                                </tbody>
                            </table>
                        </div>
                        <a href="/api/download/${response.comparison_file}" class="btn btn-sm btn-success mt-2">Download Full Results</a>
                    `;
                    
                    $('#comparisonSuccess').html(resultsHtml).show();
                    $('#comparisonError').hide();
                    $('#comparisonResults').removeClass('d-none');
                }
            },
            error: function(xhr) {
                const error = xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error occurred';
                $('#comparisonError').text('Error: ' + error).show();
                $('#comparisonSuccess').hide();
                $('#comparisonResults').removeClass('d-none');
            },
            complete: function() {
                // Reset loading state
                $('#compareText').text('Compare Destinations');
                $('#compareSpinner').addClass('d-none');
                $('button[type="submit"]').prop('disabled', false);
            }
        });
    });
});
</script>
{% endblock %}'''

    # Results page template
    results_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2>Analysis Results</h2>
        <p>View and download your extraction and comparison results.</p>
    </div>
</div>

{% if results_files %}
<div class="row">
    <div class="col-12">
        <h4>Extracted Reviews</h4>
        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Business Name</th>
                        <th>Reviews</th>
                        <th>Extraction Date</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for file in results_files %}
                    <tr>
                        <td>{{ file.business_name }}</td>
                        <td>{{ file.review_count }}</td>
                        <td>{{ file.extraction_date[:10] }}</td>
                        <td>
                            <a href="/api/download/{{ file.filename }}" class="btn btn-sm btn-primary">Download</a>
                            <button class="btn btn-sm btn-secondary" onclick="viewFile('{{ file.filename }}')">View</button>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>
{% endif %}

{% if comparison_files %}
<div class="row mt-4">
    <div class="col-12">
        <h4>Comparison Results</h4>
        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Destinations</th>
                        <th>Total Reviews</th>
                        <th>Comparison Date</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for file in comparison_files %}
                    <tr>
                        <td>{{ file.destinations | join(' vs ') }}</td>
                        <td>{{ file.total_reviews }}</td>
                        <td>{{ file.comparison_date[:10] }}</td>
                        <td>
                            <a href="/api/download/{{ file.filename }}" class="btn btn-sm btn-success">Download</a>
                            <button class="btn btn-sm btn-secondary" onclick="viewFile('{{ file.filename }}')">View</button>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>
{% endif %}

{% if not results_files and not comparison_files %}
<div class="row">
    <div class="col-12">
        <div class="alert alert-info" role="alert">
            No results found. <a href="{{ url_for('extract_page') }}">Start by extracting some reviews</a>.
        </div>
    </div>
</div>
{% endif %}

<!-- Modal for viewing file contents -->
<div class="modal fade" id="fileModal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">File Contents</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <pre id="fileContents" style="max-height: 400px; overflow-y: auto;"></pre>
            </div>
        </div>
    </div>
</div>

<script>
function viewFile(filename) {
    $.get('/api/view/' + filename)
        .done(function(data) {
            $('#fileContents').text(JSON.stringify(data, null, 2));
            $('#fileModal').modal('show');
        })
        .fail(function() {
            alert('Error loading file contents.');
        });
}
</script>
{% endblock %}'''

    # Write templates
    template_files = {
        'extract.html': extract_template,
        'compare.html': compare_template,
        'results.html': results_template
    }
    
    for filename, content in template_files.items():
        filepath = os.path.join(templates_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created template: {filename}")

def main():
    """Main setup function."""
    print("TripAdvisor Sentiment Analysis System Setup")
    print("==========================================")
    print()
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please fix the requirements installation and try again.")
        return
    
    # Setup NLTK
    if not setup_nltk_data():
        print("Warning: NLTK setup failed. Some features may not work properly.")
    
    # Install ChromeDriver
    install_chrome_driver()
    
    # Create directories
    create_directories()
    
    # Create templates
    create_basic_templates()
    
    print()
    print("✓ Setup complete!")
    print()
    print("To run the system:")
    print("1. cd sentiment_analyzer")
    print("2. python web_interface/app.py")
    print("3. Open http://localhost:5000 in your browser")
    print()
    print("For command-line usage:")
    print("python extractors/tripadvisor_extractor.py")

if __name__ == "__main__":
    main() 