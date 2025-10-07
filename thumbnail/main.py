import os
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from .thumbnail_generator import ThumbnailGenerator
import tempfile
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory containing this file (animaker package)
package_dir = os.path.dirname(os.path.abspath(__file__))
# Templates should be in the parent directory (project root)
template_dir = os.path.join(os.path.dirname(package_dir), 'templates')

app = Flask(__name__, template_folder=template_dir)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize thumbnail generator
thumbnail_generator = ThumbnailGenerator()

# Log Azure configuration status
azure_key = os.getenv('AZURE_API_KEY')
if azure_key:
    logger.info(f"Azure API Key loaded: {azure_key[:10]}...")
    logger.info("Web interface will use Azure OpenAI")
else:
    logger.info("No Azure API key found, web interface will use gradient backgrounds")

@app.route('/')
def index():
    """Main page with the thumbnail generator interface."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_thumbnail():
    """Generate thumbnail from text prompt."""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        if len(prompt) > 200:
            return jsonify({'error': 'Prompt too long (max 200 characters)'}), 400
        
        logger.info(f"Generating thumbnail for prompt: {prompt}")
        
        # Generate thumbnail
        thumbnail_path = thumbnail_generator.generate_thumbnail(prompt)
        
        if thumbnail_path and os.path.exists(thumbnail_path):
            return jsonify({
                'success': True,
                'message': 'Thumbnail generated successfully',
                'filename': os.path.basename(thumbnail_path)
            })
        else:
            return jsonify({'error': 'Failed to generate thumbnail'}), 500
            
    except Exception as e:
        logger.error(f"Error generating thumbnail: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/download/<filename>')
def download_thumbnail(filename):
    """Download the generated thumbnail."""
    try:
        # Security check - ensure filename is safe
        if not filename or '..' in filename or '/' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        thumbnail_path = os.path.join(tempfile.gettempdir(), 'thumbnails', filename)
        
        if os.path.exists(thumbnail_path):
            # Check if this is a request for display (from img tag) or download
            if request.headers.get('Accept', '').find('image/') != -1:
                # Display the image in browser
                return send_file(thumbnail_path, mimetype='image/png')
            else:
                # Download as attachment
                return send_file(thumbnail_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        logger.error(f"Error downloading thumbnail: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/preview/<filename>')
def preview_thumbnail(filename):
    """Preview the generated thumbnail (no download)."""
    try:
        # Security check - ensure filename is safe
        if not filename or '..' in filename or '/' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        thumbnail_path = os.path.join(tempfile.gettempdir(), 'thumbnails', filename)
        
        if os.path.exists(thumbnail_path):
            return send_file(thumbnail_path, mimetype='image/png')
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        logger.error(f"Error previewing thumbnail: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'animaker'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Run the Flask application."""
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Animaker on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

if __name__ == "__main__":
    main()
