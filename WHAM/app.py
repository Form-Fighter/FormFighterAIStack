from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from form_fighter import process_video

# Initialize Flask app
app = Flask(__name__)

# Define where to temporarily save uploaded files
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/process-video', methods=['POST'])
def process_video_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.mov'):

        # Save the file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the video file
        result_json = process_video(file_path)

        # Delete the file after processing
        os.remove(file_path)

        # Return the result as JSON
        return jsonify(result_json)
    else:
        return jsonify({"error": "Invalid file type, only .mov accepted"}), 400

if __name__ == '__main__':
    app.run(debug=True)
