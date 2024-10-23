from flask import Flask, request, render_template, send_file
import zipfile
import io
from retina_processing import process_image_from_stream  # Correct import path
import os

app = Flask(__name__)

# Function to process all images in a zip folder
def process_folder(zip_file):
    output_zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref, zipfile.ZipFile(output_zip_stream, 'w') as output_zip:
        for filename in zip_ref.namelist():
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Read the image file from the zip
                with zip_ref.open(filename) as file_stream:
                    output_image_stream = process_image_from_stream(file_stream)
                    output_zip.writestr(f'processed_{filename}', output_image_stream.getvalue())
    output_zip_stream.seek(0)
    return output_zip_stream

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' in request.files:
        file = request.files['file']
        if zipfile.is_zipfile(file):
            # If it's a zip file, process the folder
            output_zip_stream = process_folder(file)
            return send_file(output_zip_stream, as_attachment=True, attachment_filename='processed_images.zip', mimetype='application/zip')
        else:
            # Process a single image
            output_image_stream = process_image_from_stream(file)
            return send_file(output_image_stream, as_attachment=True, attachment_filename='processed_image.png', mimetype='image/png')
    else:
        return "No file uploaded", 400
