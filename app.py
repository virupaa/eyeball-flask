from flask import Flask, request, render_template, send_file
import zipfile
import io
from retina_processing import process_image_from_stream  # Correct import path
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
    return "Hello, World!"

@app.route('/process', methods=['POST'])
def process():
    # Print all form data
    print("Received Form Data:")
    for key, value in request.form.items():
        print(f"{key}: {value}")
    
    if 'file' in request.files:
        file = request.files['file']
        
        # Extract parameters from request.form
        fovea_radius = int(request.form.get("fovea_radius", 15))
        peripheral_active_cones = int(request.form.get("peripheral_cone_cells", 0))
        fovea_active_rods = int(request.form.get("fovea_rod_cells", 0))
        peripheral_blur_enabled = request.form.get("peripheral_blur_enabled", "False") == "True"
        peripheral_blur_kernal = tuple(map(int, request.form.get("kernel_value", "(21, 21)").strip("()").split(",")))
        foveation_type = request.form.get("fovea_type", "static")
        verbose = request.form.get("verbose", "False") == "True"
        peripheral_gaussian_sigma = int(request.form.get("peripheral_gaussian_sigma", 2))
        fovea_x = int(request.form.get("fovea_x", 0))
        fovea_y = int(request.form.get("fovea_y", 0))
        input_image_resolution = float(request.form.get("input_image_resolution", 320)),
        peripheral_grayscale = request.form.get("peripheral_grayscale", "True") == "True"

        # Pass these parameters to the processing function
        if zipfile.is_zipfile(file):
            print("Processing a folder (ZIP file).")
            output_zip_stream = process_folder(file)
            return send_file(output_zip_stream, as_attachment=True, attachment_filename='processed_images.zip', mimetype='application/zip')
        else:
            print("Processing a single image.")
            output_image_stream = process_image_from_stream(
                file_stream=file,
                fovea_radius=fovea_radius,
                peripheral_active_cones=peripheral_active_cones,
                fovea_active_rods=fovea_active_rods,
                peripheral_blur_enabled=peripheral_blur_enabled,
                peripheral_blur_kernal=peripheral_blur_kernal,
                foveation_type=foveation_type,
                verbose=verbose,
                peripheral_gaussian_sigma=peripheral_gaussian_sigma,
                fovea_x=fovea_x,
                fovea_y=fovea_y,
                input_image_resolution=input_image_resolution,
                peripheral_grayscale=peripheral_grayscale
            )
            return send_file(output_image_stream, as_attachment=True, attachment_filename='processed_image.png', mimetype='image/png')
    else:
        print("No file uploaded.")
        return "No file uploaded", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
