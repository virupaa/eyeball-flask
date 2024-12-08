from flask import Flask, request, jsonify
from ArtificialRetina import ArtificialRetina
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Hello, World!"

# Input: parameters, [inputImage_0, inputImage_1, ...]
@app.route('/process', methods=['POST'])
def process():
    try:
        if 'parameters' not in request.form:
            return "No parameters provided", 400
        
        if len(request.files) == 0:
            return "No images provided", 400

        parameters = request.form['parameters']
        parameters = json.loads(parameters) # Convert string to dictionary

        inputImageResolution = int(parameters.get("inputImageResolution", 320))
        foveaRadius = int(parameters.get("foveaRadius", 15))
        foveaX = int(parameters.get("foveaX", 0))
        foveaY = int(parameters.get("foveaY", 0))
        peripheralConeCells = int(parameters.get("peripheralConeCells", 0))
        foveaRodCells = int(parameters.get("foveaRodCells", 0))
        isPeripheralBlurEnabled = parameters.get("isPeripheralBlurEnabled", "False") == "True"
        kernelValue = tuple(map(int, parameters.get("kernelValue", "(21, 21)").strip("()").split(",")))
        peripheralSigma = float(parameters.get("peripheralSigma", 2))
        isPeripheralGrayscale = parameters.get("isPeripheralGrayscale", "true") == "true"
        foveationType = parameters.get("foveationType", "static")
        retinalWarp = parameters.get("retinalWarp", "true") == "true"

        processed_images = []
        try:
            retina = ArtificialRetina(
            P=inputImageResolution,
            fovea_radius=foveaRadius,
            fovea_center=(foveaX, foveaY),
            peripheral_active_cones=peripheralConeCells,
            fovea_active_rods=foveaRodCells,
            peripheral_gaussianBlur=isPeripheralBlurEnabled,
            peripheral_gaussianBlur_kernal=kernelValue,
            peripheral_gaussianBlur_sigma=peripheralSigma,
            peripheral_grayscale=isPeripheralGrayscale,
            foveation_type=foveationType,
            dynamic_foveation_grid_size=(2, 2),
            retinal_warp=retinalWarp,
            )
        except Exception as e:
            return f"Failed to create ArtificialRetina object: {str(e)}", 500
            
        if foveationType == "static":
            for key in request.files:
                if key.startswith('inputImage_'):
                    try:
                        file_stream = request.files[key].stream
                        processed_images.append(f"data:image/png;base64,{retina.apply(current_image=file_stream)}")
                    except Exception as e:
                        return f"Failed to process image {key}: {str(e)}", 500
        elif foveationType == "dynamic":
            file_keys = list(request.files.keys())
            for i in range(len(file_keys) - 1):
                try:
                    current_file = request.files[file_keys[i]]
                    next_file = request.files[file_keys[i + 1]]
                    processed_images.append(f"data:image/png;base64,{retina.apply(current_image=current_file.stream, next_image=next_file.stream)}")
                except Exception as e:
                    return f"Failed to process image {file_keys[i]} or {file_keys[i + 1]}: {str(e)}", 500
            try:
                last_file = request.files[file_keys[-1]]
                processed_images.append(f"data:image/png;base64,{retina.apply(current_image=last_file.stream)}")
            except Exception as e:
                return f"Failed to process image {file_keys[-1]}: {str(e)}", 500
        else:
            return "Invalid foveation type", 400

        return jsonify({"processedImages": processed_images}), 200
    except Exception as e:
        print("fail", e)
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
