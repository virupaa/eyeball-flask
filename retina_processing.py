import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ArtificialRetina_OG import ArtificialRetina

def process_image_from_stream(
    file_stream,
    fovea_radius=15,
    peripheral_active_cones=0,
    fovea_active_rods=0,
    peripheral_blur_enabled=False,
    peripheral_blur_kernal=(21, 21),
    foveation_type="static",
    verbose=False,
    peripheral_gaussian_sigma=2,
    fovea_x=0,
    fovea_y=0,
    input_image_resolution=320,
    peripheral_grayscale=True
):
    # P = 320  # Resolution of the artificial retina
    # center = (P // 2, P // 2)
    center = (fovea_x, fovea_y)

    # Reset the file stream in case it's been read already
    file_stream.seek(0)

    # Read the image from the file stream
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    if image is None:
        raise ValueError("Failed to decode the image. Ensure the file is a valid image.")

    # Initialize ArtificialRetina with user-provided parameters
    retina = ArtificialRetina(
        image=image,
        foveation_type=foveation_type,
        dynamic_foveation_grid_size=(2, 2),  # Not used for static
        fovea_center=center,
        fovea_radius=fovea_radius,
        peripheral_active_cones=peripheral_active_cones,
        fovea_active_rods=fovea_active_rods,
        peripheral_gaussianBlur=peripheral_blur_enabled,
        peripheral_gaussianBlur_kernal=peripheral_blur_kernal,
        peripheral_gaussianBlur_sigma=peripheral_gaussian_sigma,
        retinal_warp=False,
        verbose=verbose,
        display_output=False,
        video=None,
        save_output=False,
        output_dir=None,
        input_image_resolution=input_image_resolution,
        peripheral_grayscale=peripheral_grayscale

    )

    # Save the processed image to an in-memory file
    output_stream = io.BytesIO()
    plt.imsave(output_stream, retina.retina_image, format='png')  # Save as PNG format
    output_stream.seek(0)  # Reset the stream's position to the beginning

    return output_stream
