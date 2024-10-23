import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ArtificialRetina import ArtificialRetina

def process_image_from_stream(file_stream):
    P = 320  # Adjust this as needed
    center = (P // 2, P // 2)

    # Reset the file stream in case it's been read already
    file_stream.seek(0)

    # Read the image from file stream
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Use static foveation since there are no consecutive frames
    retina = ArtificialRetina(
        image=image,
        P=P,
        foveation_type='static',  # Switch to 'static'
        dynamic_foveation_grid_size=(2, 2),  # This won't be used with 'static'
        fovea_center=center,
        fovea_radius=15,
        peripheral_active_cones=0,
        fovea_active_rods=0,
        peripheral_gaussianBlur=False,
        peripheral_gaussianBlur_kernal=(21, 21),
        peripheral_gaussianBlur_sigma=2,
        peripheral_grayscale=True,
        retinal_warp=False,
        verbose=False,
        display_output=False,
        video=None,
        save_output=False,
        output_dir=None,
    )

    # Save the output image to an in-memory file
    output_stream = io.BytesIO()
    plt.imsave(output_stream, retina.retina_image, format='png')
    output_stream.seek(0)
    return output_stream
