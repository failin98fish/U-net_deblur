import numpy as np
import cv2
from scipy.special import comb
from scipy.optimize import minimize

def bezier_curve(t, control_points):
    n = len(control_points) - 1
    result = np.zeros_like(t)

    for i in range(n + 1):
        result += comb(n, i) * (1 - t) ** (n - i) * t ** i * control_points[i]

    return result

def loss_function(params, blurred_image, events):
    # Extract parameters
    control_points = params[:-1].reshape(height, width, 4, 2)
    brightness_scale = params[-1]

    # Initialize reconstructed image
    height, width = blurred_image.shape
    reconstructed_image = np.zeros_like(blurred_image)

    # Iterate over events
    for event in events:
        t, x, y, p = event

        # Get the corresponding control points
        control_points_x = control_points[y.astype(int), x.astype(int), :, 0]
        control_points_y = control_points[y.astype(int), x.astype(int), :, 1]

        # Calculate the interpolated brightness using Bezier curves
        interpolated_brightness = (
            bezier_curve(t, control_points_x) + bezier_curve(t, control_points_y)
        ) / 2

        # Update the reconstructed image
        reconstructed_image[y, x] += p * interpolated_brightness * brightness_scale

    # Calculate the difference between reconstructed image and blurred image
    diff = reconstructed_image - blurred_image

    # Calculate the total loss
    total_loss = np.sum(diff ** 2)

    return total_loss

# Example usage
blurred_image = cv2.imread('/root/Deblurring-Low-Light-Images-with-Events/data/train/share/APS_blur/scene001_000.png', cv2.IMREAD_GRAYSCALE)
events = np.load('data/train/others/events/scene001_000.npy')

# Normalize event timestamps
events[:, 0] = (events[:, 0] - np.min(events[:, 0])) / (np.max(events[:, 0]) - np.min(events[:, 0]))

# Initialize control points and brightness scale
height, width = blurred_image.shape
control_points = np.random.rand(height, width, 4, 2)
brightness_scale = 1.0

# Concatenate control points and brightness scale as initial parameters
params = np.concatenate((control_points.flatten(), [brightness_scale]))

# Minimize the loss function to find optimal parameters
result = minimize(
    loss_function,
    params,
    args=(blurred_image, events),
    method='CG'  # You can choose different optimization methods
)

# Get the optimized parameters
optimized_params = result.x

# Extract the optimized control points and brightness scale
optimized_control_points = optimized_params[:-1].reshape(height, width, 4, 2)
optimized_brightness_scale = optimized_params[-1]

# Reconstruct the sharp image using the optimized parameters
sharp_image = np.zeros_like(blurred_image)

# Iterate over events
for event in events:
    t, x, y, p = event

    # Get the corresponding control points
    control_points_x = optimized_control_points[y, x, :, 0]
    control_points_y = optimized_control_points[y, x, :, 1]

    # Calculate the interpolated brightness using Bezier curves
    interpolated_brightness = (
        bezier_curve(t, control_points_x) + bezier_curve(t, control_points_y)
    ) / 2

    # Update the sharp image
    sharp_image[y, x] += p * interpolated_brightness * optimized_brightness_scale

# Normalize the sharp image
sharp_image = sharp_image / np.max(sharp_image) * 255

# Display and save the sharp image
cv2.imshow('Sharp Image', sharp_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('sharp_image.jpg', sharp_image.astype(np.uint8))