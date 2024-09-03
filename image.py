import cv2
import argparse
from utils import *

# Features to add makeup
face_elements = [
    "LIP_LOWER",
    "LIP_UPPER",
    "EYEBROW_LEFT",
    "EYEBROW_RIGHT",
    "EYELINER_LEFT",
    "EYELINER_RIGHT",
    "EYESHADOW_LEFT",
    "EYESHADOW_RIGHT",
]

# Change the color of features
colors_map = {
    # Upper lip and lower lips
    "LIP_UPPER": [0, 0, 255],  # Red in BGR
    "LIP_LOWER": [0, 0, 255],  # Red in BGR
    # Eyeliner
    "EYELINER_LEFT": [139, 0, 0],  # Dark Blue in BGR
    "EYELINER_RIGHT": [139, 0, 0],  # Dark Blue in BGR
    # Eye shadow
    "EYESHADOW_LEFT": [0, 100, 0],  # Dark Green in BGR
    "EYESHADOW_RIGHT": [0, 100, 0],  # Dark Green in BGR
    # Eyebrow
    "EYEBROW_LEFT": [19, 69, 139],  # Dark Brown in BGR
    "EYEBROW_RIGHT": [19, 69, 139],  # Dark Brown in BGR
}

def update_image(image, weight, colors_map):
    # Create a new mask
    mask = np.zeros_like(image)
    
    # Extract facial landmarks
    face_landmarks = read_landmarks(image=image)
    
    # Apply the colors and create a mask
    mask = add_mask(
        mask,
        idx_to_coordinates=face_landmarks,
        face_connections=[face_points[idx] for idx in face_elements],
        colors=[colors_map[idx] for idx in face_elements],
    )
    
    # Combine the image and mask with the updated weight
    output = cv2.addWeighted(image, 1.0, mask, weight, 1.0)
    
    return output

def on_change(val):
    # Callback for trackbar changes
    global colors_map
    global image
    global weight
    
    weight = cv2.getTrackbarPos('Effect Weight', 'Makeup Application') / 100
    
    # Update colors based on trackbar positions for all features
    for feature in face_elements:
        colors_map[feature] = [
            cv2.getTrackbarPos(f'{feature}_B', 'Makeup Application'),
            cv2.getTrackbarPos(f'{feature}_G', 'Makeup Application'),
            cv2.getTrackbarPos(f'{feature}_R', 'Makeup Application')
        ]
    
    # Update the image with the new settings
    output = update_image(image, weight, colors_map)
    cv2.imshow('Makeup Application output', output)

def main(image_path):
    global colors_map
    global image
    global weight

    weight = 0.2  # Initial blending weight
    
    # Load image
    image = cv2.imread(image_path)
    
    # Create window
    cv2.namedWindow('Makeup Application')
    
    # Create trackbars for weight and color adjustment
    cv2.createTrackbar('Effect Weight', 'Makeup Application', int(weight * 100), 100, on_change)
    
    # Trackbars for color adjustments for all features
    for feature in face_elements:
        cv2.createTrackbar(f'{feature}_B', 'Makeup Application', colors_map[feature][0], 255, on_change)
        cv2.createTrackbar(f'{feature}_G', 'Makeup Application', colors_map[feature][1], 255, on_change)
        cv2.createTrackbar(f'{feature}_R', 'Makeup Application', colors_map[feature][2], 255, on_change)
    
    # Initial display
    output = update_image(image, weight, colors_map)
    cv2.imshow('Makeup Application output', output)
    
    # Wait for user interaction
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Image to add Facial makeup")
    parser.add_argument("--img", type=str, help="Path to the image.")
    args = parser.parse_args()
    main(args.img)
