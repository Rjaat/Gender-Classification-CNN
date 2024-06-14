# import cv2
# from mtcnn import MTCNN
# import os

# # Load the MTCNN detector
# detector = MTCNN()

# def extract_and_save_faces(image_path, output_dir):
#     # Read the image
#     image = cv2.imread(image_path)
    
#     # Convert to RGB (MTCNN works with RGB images)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Detect faces in the image
#     faces = detector.detect_faces(image_rgb)
    
#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Extract faces and save as individual image files
#     for i, face in enumerate(faces):
#         x, y, w, h = face['box']
#         extracted_face = image_rgb[y:y+h, x:x+w]
#         output_path = os.path.join(output_dir, f'face_{i+1}.jpg')
#         cv2.imwrite(output_path, cv2.cvtColor(extracted_face, cv2.COLOR_RGB2BGR))
#         print(f'Saved face {i+1} to {output_path}')

# # Example usage
# image_path = 'download.jpeg'
# output_directory = 'output_faces'
# extract_and_save_faces(image_path, output_directory)




import cv2
from mtcnn import MTCNN
import os

# Load the MTCNN detector
detector = MTCNN()

def extract_and_save_faces(image_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to RGB (MTCNN works with RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    faces = detector.detect_faces(image_rgb)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract faces and save as individual image files
    extracted_faces_paths = []
    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        extracted_face = image_rgb[y:y+h, x:x+w]
        output_path = os.path.join(output_dir, f'face_{i+1}.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(extracted_face, cv2.COLOR_RGB2BGR))
        extracted_faces_paths.append(output_path)
        print(f'Saved face {i+1} to {output_path}')
    
    return extracted_faces_paths
