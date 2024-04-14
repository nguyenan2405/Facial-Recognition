import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

def read_images_from_subdirectories(dataset_dir):
    imgList = []
    studentIds = []

    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                full_path = os.path.join(root, filename)
                try:
                    img = cv2.imread(full_path)
                    if img is None:
                        raise ValueError(f"Failed to read image: {full_path}")
                    imgList.append(img)
                    student_id = os.path.splitext(filename)[0]  # Extract student ID
                    studentIds.append(student_id)

                except Exception as e:
                    print(f"Error processing file '{full_path}': {str(e)}")

    return imgList, studentIds

def find_encodings(images_list):
    encode_list = []
    for img in images_list:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encode_list.append(encode)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            continue  # Skip to the next image

    return encode_list

if __name__ == "__main__":
    dataset_dir = "Dataset"  # Replace with the actual path to your dataset directory

    images, student_ids = read_images_from_subdirectories(dataset_dir)

    if images:
        encodings = find_encodings(images)
        encoded_data = [encodings, student_ids]

        with open("EncodeFile.p", 'wb') as file:
            pickle.dump(encoded_data, file)
        print("Encoding and saving completed.")
    else:
        print("No images were found in the dataset directory. Please check the path and file structure.")