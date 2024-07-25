import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def load_images(path):
    images = []
    class_names = []
    my_list = os.listdir(path)
    
    if my_list:
        print(f"Image names: {my_list}")
    else:
        print("No images found in the directory")
        return [], []

    for cls in my_list:
        current_image = cv2.imread(os.path.join(path, cls))
        images.append(current_image)
        class_names.append(os.path.splitext(cls)[0])
    
    print(f"Just names, after removing the extension: {class_names}")
    return images, class_names

def find_encodings(images):
    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img_rgb)[0]
        encode_list.append(encode)
    return encode_list

def check_person(name):
    filename = 'personList.csv'
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    time_string = now.strftime('%H:%M:%S')
    
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('Name,Date,Time\n')
    
    with open(filename, 'r+') as f:
        my_data_list = f.readlines()
        name_list = [line.split(',')[0] for line in my_data_list[1:]]
        
        if name not in name_list:
            f.write(f'{name},{date_string},{time_string}\n')
            print(f"Person marked for {name} at {time_string} on {date_string}")
            print("Person marked successfully")
        else:
            print("Person already marked.")
            print("Person not marked successfully")

def process_frame(img, encode_list_known, class_names, scale_factor):
    img_small = cv2.resize(img, (0, 0), None, scale_factor, scale_factor)
    img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    
    faces_current_frame = face_recognition.face_locations(img_small_rgb, model="cnn")
    encodes_current_frame = face_recognition.face_encodings(img_small_rgb, faces_current_frame)
    
    detected_faces = []
    
    for encode_face, face_location in zip(encodes_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_distances = face_recognition.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(face_distances)

        if matches[match_index]:
            name = class_names[match_index].upper()
            y1, x2, y2, x1 = [int(coord / scale_factor) for coord in face_location]
            detected_faces.append((name, (x1, y1, x2, y2)))
            check_person(name)
    
    return detected_faces

def draw_faces(img, detected_faces):
    for name, (x1, y1, x2, y2) in detected_faces:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    return img

def main():
    path = "live_detection_images"
    images, class_names = load_images(path)
    
    if not images:
        print("No images loaded. Exiting.")
        return

    encode_list_known = find_encodings(images)
    print("Encode Completed")
    print(f"Encode list length: {len(encode_list_known)}")

    camera = cv2.VideoCapture(0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            success, img = camera.read()
            if not success:
                print("Failed to grab frame")
                break

            # Process frame at different scales
            future_near = executor.submit(process_frame, img, encode_list_known, class_names, 0.25)
            future_far = executor.submit(process_frame, img, encode_list_known, class_names, 1.0)

            detected_faces_near = future_near.result()
            detected_faces_far = future_far.result()

            # Combine and de-duplicate detected faces
            all_faces = detected_faces_near + detected_faces_far
            unique_faces = list(set(all_faces))

            img_with_faces = draw_faces(img.copy(), unique_faces)

            # Display detection status
            status_text = f"Detected: {len(unique_faces)} face(s)"
            cv2.putText(img_with_faces, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Face Recognition', img_with_faces)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()