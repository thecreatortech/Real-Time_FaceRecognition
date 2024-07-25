import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import time

# Constants
ENCODING_SCALE = 0.25  # Scale factor for input images
DETECTION_INTERVAL = 0.5  # seconds between full face detections
CONFIDENCE_THRESHOLD = 0.6  # Threshold for face recognition confidence
MAX_DISAPPEARED = 15  # Maximum number of frames a face can disappear before removing it
IOU_THRESHOLD = 0.5  # Intersection over Union threshold for face matching
MAX_FACES = 5  # Maximum number of faces to track simultaneously

def load_images(path):
    """Load images and class names from the specified path."""
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
    
    print(f"Class names: {class_names}")
    return images, class_names

def find_encodings(images):
    """Find face encodings for the given images."""
    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img_rgb)[0]
        encode_list.append(encode)
    return encode_list

def check_person(name):
    """Record the detected person's name with timestamp."""
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
        else:
            print(f"{name} already marked.")

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersect_w = max(0, min(x2, x4) - max(x1, x3))
    intersect_h = max(0, min(y2, y4) - max(y1, y3))
    intersection = intersect_w * intersect_h

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

class FaceTracker:
    def __init__(self):
        self.faces = {}
        self.disappeared = {}
        self.next_face_id = 0

    def update(self, face_locations, face_names):
        current_faces = set()

        # Update existing faces and add new ones
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            matched = False
            for face_id, face_info in self.faces.items():
                stored_loc = face_info["location"]
                if calculate_iou((left, top, right, bottom), stored_loc) > IOU_THRESHOLD:
                    self.faces[face_id]["location"] = (top, right, bottom, left)
                    self.faces[face_id]["name"] = name  # Update name in case it changed
                    self.disappeared[face_id] = 0
                    current_faces.add(face_id)
                    matched = True
                    break
            
            if not matched and len(self.faces) < MAX_FACES:
                new_face_id = self.next_face_id
                self.faces[new_face_id] = {"name": name, "location": (top, right, bottom, left)}
                self.disappeared[new_face_id] = 0
                current_faces.add(new_face_id)
                self.next_face_id += 1

        # Remove faces that have disappeared
        for face_id in list(self.faces.keys()):
            if face_id not in current_faces:
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > MAX_DISAPPEARED:
                    del self.faces[face_id]
                    del self.disappeared[face_id]

        return self.faces

def process_frame(img, encode_list_known, class_names):
    """Process a single frame to detect and recognize faces."""
    img_small = cv2.resize(img, (0, 0), None, ENCODING_SCALE, ENCODING_SCALE)
    img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(img_small_rgb, model="hog")
    face_encodings = face_recognition.face_encodings(img_small_rgb, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(encode_list_known, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(encode_list_known, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index] and face_distances[best_match_index] < CONFIDENCE_THRESHOLD:
            name = class_names[best_match_index].upper()
            check_person(name)
        face_names.append(name)
    
    return face_locations, face_names

def draw_faces(img, faces):
    """Draw bounding boxes and names for detected faces."""
    for face_id, face_info in faces.items():
        name = face_info["name"]
        (top, right, bottom, left) = face_info["location"]
        top = int(top / ENCODING_SCALE)
        right = int(right / ENCODING_SCALE)
        bottom = int(bottom / ENCODING_SCALE)
        left = int(left / ENCODING_SCALE)

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
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
    face_tracker = FaceTracker()
    last_detection_time = time.time() - DETECTION_INTERVAL

    while True:
        success, img = camera.read()
        if not success:
            print("Failed to grab frame")
            break

        current_time = time.time()
        if current_time - last_detection_time >= DETECTION_INTERVAL:
            face_locations, face_names = process_frame(img, encode_list_known, class_names)
            faces = face_tracker.update(face_locations, face_names)
            last_detection_time = current_time
        else:
            faces = face_tracker.faces  # Use the last known face positions

        img_with_faces = draw_faces(img.copy(), faces)

        # Display detection status
        status_text = f"Detected: {len(faces)} face(s)"
        cv2.putText(img_with_faces, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', img_with_faces)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()