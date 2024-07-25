import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import time
import logging
from threading import Thread
from queue import Queue

# Constants
ENCODING_SCALE = 0.25
DETECTION_INTERVAL = 0.5
CONFIDENCE_THRESHOLD = 0.6
MAX_DISAPPEARED = 15
IOU_THRESHOLD = 0.5
MAX_FACES = 5
RECOGNITION_PERSISTENCE = 5

# Configure logging
logging.basicConfig(filename='face_recognition.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FaceRecognitionSystem:
    def __init__(self):
        self.encode_list_known = []
        self.class_names = []
        self.face_tracker = FaceTracker()
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue()
        self.is_running = False
        self.dynamic_threshold = CONFIDENCE_THRESHOLD

    def load_images(self, path):
        images = []
        class_names = []
        my_list = os.listdir(path)

        if not my_list:
            logging.warning("No images found in the directory")
            return [], []

        for cls in my_list:
            try:
                current_image = cv2.imread(os.path.join(path, cls))
                if current_image is None:
                    logging.warning(f"Image {cls} could not be loaded.")
                    continue
                images.append(current_image)
                class_names.append(os.path.splitext(cls)[0])
            except Exception as e:
                logging.error(f"Error loading image {cls}: {e}")
        
        logging.info(f"Class names: {class_names}")
        return images, class_names

    def find_encodings(self, images):
        encode_list = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                encode = face_recognition.face_encodings(img_rgb)[0]
                encode_list.append(encode)
            except IndexError:
                logging.warning("No faces found in an image during encoding.")
            except Exception as e:
                logging.error(f"Error finding encodings: {e}")
        return encode_list

    def process_frame(self, frame):
        img_small = cv2.resize(frame, (0, 0), None, ENCODING_SCALE, ENCODING_SCALE)
        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(img_small_rgb, model="hog")
        face_encodings = face_recognition.face_encodings(img_small_rgb, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.encode_list_known, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.encode_list_known, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < self.dynamic_threshold:
                name = self.class_names[best_match_index].upper()
            face_names.append(name)
        
        return face_locations, face_names

    def draw_faces(self, img, faces):
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

    def capture_frames(self, camera):
        while self.is_running:
            success, frame = camera.read()
            if not success:
                logging.error("Failed to grab frame")
                continue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def process_frames(self):
        last_detection_time = time.time() - DETECTION_INTERVAL
        while self.is_running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                current_time = time.time()
                if current_time - last_detection_time >= DETECTION_INTERVAL:
                    face_locations, face_names = self.process_frame(frame)
                    faces = self.face_tracker.update(face_locations, face_names)
                    last_detection_time = current_time
                else:
                    faces = self.face_tracker.faces

                img_with_faces = self.draw_faces(frame.copy(), faces)
                self.result_queue.put(img_with_faces)

    def adjust_threshold(self, light_level):
        # Simple linear adjustment based on light level (0-255)
        self.dynamic_threshold = max(0.4, min(0.8, CONFIDENCE_THRESHOLD + (light_level - 128) / 255 * 0.2))

    def run(self):
        path = "live_detection_images"
        images, self.class_names = self.load_images(path)
        
        if not images:
            logging.error("No images loaded. Exiting.")
            return

        self.encode_list_known = self.find_encodings(images)
        logging.info("Encode Completed")
        logging.info(f"Encode list length: {len(self.encode_list_known)}")

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logging.error("Error opening video capture.")
            return

        self.is_running = True
        capture_thread = Thread(target=self.capture_frames, args=(camera,))
        process_thread = Thread(target=self.process_frames)
        capture_thread.start()
        process_thread.start()

        try:
            while self.is_running:
                if not self.result_queue.empty():
                    frame = self.result_queue.get()
                    cv2.imshow('Face Recognition', frame)
                    
                    # Adjust threshold based on average brightness
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    avg_brightness = np.mean(gray)
                    self.adjust_threshold(avg_brightness)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
        finally:
            self.is_running = False
            capture_thread.join()
            process_thread.join()
            camera.release()
            cv2.destroyAllWindows()

class FaceTracker:
    def __init__(self):
        self.faces = {}
        self.disappeared = {}
        self.next_face_id = 0
        self.name_history = {}

    def update(self, face_locations, face_names):
        current_faces = set()

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            matched = False
            for face_id, face_info in self.faces.items():
                stored_loc = face_info["location"]
                if calculate_iou((left, top, right, bottom), stored_loc) > IOU_THRESHOLD:
                    self.faces[face_id]["location"] = (top, right, bottom, left)
                    self.name_history[face_id].append(name)
                    if len(self.name_history[face_id]) > RECOGNITION_PERSISTENCE:
                        self.name_history[face_id].pop(0)
                    self.faces[face_id]["name"] = max(set(self.name_history[face_id]), key=self.name_history[face_id].count)
                    self.disappeared[face_id] = 0
                    current_faces.add(face_id)
                    matched = True
                    break
            
            if not matched and len(self.faces) < MAX_FACES:
                new_face_id = self.next_face_id
                self.faces[new_face_id] = {"name": name, "location": (top, right, bottom, left)}
                self.name_history[new_face_id] = [name]
                self.disappeared[new_face_id] = 0
                current_faces.add(new_face_id)
                self.next_face_id += 1

        for face_id in list(self.faces.keys()):
            if face_id not in current_faces:
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > MAX_DISAPPEARED:
                    del self.faces[face_id]
                    del self.disappeared[face_id]
                    del self.name_history[face_id]

        return self.faces

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersect_w = max(0, min(x2, x4) - max(x1, x3))
    intersect_h = max(0, min(y2, y4) - max(y1, y3))
    intersection = intersect_w * intersect_h

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def main():
    face_recognition_system = FaceRecognitionSystem()
    face_recognition_system.run()

if __name__ == "__main__":
    main()