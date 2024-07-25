import cv2
import face_recognition
import numpy as np

# Set the desired width and height
desired_width = 500
desired_height = 500

# Load image for testing from local disk and convert them into RGB from BGR
imgTest = face_recognition.load_image_file('Testing_image/Elon musk.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

imgRealTest = face_recognition.load_image_file("Testing_image/Elon_test.jpg")
imgRealTest = cv2.cvtColor(imgRealTest, cv2.COLOR_BGR2RGB)

# Resize the images to the desired width and height
imgTest = cv2.resize(imgTest, (desired_width, desired_height))
imgRealTest = cv2.resize(imgRealTest, (desired_width, desired_height))

# Recognize the image coordinates/Location and encode it
face_locations_test = face_recognition.face_locations(imgTest)
if face_locations_test:
    face_location_test = face_locations_test[0]
    encodeTest = face_recognition.face_encodings(imgTest)[0]
    cv2.rectangle(imgTest, (face_location_test[3], face_location_test[0]), (face_location_test[1], face_location_test[2]), (0, 0, 255), 2)
else:
    print("No faces found in the test image.")

face_locations_real_test = face_recognition.face_locations(imgRealTest)
if face_locations_real_test:
    face_location_real_test = face_locations_real_test[0]
    encodeTest_real_test = face_recognition.face_encodings(imgRealTest)[0]
    cv2.rectangle(imgRealTest, (face_location_real_test[3], face_location_real_test[0]), (face_location_real_test[1], face_location_real_test[2]), (0, 0, 255), 2)
else:
    print("No faces found in the real test image.")

# Print the face location for the test image if found
if face_locations_test:
    print("Face location in test image:", face_location_test)


#compare the face and find the distance of the face 
results = face_recognition.compare_faces([encodeTest],encodeTest_real_test) #list of know faces

#find the distance for best match
face_distance = face_recognition.face_distance([encodeTest],encodeTest_real_test)
print(results)
print(face_distance)

cv2.putText(imgTest,f'{results} {round(face_distance[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)


# Display the converted and resized images
cv2.imshow('Elon Musk', imgTest)
cv2.imshow('Elon Test', imgRealTest)

# Wait for a key press and close the windows
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
