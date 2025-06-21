import cv2
import os

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

current_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
samples_dir = os.path.join(current_dir, "samples")
os.makedirs(samples_dir, exist_ok=True)

detector = cv2.CascadeClassifier(cascade_path)

face_id = input("Enter a Numeric user ID  here:  ")
print("Taking samples, look at camera ....... ")
count = 0

while True:
    ret, img = cam.read()
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(converted_image, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        try:
            face_img = converted_image[y:y+h, x:x+w]
            save_path = os.path.join(samples_dir, f"face.{face_id}.{count}.jpg")
            cv2.imwrite(save_path, face_img)
        except Exception as e:
            print("Error saving image:", e)

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 100:
         break

print("Samples taken now closing the program....")
cam.release()
cv2.destroyAllWindows()
