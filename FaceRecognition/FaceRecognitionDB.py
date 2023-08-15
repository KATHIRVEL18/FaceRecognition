import cv2
import os

dataset = "dataset"
name = "kathir"

path = os.path.join(dataset, name)
if not os.path.isdir(path):
    os.makedirs(path)

(width, height) = (130, 100)




Algorithm = "haarcascade_frontalface_default.xml"


HaarCascade = cv2.CascadeClassifier(Algorithm)

Camera = cv2.VideoCapture(0)

count = 1

while count < 31:

    print(count)
    _,Image = Camera.read()
    GrayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    Face = HaarCascade.detectMultiScale(GrayImage, 1.3, 4)

    for (x,y,w,h) in Face:
        cv2.rectangle(Image, (x,y), (x+w,y+h), (0,0,255), 2)
        faceOnly = GrayImage[y:y+h, x:x+w]

        resizeImage = cv2.resize(faceOnly, (width, height))
        cv2.imwrite("%s/%s.jpg" %(path,count), faceOnly)
        count+=1

       
    cv2.imshow("FaceDetection.",Image)
    Key = cv2.waitKey(10)
    if Key == 27:
        break;
print("Image Captured Successfully.")
cv2.Camera.release()
cv2.destroyAllWindows()
