import cv2
import numpy
import os

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

datasets = 'datasets'
print('Training...')
(Images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectPath = os.path.join(datasets, subdir)
        for fileName in os.listdir(subjectPath):
            path = subjectPath + '/' + fileName
            label = id
            Images.append(cv2.imread(path, 0))
            labels.append(int(label))
           #print(labels)
        id+=1
# Resize all images to a common size (width x height)
resized_images = [cv2.resize(img, (100, 100)) for img in Images]

# Convert lists to NumPy arrays
Images = numpy.array(resized_images)
labels = numpy.array(labels)

print(Images, labels)

(width, height) =  (130, 100)

model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face.FisherFaceRecognizer_create()


model.train(Images, labels)

DefaultCamera = cv2.VideoCapture(0)
count = 0

while True:
    (_,Im) = DefaultCamera.read()
    Gray =  cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(Gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(Im, (x,y), (x+w, y+h), (0, 0, 255), 2)
        face = Gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(Im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if prediction[1] < 800:
            cv2.putText(Im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            print(names[prediction[0]])
            count = 0

        else:
            cont+=1
            cv2.putText(Im, "Unknown.", (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            print("UnKnown Person Is Detected.")
            cv2.imwrite("unknown.jpg", Im)
            cont = 0

    cv2.imshow('FaceRecognition', Im)
    key = cv2.waitKey(10)
    if key == 27:
        break

DefaultCamera.release()
cv2.destroyAllWindows()


