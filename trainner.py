import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
#detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path='dataSet'

def getImagesWithID(path):
    #the path of all the files 
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faces=[]
    #create empty ID list
    IDs=[]
    #looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        faceImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        faceNp=np.array(faceImage,'uint8')
        #getting the Id from the image
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        # extract the face from the training image sample
        #faces=detector.detectMultiScale(imageNp)
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("traning",faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces
Ids,faces = getImagesWithID(path)
recognizer.train(faces,Ids)
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
