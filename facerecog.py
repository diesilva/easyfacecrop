


import cv2

size = 1
webcam = cv2.VideoCapture(0) #camera 0

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,0)

    thumbnail = cv2.resize(im, (im.shape[1] / size, im.shape[0] / size))


    faceslist = classifier.detectMultiScale(thumbnail)

   
    for f in faceslist:
        (x, y, w, h) = [v * size for v in f]
        cv2.rectangle(im, (x, y), (x + w, y + h),(125,255,60),thickness=1)

        sub_face = im[y:y+h, x:x+w]
        FaceFileName = "/Users/diegosilva/walmart/new_face_crop_DS/unknownfaces/pic_" + str(y) + ".jpg"
        cv2.imwrite(FaceFileName, sub_face)

 
    cv2.imshow('face recog',   im)

    key = cv2.waitKey(10)

    if key == 27: 
            break
