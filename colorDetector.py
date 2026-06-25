import cv2
import numpy as np

def contours(thresh) :
    contoursList, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contoursList

def dibujarBBoxes(contoursList, img) -> None :
    for cnt in contoursList :
        area = cv2.contourArea(cnt)
        if area > 50 :
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def obtenerMascaraAzul(img) :
    blueBajo = np.array([90, 150, 30], np.uint8)
    blueAlto = np.array([145, 255, 255], np.uint8)

    img = denoise(img)
    frameHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maskAzul = cv2.inRange(frameHsv, blueBajo, blueAlto)

    return maskAzul

def denoise(img) :
    return cv2.GaussianBlur(img, (3,3), 3)

def __main__() :
    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()
        if ret:
            # Procesamiento
            maskAzul = obtenerMascaraAzul(frame)
            contoursList = contours(maskAzul)
            dibujarBBoxes(contoursList, frame)

            # Mostrar frame
            cv2.imshow("frame", frame)
            cv2.imshow("maskAzul", maskAzul)
            key = cv2.waitKey(20)
            if key & 0xFF == ord('q'):
                break
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    __main__()