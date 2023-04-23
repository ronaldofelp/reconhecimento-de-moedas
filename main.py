import cv2
import numpy as np
from keras.models import load_model

video = cv2.VideoCapture(0)

model = load_model('modelo/keras_model.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
classes = ["1 real", "50 centavos", "10 centavos", "25 centavos"]


def pre_processing(img):
    img_pre = cv2.GaussianBlur(img, (5, 5), 3)  # Responsável por expandir os pixels
    img_pre = cv2.Canny(img_pre, 98, 140)
    kernel = np.ones((4, 4), np.uint8)
    img_pre = cv2.dilate(img_pre, kernel, iterations=2)
    img_pre = cv2.erode(img_pre, kernel, iterations=1)

    return img_pre


def detect_currency(img):
    img_currency = cv2.resize(img, (224, 224))
    img_currency = np.asanyarray(img_currency)
    img_currency_normalize = (img_currency.astype(np.float32) / 127.0) - 1
    data[0] = img_currency_normalize
    prediction = model.predict(data)
    index = np.argmax(prediction)
    percent = prediction[0][index]
    classe = classes[index]

    return classe, percent


while True:

    _, image = video.read()
    image = cv2.resize(image, (640, 480))
    img_processing = pre_processing(image)
    contours, h1 = cv2.findContours(img_processing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 2000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (245, 5, 5), 2)
            clipping = image[y:y + h, x:x + w]
            classe, conf = detect_currency(clipping)
            cv2.putText(image, str(classe), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Imagem", image)
    cv2.imshow("Imagem pre processada", img_processing)
    if cv2.waitKey(1) & 0xFF == 27:  # Aperte Esc para sair.
        break
        cv2.destroyAllWindows()
        exit()
