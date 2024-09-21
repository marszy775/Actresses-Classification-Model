import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

__model = None
__class_name_to_num = {}
__class_num_to_name = {}


def classify_image(img_base64, file_path=None):
    imgs = get_cropped_img(file_path, img_base64)

    result = []

    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        im_haar = w2d(img, 'db2', 3)
        scalled_im_haar = cv2.resize(im_haar, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_im_haar.reshape(32 * 32, 1)))

        len_img_array = 32 * 32 + 32 * 32 * 3

        final = combined_img.reshape(1, len_img_array).astype(float)

        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __class_name_to_num
        })

    return result


def class_number_to_name(class_num):
    return __class_num_to_name[class_num]


def get_cv2_img_from_base64_string(base64str):
    if ',' in base64str:
        base64str = base64str.split(',')[1]

    nparr = np.frombuffer(base64.b64decode(base64str), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_img(img_path, img_base64):
    face_cascade = cv2.CascadeClassifier("./haar_cascade/haarcascade_frontalface_alt.xml")
    eye_cascade = cv2.CascadeClassifier("./haar_cascade/haarcascade_eye.xml")

    if img_path:
        img = cv2.imread(img_path)
    else:
        img = get_cv2_img_from_base64_string(img_base64)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces


def load_saved_artifacts():
    print('Loading saved artifacts...start')

    global __class_name_to_num
    global __class_num_to_name
    with open('./artifacts/class_dictionary.json', 'r') as f:
        __class_name_to_num = json.load(f)
        __class_num_to_name = {v: k for k, v in __class_name_to_num.items()}

    global __model
    if __model is None:
        with open('./artifacts/celeb_model.pkl', 'rb') as f:
            __model = joblib.load(f)

    print('loading saved artifacts... done')


if __name__ == '__main__':
    load_saved_artifacts()
    # print(classify_image(None, "./test_img/angelina2.jpg"))
    # print(classify_image(None, "./test_img/lawrence1.jpg"))
    # with open("./test_img/angelina1.txt", "r") as f:
    #     img_base64 = f.read()
    #
    # print(classify_image(img_base64, None))
