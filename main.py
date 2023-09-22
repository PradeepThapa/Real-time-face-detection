# Import libraries
import cv2
import numpy as np
import requests
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# Azure credentials
SUBSCRIPTION_KEY = <<Replace with your subscription key>>
ENDPOINT = None #'https://<<Replace with your endpoint>>.cognitiveservices.azure.com/'

# Initiate client
computervision_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(SUBSCRIPTION_KEY))

# Select the visual feature you want.
remote_image_features = ["faces"]

# use analyse url
analyze_url = ENDPOINT + "vision/v2.1/analyze"

# image path
image_path = <<replace with your image path>>

# capture video using built in camera
video = cv2.VideoCapture(0)

# text font
font = cv2.FONT_ITALIC

# continuosly play video
while True:
    ret, frame = video.read()  # read video
    color = cv2.cvtColor(frame, 1)

    # get width and height of a frame
    height, width, channel = frame.shape

    # resize frame
    image = cv2.resize(frame, (round(width / 2), height), None, .25, .25)

    # save image
    cv2.imwrite("test.Jpeg", image)

    # Read the image into a byte array
    image_data = open(image_path, "rb").read()

    # process image for analysis in azure
    headers = {'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
               'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'faces'}
    response = requests.post(
        analyze_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    analysis = response.json()  # return result in json format
    print("Azure Image Analysis: ", analysis)

    if len(analysis['faces']) == 0:
        frame = cv2.resize(frame, (round(width / 2), height))  # resize image
        no_face_text = cv2.putText(frame, 'No face detected!', (100, 100), font, 1, (255, 255, 255), 1, cv2.LINE_4)
        print('no face detected')
    else:
        # for each face in the frame
        for face in analysis["faces"]:
            age = face['age']  # age
            gender = face['gender']  # gender

            # coordinates for face rectangle
            x1 = face['faceRectangle']['left']
            y1 = face['faceRectangle']['top']
            x2 = x1 + face['faceRectangle']['width']
            y2 = y1 + face['faceRectangle']['height']

            # print information in console
            print("'{}' of age {} at location {}, {}, {}, {}".format(gender, age, x1, y1, x2, y2))

            start_point = (x1, y1)
            end_point = (x2, y2)

            # draw triangle
            frame = cv2.resize(frame, (round(width / 2), height))  # resize image
            cv2.rectangle(frame, end_point, start_point, color=(255, 30, 30), thickness=3)

            # print gender and age
            gender_text = cv2.putText(frame, 'Gender: ' + gender, (x1, y1 - 40), font, 1, (255, 255, 255), 1,
                                      cv2.LINE_4)
            age_text = cv2.putText(frame, 'Age: ' + str(age), (x1, y1 - 10), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # resize frame
    image2 = cv2.resize(frame, (round(width / 2), height), None, .25, .25)

    # join two frames together
    frame = np.concatenate((image, image2), axis=1)

    # Display frame
    cv2.imshow('----My Face Detector----', frame)

    # i millisecond frame
    key = cv2.waitKey(1)
    # and q key will quit
    if key == ord('q'):
        break
cv2.destroyAllWindows()
# When everything done, release the capture
