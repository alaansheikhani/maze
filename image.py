import cv2
from matplotlib import pyplot as plt
import easyocr
from ultralytics import YOLO
from enum import Enum
import streamlit as st
from PIL import Image
import numpy as np



class Detection_class(Enum): 
    JPN = 0
    TABLE = 1

def extract_coordinates(detection_box):
    coordinates_list=  detection_box.xyxy.squeeze(dim=0).tolist()  
    x1 = int(coordinates_list[0])
    y1 = int(coordinates_list[1])
    x2 = int(coordinates_list[2])
    y2 = int(coordinates_list[3])
    return x1,y1,x2,y2

def crop_image(img, x1, y1, x2, y2):
    
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img

def get_boxes(detection_result):
    detection_boxes = detection_result.boxes
    return detection_boxes

def is_class_JPN(detection_box):
    if detection_box.cls == Detection_class.JPN.value:
        return True
    else:
        return False

def is_class_TABLE(detection_box):
    if detection_box.cls == Detection_class.TABLE.value:
        return True
    else:
        return False
    
def get_text_list(img, reader):
    text_list = reader.readtext(img, detail=0)
    return text_list

def extract_jpn(img, dic, reader):
    text_list= get_text_list(img, reader)
    for text in text_list:
        if text not in dic:
            dic["JPN"] = text

def show_img(img):
    cv2.imshow("cropped", img)
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        cv2.destroyAllWindows()

def detect_qr_codes(img, qr_detector):

    qr_list=qr_detector.detectAndDecodeMulti(img)
    retval, decoded_info, points, straight_qrcode = qr_list
    return retval, decoded_info, points


def overlay_detection_boxes(img, points):
    new_img= cv2.polylines(img, points.astype(int), True, (0, 255, 0), 3)
    
    return new_img
    



def main():

    st.title('Image detection App')

    uploaded_file = st.file_uploader('Upload an image')

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        img = np.array(img)

        if st.button('Process'):

            # img_path = "/Users/alaansheikhani/Work/maze/images/first.jpeg"

            # img = cv2.imread(img_path)

            my_dic = dict.fromkeys(["JPN"])
            
            model = YOLO("best_2.pt")
            # qr_detector = cv2.QRCodeDetector()

            reader = easyocr.Reader(['en'])  # Specify language(s), e.g., 'en' for English
            
            #################
            
            # retval, decoded_info, points = detect_qr_codes(img, qr_detector)

        
            # if retval==True: 

            #     img = overlay_detection_boxes(img, points)


            #     for num, qr_info in enumerate(decoded_info):
            #         my_dic[f"QR_{num}"]=qr_info

            

            prediction_results = model.predict(source=img,conf=0.3)

            try:
                if len(prediction_results)==0:
                    # st.error("No boxes been detected!", "cat" )
                    st.write("No boxes been detected!")
                    raise Exception("No boxes been detected!")
                    
                
                else:
            
                    for result in prediction_results:

                        detection_boxes = get_boxes(result)

                        for detection_box in detection_boxes:

                            if is_class_JPN(detection_box)==True:

                                x1, y1, x2, y2 = extract_coordinates(detection_box)

                                annotated_img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                                cropped_img = crop_image(img, x1, y1, x2, y2)

                                extract_jpn(cropped_img, my_dic, reader)

                                # show_img(cropped_img)
                                st.image(annotated_img, caption='detection result', use_column_width=True)

                            else:
                                st.info("No JPN can be read!")
            except Exception as err: 
                print(err) 


            st.json(my_dic)
            # print(my_dic)


if __name__ == "__main__":
    main()  