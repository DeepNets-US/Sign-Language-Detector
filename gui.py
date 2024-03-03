# Model Imports
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp

# GUI Imports
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Model Loading
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "DEL", "NOTHING", "SPACE"]
model = keras.models.load_model("weights.h5")

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize GUI
top = Tk()
top.geometry('800x600')
top.title("Sign Language Detector")
top.configure(background='#CDCDCD')

# Defining Labels to be displayed
sign_label = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
image_input = Label(top)

def detect_hands_and_crop(img_path, enlargement=50, target_size=(64, 64)):

    # Convert BGR image to RGB
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get results from Mediapipe Hands model
    results = hands.process(rgb_img)

    # Crop and return the detected hands
    cropped_hands = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract bounding box coordinates
            bboxCorners = []
            for landmark in hand_landmarks.landmark:
                x, y, _ = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0]), int(landmark.z * img.shape[1])
                bboxCorners.append((x, y))

            # Calculate enlarged bounding box
            x_min = max(0, min(bboxCorners, key=lambda x: x[0])[0] - enlargement)
            y_min = max(0, min(bboxCorners, key=lambda x: x[1])[1] - enlargement)
            x_max = min(img.shape[1], max(bboxCorners, key=lambda x: x[0])[0] + enlargement)
            y_max = min(img.shape[0], max(bboxCorners, key=lambda x: x[1])[1] + enlargement)

            # Crop the detected hand region and append to the list
            cropped_hand = img[y_min:y_max, x_min:x_max]

            # Resize the cropped hand to the target size (64x64)
            cropped_hand = cv2.resize(cropped_hand, target_size)

            # Normalize the pixel values to be between 0 and 1
            cropped_hand = cv2.cvtColor(cropped_hand, cv2.COLOR_RGB2BGR)
            cropped_hand = cropped_hand / 255.0

            # Append the processed hand to the list
            cropped_hands.append(cropped_hand)

    if len(cropped_hands)==0:
        img = cv2.resize(img, target_size)
        img = img /255.
        img = img[np.newaxis, ...]
        return img

    else:
        return np.array(cropped_hands)


def detect_sign_language(file_path):
    """
    Perform sign language detection on the given image file path.

    Args:
    - file_path (str): The path to the input image file.

    Returns:
    None
    """
    img = detect_hands_and_crop(file_path)

    # Predict the sign language
    pred_sign = classes[tf.argmax(tf.squeeze(model.predict(img, verbose=0)))]

    # Display the results
    print("Detected Sign: {}".format(pred_sign))
    sign_label.configure(foreground="#011638", text="Sign: {}".format(pred_sign))

def show_detect_btn(file_path):
    """
    Show the 'Detect Image' button after an image is uploaded.

    Args:
    - file_path (str): The path to the input image file.

    Returns:
    None
    """
    detect_btn = Button(top, 
        text="Detect Image", 
        command=lambda: detect_sign_language(file_path),
        padx=10, pady=5
    )

    detect_btn.configure(
        background="#364156",
        foreground="white",
        font=('arial', 10, 'bold')
    )

    detect_btn.place(relx=0.79, rely=0.46)

def upload_image():
    """
    Open a file dialog to upload an image and display it.

    Args:
    None

    Returns:
    None
    """
    try:
        filepath = filedialog.askopenfilename()
        uploaded = Image.open(filepath)
        uploaded.thumbnail((
            (top.winfo_width()/2.25), (top.winfo_height()/2.25)
        ))

        image = ImageTk.PhotoImage(uploaded)
        image_input.configure(image=image)
        image_input.image = image
        sign_label.configure(text='')
        show_detect_btn(filepath)

    except Exception as e:
        print("Error:", e)

# Detecting Sign Language
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
upload.pack(side="bottom", expand=True)

# Visual Setting
image_input.pack(side="bottom", expand=True)
sign_label.pack(side="bottom", expand=True) 

# Set Heading / Title
heading = Label(top, text="American Sign Language Detector", pady=20, font=('arial', 20, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Start program
top.mainloop()

# DEBUG
print("Done")
