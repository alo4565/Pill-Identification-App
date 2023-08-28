# import necessary libraries
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import sqlite3
import os
import math


# create database connection
conn = sqlite3.connect('Pill_Identification.db')

cur = conn.cursor()
# create tabel for our pill id database
# cur.execute(""" CREATE TABLE Pill_Identification (
#                 pill_name VARCHAR(60) NOT NULL UNIQUE PRIMARY KEY,
#                 pill_manufacture VARCHAR(30),
#                 pill_shape VARCHAR(30),
#                 pill_usage TEXT
#
# );
#
# """)
#
# conn.commit()


# pill classification class
class PillClassification:

    # define class constructor which will be responsible for GUI
    def __init__(self):

        # create main window
        self.root = tk.Tk()
        self.root.title("Pill Classification")
        self.root.geometry("1300x700")
        self.root.configure()

        # create first tab for image acquisition
        self.pill_classification_notebook = ttk.Notebook(self.root)
        self.image_acquisition = tk.Frame(self.pill_classification_notebook)

        self.image_acquisition.configure(bg="#F5F2DA")

        self.pill_classification_notebook.add(self.image_acquisition, text="Image Acquisition")

        self.pill_classification_notebook.pack(fill="both", expand=1)

        # self.pill_classification_notebook.bind("<<NotebookTabChanged>>", self.update_current_tab)
        # self.current_tab_index = 0

        # first tab title label
        self.program_header = tk.Label(self.image_acquisition, text="Pill Classification Program", font=("Arial", 24), bg="#F5F2DA")
        self.program_header.pack(pady=10)
        # Graphics window
        self.imageFrame = tk.Frame(self.image_acquisition, width=600, height=500, bg="#F5F2DA")
        self.imageFrame.pack(side=tk.TOP, padx=10, pady=2)

        # Capture video frames
        self.cap = cv.VideoCapture(0)

        # display 1 and display 2 will show video feed
        self.display1 = tk.Label(self.imageFrame)
        self.display1.grid(row=0, column=0, padx=10, pady=2)  # Display 1
        self.display2 = tk.Label(self.imageFrame)
        self.display2.grid(row=0, column=1)  # Display 2

        # create has been captured variable for capturing image later on
        self.has_been_captured = False

        # create frame for our main buttons
        self.image_buttons = tk.Frame(self.image_acquisition, bg="#F5F2DA")
        self.image_buttons.pack()

        # choose image button
        self.choose_image_label = tk.Label(self.image_buttons, text="Choose Pill: ", bg="#F5F2DA")
        self.choose_image_label.grid(row=0, column=0, padx=5)

        # pill dropdown menu button
        try:
            self.selected_pill_name = tk.StringVar()
            cur.execute("SELECT pill_name from Pill_Identification")
            result = cur.fetchall()
            self.pill_names = [row[0] for row in result]
            self.pill_option_dropdown = tk.OptionMenu(self.image_buttons, self.selected_pill_name, *self.pill_names)
            self.pill_option_dropdown.config(bg="#6A94BB")
            self.pill_option_dropdown.grid(row=0, column=1)
        except TypeError:
            self.pill_names = ["Enter Pill Information First to Update List"]
            self.pill_option_dropdown = tk.OptionMenu(self.image_buttons, self.selected_pill_name, *self.pill_names)
            self.pill_option_dropdown.config(bg="#6A94BB")
            self.pill_option_dropdown.grid(row=0, column=1)
        # placeholder for captured image that is used for saving
        self.captured_image = None
        # placeholder for checking for uploaded images
        self.uploaded_image_check = False
        # upload image button
        self.upload_image_button = tk.Button(self.image_buttons, text="Upload Image", bg="#6A94BB", activebackground="#AAC5DF", command=self.upload_image)
        self.upload_image_button.grid(row=0, column=4, pady=10)

        # image count for adding files to directory later on
        self.image_count = 0
        # create save button
        self.save_image_button = tk.Button(self.image_buttons, text="Save Image", bg="#6A94BB", activebackground="#AAC5DF", command=self.save_image)
        self.save_image_button.grid(row=0, column=5, pady=10)

        # create capture button
        self.screenshot_button = tk.Button(self.image_buttons, text="Capture Image", bg="#6A94BB", activebackground="#AAC5DF", command=self.capture_image)
        self.screenshot_button.grid(row=0, column=2, pady=10)

        # create classify button
        self.classify_button = tk.Button(self.image_buttons, text="Classify Image", bg="#6A94BB", activebackground="#AAC5DF", command=self.classify_image)
        self.classify_button.grid(row=0, column=6)

        # create reset button
        self.reset_image_button = tk.Button(self.image_buttons, text="Reset Image", bg="#6A94BB", activebackground="#AAC5DF", command=self.reset_image)
        self.reset_image_button.grid(row=0, column=7)

        # create frame for entering pill information, which will be stored in database
        self.image_labels = tk.LabelFrame(self.image_acquisition, text="Enter Pill Information:", font=("Arial", 14), labelanchor='n', bd=4, bg="#F7F5E4")
        self.image_labels.pack(side=tk.LEFT, padx=5)

        # pill name label and entry
        self.pill_name_label = tk.Label(self.image_labels, text="Enter Pill Name:", bg="#F7F5E4")
        self.pill_name_label.grid(row=0, column=0)

        self.pill_name = tk.Entry(self.image_labels, width=53)
        self.pill_name.grid(row=0, column=1)

        # pill manufacture label and entry
        self.pill_manufacture_label = tk.Label(self.image_labels, text="Enter Pill Manufacture: ", bg="#F7F5E4")
        self.pill_manufacture_label.grid(row=1, column=0)

        self.pill_manufacture = tk.Entry(self.image_labels, width=53)
        self.pill_manufacture.grid(row=1, column=1)

        # pill shape label and entry
        self.pill_shape_label = tk.Label(self.image_labels, text="Enter Pill Shape:", bg="#F7F5E4")
        self.pill_shape_label.grid(row=2, column=0)

        self.pill_shape = tk.Entry(self.image_labels, width=53)
        self.pill_shape.grid(row=2, column=1)

        # pill usage label and entry
        self.pill_usage_label = tk.Label(self.image_labels, text="Enter Pill Usage", bg="#F7F5E4")
        self.pill_usage_label.grid(row=3, column=0)

        self.pill_usage = tk.Text(self.image_labels, width=40, height=8)
        self.pill_usage.grid(row=3, column=1, pady=2)

        # create submit button for submitting form
        self.submit_button = tk.Button(self.image_labels, text="Submit", font=("Arial", 12),
                                       command=self.submit_information)
        self.submit_button.grid(row=3, column=3, padx=5)

        # create clear button for clearing form
        self.clear_button = tk.Button(self.image_labels, text="Clear", font=("Arial", 12),
                                      command=self.clear_information)
        self.clear_button.grid(row=2, column=3, padx=5)

        # create second tab for training model
        self.feature_extraction = tk.Frame(self.pill_classification_notebook)

        self.feature_extraction.configure(bg="#F5F2DA")
        self.feature_extraction.pack()

        self.pill_classification_notebook.add(self.feature_extraction, text="Feature Extraction")
        self.feature_extraction_label = tk.Label(self.feature_extraction, text="Feature Extraction", font=("Arial", 24), bg="#F5F2DA")
        self.feature_extraction_label.pack()

        # create frame for our holding default utrgv banner image
        self.feature_extraction_image_frame = tk.Frame(self.feature_extraction, bg="#F5F2DA")
        self.feature_extraction_image_frame.pack(padx=10, pady=10)

        self.default_image = ImageTk.PhotoImage(Image.open("Images/utrgv_default_image.jpg").resize((400, 300), Image.LANCZOS))
        self.default_image_label = tk.Label(self.feature_extraction_image_frame, image=self.default_image)
        self.default_image_label.grid(row=0, column=0, padx=20, pady=10)

        # create frame for holding training operations
        self.feature_extraction_operation_frame = tk.LabelFrame(self.feature_extraction, text="Training Operations", font=("Arial", 14), labelanchor='n', bd=4, bg="#F7F5E4")
        self.feature_extraction_operation_frame.pack()

        # create label for choose pill
        self.operation_instruction_label = tk.Label(self.feature_extraction_operation_frame, text="Choose Pill to train model with:", font=("Arial", 12), bg="#F7F5E4")
        self.operation_instruction_label.grid(row=0, column=1)

        self.choose_pill_label = tk.Label(self.feature_extraction_operation_frame, text="Choose Pill: ", font=("Arial", 12), bg="#F7F5E4")
        self.choose_pill_label.grid(row=1, column=0)

        # dropdown menu button showing pills available to train
        try:
            self.feature_extract_selected_pill_name = tk.StringVar()
            cur.execute("SELECT pill_name from Pill_Identification")
            result = cur.fetchall()
            self.feature_extract_pill_names = [row[0] for row in result]
            self.feature_extract_pill_option_dropdown = tk.OptionMenu(self.feature_extraction_operation_frame, self.selected_pill_name, *self.pill_names)
            self.feature_extract_pill_option_dropdown.grid(row=1, column=1)
        except TypeError:
            self.feature_extract_pill_names = ["Enter Pill Information First to Update List"]
            self.feature_extract_pill_option_dropdown = ttk.OptionMenu(self.image_buttons, self.selected_pill_name, *self.pill_names)
            self.feature_extract_pill_option_dropdown.grid(row=1, column=1, padx=5)


        # create training button
        self.start_training_button = tk.Button(self.feature_extraction_operation_frame, text="Start Training", font=("Arial", 12),  command=self.train_model)
        self.start_training_button.grid(row=1, column=2, padx=10)

        self.training_finished_frame = tk.Frame(self.feature_extraction)
        self.training_finished_frame.pack()
        self.training_finished_label = ""

        # create frame for showing the closest image match
        self.image_match_frame = tk.LabelFrame(self.image_acquisition, text="Closest Match", font=("Arial", 14), labelanchor='n', bd=4, bg="#F7F5E4")
        self.image_match_frame.pack(side=tk.RIGHT, padx=25, pady=5)

        # placeholder values for image match
        self.pill_name_label_match = None
        self.pill_name_match = None

        # placeholder values for image match
        self.pill_manufacture_label_match = None
        self.pill_manufacture_match = None

        # placeholder values for image match
        self.pill_shape_match = None
        self.pill_shape_match = None

        # placeholder values for image match
        self.piil_usage_label_match = None
        self.pill_usage_match = None

        # placeholder values for image match
        self.pill_image_match_label = None
        self.pill_image_match = None

        # call function to display video feed
        self.show_frame_identification()  # Display

        self.root.mainloop()  # Starts GUI


    # create function to show video feed
    def show_frame_identification(self):

        _, frame = self.cap.read()
        cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        img = img.resize((400, 300))
        imgtk = ImageTk.PhotoImage(image=img)
        if self.has_been_captured is False and self.uploaded_image_check is False:
            self.display1.imgtk = imgtk #Shows frame for display 1
            self.display1.configure(image=imgtk)
            self.display2.imgtk = imgtk #Shows frame for display 2
            self.display2.configure(image=imgtk)
        elif self.has_been_captured is True and self.uploaded_image_check is False:
            self.display1.imgtk = imgtk #Shows frame for display 1
            self.display1.configure(image=imgtk)
            captured_image = Image.open("Images/temp/captured_image.jpg").resize((400, 300))
            captured_image_tk = ImageTk.PhotoImage(captured_image)
            self.display2.configure(image=captured_image_tk)
            self.display2.image = captured_image_tk
            self.has_been_captured = True
        elif self.has_been_captured is False and self.uploaded_image_check is True:
            self.display1.imgtk = imgtk #Shows frame for display 1
            self.display1.configure(image=imgtk)
            uploaded_image = Image.open("Images/temp/uploaded_image.jpg").resize((400, 300))
            uploaded_image_tk = ImageTk.PhotoImage(uploaded_image)
            self.display2.configure(image=uploaded_image_tk)
            self.display2.image = uploaded_image_tk

        self.root.after(10, self.show_frame_identification)

    # create function to reset image
    def reset_image(self):

        self.uploaded_image_check = False
        self.has_been_captured = False

    # create function to capture image
    def capture_image(self):

        _, frame = self.cap.read()
        cv.imwrite("Images/temp/captured_image.jpg", frame)

        self.captured_image = Image.open("Images/temp/captured_image.jpg").resize((400, 300))
        copy_of_captured_image = self.captured_image
        captured_image_tk = ImageTk.PhotoImage(copy_of_captured_image)
        self.display2.configure(image=captured_image_tk)
        self.display2.image = captured_image_tk
        self.has_been_captured = True
        self.uploaded_image_check = False

    # create function to upload image
    def upload_image(self):

        # choose image (jpg, png, all files)
        uploaded_image_filepath = filedialog.askopenfilename(initialdir="/", title="Select an Image", filetypes=(
            ("All Files", "*.*"), ("PNG Files", "*.png"), ("JPG Files", "*.jpg")))

        opencv_uploaded_image = cv.imread(uploaded_image_filepath)
        cv.imwrite("Images/temp/uploaded_image.jpg", opencv_uploaded_image)

        # resize chosen image from filepath
        self.has_been_captured = False
        self.uploaded_image_check = True
        # update after image to before image

    # create function to clear information from form
    def clear_information(self):

        self.pill_name.delete(0, tk.END)
        self.pill_manufacture.delete(0, tk.END)
        self.pill_shape.delete(0, tk.END)
        self.pill_usage.delete('1.0', tk.END)

    # create function for updating dropdown menu
    def update_dropdown_menu(self):
        cur.execute("SELECT pill_name FROM Pill_Identification")
        result = cur.fetchall()
        self.pill_names = [row[0] for row in result]

        # Clear existing options
        menu = self.pill_option_dropdown['menu']
        menu.delete(0, 'end')

        # Add new options
        for name in self.pill_names:
            menu.add_command(label=name, command=lambda value=name: self.selected_pill_name.set(value))

        # Set the first option as the default selection
        if self.pill_names:
            self.selected_pill_name.set(self.pill_names[0])

    # create function to submit information for form
    def submit_information(self):

        pill_name = self.pill_name.get()
        pill_manufacture = self.pill_manufacture.get()
        pill_shape = self.pill_shape.get()
        pill_usage = self.pill_usage.get("1.0", tk.END)

        if pill_name != "" and pill_shape != "" and pill_manufacture != "" and pill_usage != "":

            cur.execute(f""" INSERT INTO Pill_Identification VALUES ('{pill_name}', '{pill_manufacture}', '{pill_shape}',
                                                                    '{pill_usage}')""")

            conn.commit()

            self.update_dropdown_menu()

            self.clear_information()

    # create function to save image
    def save_image(self):

        image_to_save = None

        if self.has_been_captured is True and self.uploaded_image_check is False:
            image_to_save = self.captured_image
        elif self.has_been_captured is False and self.uploaded_image_check is True:
            uploaded_image_filepath = "Images/temp/uploaded_image.jpg"
            image_to_save = Image.open(uploaded_image_filepath)

        selected_pill_name = self.selected_pill_name.get()

        directory = f"Images/{selected_pill_name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{selected_pill_name}.jpg"

        if not os.path.exists(os.path.join(directory, filename)):
            image_path = os.path.join(directory, filename)
            image_to_save.save(image_path)
            self.image_count = 0
        else:
            self.image_count += 1
            filename = f"{selected_pill_name}_{self.image_count}.jpg"
            image_path = os.path.join(directory, filename)
            image_to_save.save(image_path)

    # create function to classify image
    def classify_image(self):

        classify_image_fv = []

        print("has been captured", self.has_been_captured)
        print("uploaded image check", self.uploaded_image_check)

        if self.has_been_captured is False and self.uploaded_image_check is True:
            image_filepath = "Images/temp/uploaded_image.jpg"
            image_to_classify = cv.imread(image_filepath)
            cv.imwrite("Images/classify_images/image_to_classify.jpg", image_to_classify)
            feature_vector = self.shape_feature_extractor(image_filepath)
            feature_vector = feature_vector.flatten()

            feature_vector_filename = "feature_data.txt"

            file = open("Images/classify_images/classify_feature_data.txt", 'w')
            for i in range(len(feature_vector)):
                if i <= 6:
                    file.write(f"huMoment{i} = {feature_vector[i]} \n")
                if i == 7:
                    file.write(f"Compactness = {feature_vector[i]} \n")
                if i == 8:
                    file.write(f"Hue = {feature_vector[i]} \n")
                if i == 9:
                    # average_feature_vector[i] = str(average_feature_vector[i])
                    file.write(f"Saturation = {feature_vector[i]} \n")

            file.close()

        elif self.has_been_captured is True and self.uploaded_image_check is False:
            image_filepath = "Images/temp/captured_image.jpg"
            image_to_classify = cv.imread(image_filepath)
            cv.imwrite("Images/classify_images/image_to_classify.jpg", image_to_classify)
            feature_vector = self.shape_feature_extractor(image_filepath)
            feature_vector = feature_vector.flatten()

            feature_vector_filename = "feature_data.txt"

            file = open("Images/classify_images/classify_feature_data.txt", 'w')
            for i in range(len(feature_vector)):
                if i <= 6:
                    file.write(f"huMoment{i} = {feature_vector[i]} \n")
                if i == 7:
                    file.write(f"Compactness = {feature_vector[i]} \n")
                if i == 8:
                    file.write(f"Hue = {feature_vector[i]} \n")
                if i == 9:
                    # average_feature_vector[i] = str(average_feature_vector[i])
                    file.write(f"Saturation = {feature_vector[i]} \n")

            file.close()

        self.gather_fv()

    # gather_fv function gathers fv features
    def gather_fv(self):

        np.set_printoptions(suppress=True)

        pill_dict = {}

        # get classify image vector
        classify_image_filepath = "Images/classify_images/classify_feature_data.txt"
        f = open(classify_image_filepath, 'r')
        classify_image_fv = np.loadtxt(classify_image_filepath,  usecols=2, dtype='float')

        cur.execute("SELECT pill_name from Pill_Identification")

        result = cur.fetchall()

        list_of_pill_names = []

        for name in result:
            name = name[0]
            list_of_pill_names.append(name)

        for i in range(len(list_of_pill_names)):
            filepath = f"Images/{list_of_pill_names[i]}/feature_data.txt"
            pill_dict[list_of_pill_names[i]] = np.loadtxt(filepath, skiprows=1, usecols=2, dtype='float')

        euc_dist_dict = {}

        for key, value in pill_dict.items():
            euc_dist = cv.norm(value, classify_image_fv, cv.NORM_L2)
            euc_dist_dict[key] = euc_dist

        image_match = min(euc_dist_dict, key=euc_dist_dict.get)

        self.display_match(image_match)

    # create image to display image match
    def display_match(self, image_match):

        cur.execute("SELECT * FROM Pill_identification WHERE pill_name = ?", (image_match, ))

        result = cur.fetchall()

        pill_name_match = tk.StringVar()
        pill_manufacture_match = tk.StringVar()
        pill_shape_match = tk.StringVar()
        pill_usage_match = result[0][3]

        pill_name_match.set(result[0][0])
        pill_manufacture_match.set(result[0][1])
        pill_shape_match.set(result[0][2])

        self.pill_name_label_match = tk.Label(self.image_match_frame, text="Pill Name:", bg="#F7F5E4")
        self.pill_name_label_match.grid(row=0, column=1)

        self.pill_name_match = tk.Entry(self.image_match_frame, width=53, textvariable=pill_name_match, state="readonly")
        self.pill_name_match.grid(row=0, column=2)

        self.pill_manufacture_label_match = tk.Label(self.image_match_frame, text="Pill Manufacture: ", bg="#F7F5E4")
        self.pill_manufacture_label_match.grid(row=1, column=1)

        self.pill_manufacture_match = tk.Entry(self.image_match_frame, width=53, textvariable=pill_manufacture_match, state="readonly")
        self.pill_manufacture_match.grid(row=1, column=2)

        self.pill_shape_label_match = tk.Label(self.image_match_frame, text="Pill Shape:", bg="#F7F5E4")
        self.pill_shape_label_match.grid(row=2, column=1)

        self.pill_shape_match = tk.Entry(self.image_match_frame, width=53, textvariable=pill_shape_match, state="readonly")
        self.pill_shape_match.grid(row=2, column=2)

        self.pill_usage_label_match = tk.Label(self.image_match_frame, text="Pill Usage", bg="#F7F5E4")
        self.pill_usage_label_match.grid(row=3, column=1)

        self.pill_usage_match = tk.Text(self.image_match_frame, width=40, height=8)
        self.pill_usage_match.insert('1.0', pill_usage_match)
        self.pill_usage_match.config(state='disabled')
        self.pill_usage_match.grid(row=3, column=2, pady=2)

        self.pill_image_match = ImageTk.PhotoImage(Image.open(f"Images/{result[0][0]}/{result[0][0]}.jpg").resize((200, 150)))
        self.pill_image_match_label = tk.Label(self.image_match_frame, image=self.pill_image_match)
        self.pill_image_match_label.grid(row=3, column=0)

    # this function is responsible for prepping images for feature extraction
    def preprocess(self, image):
        # convert image to black and white
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # noise reduction
        blur = cv.GaussianBlur(gray_image, (5, 5), 0)

        # otsu optimal thesholding
        ret_th, thresholded_image = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # self.show_image(thresholded_image)

        # Morphology to get rid of little holes if any
        kernel = np.ones((3, 3), np.uint8)
        iterations = 2
        erosion = cv.erode(thresholded_image, kernel, iterations)
        image = cv.dilate(erosion, kernel, iterations)

        dilated = cv.dilate(image, kernel, iterations)
        processed_image = cv.erode(dilated, kernel, iterations)
        return processed_image

    # this function is responsible for extracting features from image
    def shape_feature_extractor(self, image):
        # initilize the feature vector:
        # I added Hue and Saturation, so it became 10 instead of 9
        feature_vector = np.ones((10, 1), dtype=float)

        img = cv.imread(image)
        print('channel', img.shape)

        # pre-process image
        img1 = self.preprocess(img)

        contours, hierarchy = cv.findContours(img1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # max area
        areas = []
        for cnt in contours:
            areas.append(cv.contourArea(cnt))
        areasArray = np.array(areas)
        index = areasArray.argmax(axis=0)
        cnt = contours[index]
        # draw specific contour
        imgofpillonly = cv.drawContours(img, [cnt], 0, (0, 255, 255), 1)

        # Features
        M = cv.moments(cnt)

        # Calculate Hu Moments
        huMoments = cv.HuMoments(M)

        # Log scale hu moments
        feature_vector = np.ones((10, 1), dtype=float)

        for i in range(0, 7):
            huMoments[i] = -1 * np.sign(huMoments[i]) * math.log10(abs(huMoments[i]))
            feature_vector[i] = huMoments[i]

        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)

        # calculate compactness
        # compactness
        # R=(4piA)/P^2
        r = 4 * math.pi * area / (perimeter * perimeter)
        feature_vector[7] = r

        # Center of the pill
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print("cx=" + str(cx))

        print("cy=" + str(cy))
        img_with_centerofmass = img.copy()
        # check if it is the correct value on the original image
        cv.circle(img_with_centerofmass, (cx, cy), 2, (0, 0, 255), 2)
        # self.show_image(img_with_centerofmass)

        # pick up 9 neighbors around the cx, cy from the original bgr image
        cx, cy = (cy, cx)
        # make neighbor to pick a parameter
        delta = 2

        # Get tiny window around center of mass of the pills
        tiny_bgr = img[cx - delta + 1:cx + delta, cy - delta:cy + delta + 1]
        print('tingybgr', tiny_bgr)
        # self.show_image(tiny_bgr)

        # convert batch to HSV
        tiny_hsv = cv.cvtColor(tiny_bgr, cv.COLOR_BGR2HSV)
        print('tiny hsv', tiny_hsv)

        # average and get one h & s << seems that both h and s are important
        # and pick the h value only
        tiny_h_only = tiny_hsv[:, :, 0]  # only the hue channel
        print('h vals', tiny_h_only)
        tiny_s_only = tiny_hsv[:, :, 1]  # only the saturation channel

        # median is giving me better results
        tiny_h_v = np.median(tiny_h_only)
        tiny_s_v = np.median(tiny_s_only)

        # Check if the color is correct by going back to rgb
        h = (tiny_h_v * np.ones((256, 256), dtype=np.uint8)).astype(np.uint8)
        print(h.shape)
        s = (tiny_s_v * np.ones((256, 256), dtype=np.uint8)).astype(np.uint8)
        print(s.shape)
        v = (255 * np.ones((256, 256), dtype=np.uint8)).astype(np.uint8)
        print(v.shape)
        print(v.dtype)
        hsv = cv.merge([h, s, v])
        reversergb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        # self.show_image(reversergb)

        # put in feature vector
        feature_vector[8] = tiny_h_v
        feature_vector[9] = tiny_s_v

        return feature_vector

    # create function to train model with image features
    def train_model(self):

        if self.training_finished_label != "":
            self.training_finished_label.destroy()


        selected_pill_name = self.selected_pill_name.get()
        list_vectors = []

        directory = f"Images/{selected_pill_name}"

        vector_data_check = f"Images/{selected_pill_name}/feature_data.txt"
        print(os.path.isfile(vector_data_check))
        if os.path.isfile(vector_data_check) is False:
            if selected_pill_name != "":
                for file in os.listdir(directory):
                    filepath = os.path.join(directory, file).replace("\\", "/")
                    filepath_for_writing = filepath
                    print(filepath)
                    feature_vector = self.shape_feature_extractor(filepath)
                    list_vectors.append(feature_vector)

            average_feature_vector = np.mean(list_vectors, axis=0)
            average_feature_vector = average_feature_vector.flatten()

            feature_vector_filename = "feature_data.txt"

            file = open(os.path.join(directory, feature_vector_filename), 'w')
            file.write(f"Pill Name: {selected_pill_name} \n")
            for i in range(len(average_feature_vector)):

                if i <= 6:
                    # average_feature_vector[i] = str(average_feature_vector[i])
                    file.write(f"huMoment{i} = {average_feature_vector[i]} \n")
                if i == 7:
                    # average_feature_vector[i] = str(average_feature_vector[i])
                    file.write(f"Compactness = {average_feature_vector[i]} \n")
                if i == 8:
                    # average_feature_vector[i] = str(average_feature_vector[i])
                    file.write(f"Hue = {average_feature_vector[i]} \n")
                if i == 9:
                    # average_feature_vector[i] = str(average_feature_vector[i])
                    file.write(f"Saturation = {average_feature_vector[i]} \n")

            file.close()

            self.training_finished_label = tk.Label(self.training_finished_frame, text="Training Complete",
                                                    font=("Arial", 14))
            self.training_finished_label.pack(padx=25, pady=25, anchor=tk.CENTER)

        else:
            self.training_finished_label = tk.Label(self.training_finished_frame, text="Pill Already Trained",
                                                        font=("Arial", 14))
            self.training_finished_label.pack(padx=25, pady=25, anchor=tk.CENTER)


# Create instance of class and run
PillClassification()
