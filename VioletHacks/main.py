from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from google import genai
import subprocess
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from squat import SquatFormChecker, get_knee_angle, get_hip_angle
import secrets


interperter = tf.lite.Interpreter(model_path="3.tflite") #downloaded model
interperter.allocate_tensors() 
squat = False

def form_detection():
    # access webcam and make detections
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    window_name = 'Press \'q\' to quit'

    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    squat_checker = SquatFormChecker()  # Create an instance of SquatFormChecker
    form_score = 100  # Initialize form score



    while cap.isOpened():
        ret, frame = cap.read() #reads frame from webcam

        # flip the camera for our sanity
        flipped_frame = cv2.flip(frame, 1)

        #Reshape iamge
        img = flipped_frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
        input_image = tf.cast(img, dtype=tf.float32)


        #Setup input and output
        input_details = interperter.get_input_details()
        out_details = interperter.get_output_details()

        #Make predictions
        # set input details equal to image
        interperter.set_tensor(input_details[0]['index'], np.array(input_image))
        # make predictions
        interperter.invoke()
        # interpert output details
        keypoints_with_scores = interperter.get_tensor(out_details[0]['index'])


        if (squat):
            # Extract relevant keypoints
            left_hip = keypoints_with_scores[0][0][11]
            left_knee = keypoints_with_scores[0][0][13]
            left_ankle = keypoints_with_scores[0][0][15]
            left_shoulder = keypoints_with_scores[0][0][5]

            # Calculate angles
            knee_angle = get_knee_angle(left_hip, left_knee, left_ankle)
            hip_angle = get_hip_angle(left_shoulder, left_hip, left_knee)

            # Check the squat form
            form_score_temp, state = squat_checker.check_squat(knee_angle, hip_angle)

            if form_score_temp is not None and form_score is not None and form_score >= form_score_temp:
                form_score = form_score_temp

            if form_score is not None:
                print(f"Form Score: {form_score}, Form Score Temp: {form_score_temp}, State: {state}, Knee Angle {knee_angle}, Hip angle {hip_angle}")

        # Rendering
        draw_connections(flipped_frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(flipped_frame, keypoints_with_scores, 0.4)

        cv2.imshow(window_name, flipped_frame) # rending frame

        if (cv2.waitKey(10) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() #close frame
    print(f"Form Score Calculated: {form_score}")  # Debugging output
    return form_score


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape # shape for x y c
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1])) # keypoints * frame shape for locations

    for kp in shaped: # loop through values in shaped
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,128,255), -1) # draw circles

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5 ,7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2] # coords of where points are located

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,128), 2)

#python -m flask --app main run 
#Kills the port before running

def start(port):
    try:
        # Get active ports and processes
        netstat_cmd = ["netstat", "-ano"]
        netstat_output = subprocess.check_output(netstat_cmd, text=True)

        # Find the process using the specified port
        for line in netstat_output.splitlines():
            if f":{port}" in line:
                fields = line.split()
                pid = fields[-1]  # PID is the last column
                print(f"Found PID {pid} using port {port}")

                # Kill the process by PID
                taskkill_cmd = ["taskkill", "/PID", pid, "/F"]
                subprocess.run(taskkill_cmd, check=True)
                print(f"Successfully killed process on port {port} (PID: {pid})")
                return
        
        print(f"No process found using port {port}.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error killing process: {e}")
        
start("5000")


app = Flask(__name__)
app.secret_key = 'your_super_secret_key'

@app.route('/WorkoutSuggestor',methods=['GET', 'POST'])
def WorkoutSuggestor():
    return render_template("WorkoutSuggestor.html")

@app.route('/submitWS', methods=['POST'])
def submit():
    muscle_groups = request.form.getlist('muscle')
    intensity = request.form.get('intensity') 
    focus = request.form.get('focus') 

    if focus == "Both":
        focus = "Both Hypertrophy and Stregnth"
    if not muscle_groups:
        return "No muscle groups selected.<br>"

    prompt = f"I want to workout these muscle(s): {', '.join(muscle_groups)} with an {intensity} intensity level. Please make it a list with rep range and sets and only in text, no **."
    prompt = prompt + " Low = 3 exercise, Medium = 4, High = 5"
    prompt = prompt + "I want the workout to be focused on " + focus
    client = genai.Client(api_key="AIzaSyCI3xFoQSsuAWIj6bPpfoumUZm3kVYwNog")
    response = client.models.generate_content(
        model="gemini-1.5-flash", contents=prompt 
    )
    
    return response.text

@app.route('/FormWatcher')
def FormWatcher():
    return render_template("WorkoutTracker2.html")


@app.route('/callScript', methods=['POST'])
def callScript():
    
    workout = request.get_json().get('workout')
    if (workout == "squat"):
        squatWorkout = True
    else:
        squatWorkout = False

    form_score = form_detection()

    if form_score is None:
        form_score = "No score detected. Please try again."

    # Store score in session
    session['form_score'] = form_score

    # Redirect to display the score
    return redirect(url_for('display_score'))

@app.route('/display_score')
def display_score():
    form_score = session.get('form_score', None)
    
    if form_score is None:
        form_score = "No score available. Please try again."

    return render_template("WorkoutTracker2.html", score=form_score)
@app.route("/")
def main():
    return render_template("mainPage.html")
