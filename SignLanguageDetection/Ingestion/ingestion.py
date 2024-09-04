import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import sys
import tensorflow as tf

class ShowVideoKeypoints:
    def __init__(self) -> None:
        self.mp_holistic = mp.solutions.holistic # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        pass
    
    def mediapipe_detection(self,image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    
    def draw_landmarks(self,image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION) # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

    
    def run(self,):

        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = self.mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                self.draw_landmarks(image, results)

                # Show to screen
                cv2.putText(image, 'Press q to stop the capturing', (120,200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("quitting")
                    break
    
            cap.release()
            print("Video capture released.")  # Debug: Confirm release

            cv2.destroyWindow('OpenCV Feed')
            cv2.waitKey(1)  # Small delay to ensure the window closes, It allows the OpenCV window (cv2.imshow()) to update, display the image, and handle GUI events (like closing or resizing the window).
            
            # Close the OpenCV windows
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            print("All windows destroyed.")  # Debug: Confirm window closure
                    
        

        
class ExtractKeyPoints():
    def __init__(self,path:str, no_sequences:int, sequence_length:int, start_folder:int, actions:list[str] ) -> None:
        self.mp_holistic = mp.solutions.holistic # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities
         # Path to root dir
        self.DATA_PATH = path
        # number of  videos worth of data
        self.no_sequences = no_sequences
        # length of each video
        self.sequence_length = sequence_length
        
        self.start_folder = start_folder
        # list of actions to store
        self.actions = actions
        
    def extract(self):
        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Loop through actions
            for action in self.actions:
                # Loop through sequences aka videos
                for sequence in range(self.no_sequences):
                    # Loop through video length aka sequence length
                    for frame_num in range(self.sequence_length):

                        # Read feed
                        ret, frame = cap.read()

                        # Make detections
                        image, results = self.mediapipe_detection(frame, holistic)
        #                 print(results)

                        # Draw landmarks
                        self.draw_landmarks(image, results)
                        
                        # NEW Apply wait logic
                        if frame_num == 0: 
                            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                        else: 
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                        
                        # NEW Export keypoints
                        keypoints = self.extract_keypoints(results)
                        npy_path = os.path.join(self.DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                            
            
            cap.release()
            print("Video capture released.")  # Debug: Confirm release

            cv2.destroyWindow('OpenCV Feed')
            cv2.waitKey(1)  # Small delay to ensure the window closes, It allows the OpenCV window (cv2.imshow()) to update, display the image, and handle GUI events (like closing or resizing the window).
            
            # Close the OpenCV windows
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            print("All windows destroyed.") 
            
    def mediapipe_detection(self,image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
            
    def draw_landmarks(self,image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION) # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

    
    def extract_keypoints(self,results):
        ''' error handling since if there is no object i have a problem '''
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    
    
    
    
    def Return_dataset(self):
        self.label_map = {label:num for num, label in enumerate(self.actions)}

        sequences, labels = [], []
        for action in self.actions:
            for sequence in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(self.label_map[action])
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        labels = tf.keras.utils.to_categorical(labels).astype(int)
        return sequences, labels
            



class TestTime():
    def __init__(self, model, actions) -> None:
        self.mp_holistic = mp.solutions.holistic # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.model = model
        self.actions = actions
    
    def mediapipe_detection(self,image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    
    def draw_landmarks(self,image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION) # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

    def extract_keypoints(self,results):
        ''' error handling since if there is no object i have a problem '''
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    
    
    def prob_viz(self,res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output_frame

    def run(self):

        # 1. New detection variables
        sequence = []
        sentence = []
        threshold = 0.8

        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with self.mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = self.mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                self.draw_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = self.extract_keypoints(results)
                # sequence.insert(0,keypoints)
                # sequence = sequence[:30]
                sequence.append(keypoints)
                sequence = sequence[-50:]
                
                if len(sequence) == 50:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(self.actions[np.argmax(res)])
                    
                    
                #3. Viz logic
                    if res[np.argmax(res)] > threshold: 
                            
                        if len(sentence) > 0: 
                            if self.actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(self.actions[np.argmax(res)])
                        else:
                            sentence.append(self.actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = self.prob_viz(res, self.actions, image, self.colors)
                    
                    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        
        
        cap.release()
        print("Video capture released.")  # Debug: Confirm release

        cv2.destroyWindow('OpenCV Feed')
        cv2.waitKey(1)  # Small delay to ensure the window closes, It allows the OpenCV window (cv2.imshow()) to update, display the image, and handle GUI events (like closing or resizing the window).
        
        # Close the OpenCV windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("All windows destroyed.")  # Debug: Confirm window closure
                    