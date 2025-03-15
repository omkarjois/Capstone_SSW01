import cv2
from ultralytics import YOLO

#This function tracks the ball
def TrackBallPos(results, first_detect, frame, previous_detections, tracker, threshold):
    detect = 0
    BALL_POS = 0

    # If there are detections, update last known position and add it to previous_detections
    if results.boxes.data.tolist():
        for result in results.boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = result
            if class_id == 0 and score > threshold:
                last_known_position = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                previous_detections.append(last_known_position)

                tracker.init(frame, last_known_position)

                detect = 1
                first_detect = 1
                BALL_POS = last_known_position
                (x,y,w,h) = BALL_POS
                break
                # cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 4)
    
    #Tracking if no detection
    if detect == 0 and first_detect == 1:
        success, box = tracker.update(frame)
        if (success):
            (x,y,w,h) = [int(a) for a in box]
            #changing previous prediction
            last_known_position = (int(x), int(y), int(w), int(h))
            # cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 4)
            previous_detections.append(last_known_position)
            BALL_POS = last_known_position

        else:
            BALL_POS=0

    # Write the frame to the output video
    return first_detect, frame, previous_detections, tracker, BALL_POS