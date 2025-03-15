#This function tracks the ball
def TrackRingPos(results, frame, threshold, previous_detections, first_detect, tracker):
    RIM_POS = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold and class_id==2:
            RIM_POS = (int(x1), int(y1), int(x2-x1), int(y2-y1))
            previous_detections.append(RIM_POS)
            tracker.init(frame, RIM_POS)
            first_detect = 1
    
    return frame, RIM_POS, previous_detections,first_detect, tracker