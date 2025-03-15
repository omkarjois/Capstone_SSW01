import cv2
import numpy as np
from functions import Center_of_Bounded_Box, Distance, createTracker
from sklearn.cluster import KMeans

from homography import classify_team, extract_dominant_color

def Track_Humans(threshold, results, prev_detections, tracker_list, frame, previous_frame, tracked_distance, scores, players, player_fixed_team_index):
    # detected
    # not detected and tracked
    # not detected and un tracked
    # Distance near
    # in the frame Or outside

    player_positions = [0 for i in range(len(tracker_list))]
    detected = [0 for i in range(len(tracker_list))]
    i = 0

    if(len(previous_frame)<5):
        previous_frame.append(frame)
    else:
        previous_frame.pop(0)
        previous_frame.append(frame)

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        # If it is a human
        if score > threshold and class_id==1:
            distances = []
            color_distance = []

            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            roi_vector = read_and_preprocess_image(roi)

            for idx, prev in enumerate(prev_detections):
                # Predict next position for each previous detection
                predicted_pos = predict_position(prev)
                x,y,w,h = predicted_pos
                predicted_box = Center_of_Bounded_Box((int(x), int(y), int(w), int(h)))
                current_box = Center_of_Bounded_Box((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

                # Calculate Euclidean distance
                euclidean_dist = Distance(predicted_box, current_box)

                # Calculate cosine similarity with recent frames
                avg_cosine_sim = calculate_average_similarity(roi_vector, prev, previous_frame)

                roi = frame[y+(h//4) : y+(h//2), int(x+ w*0.3) : int(x+ w*0.7)]
                dominant_color = extract_dominant_color(roi)
                dominant_color = tuple(dominant_color)

                expected_team = player_fixed_team_index[idx]
                current_team_color = classify_team(dominant_color, (0,255,0), (0,0,255))  # Use your classify function

                # Penalize distance if color doesn't match expected team
                if (expected_team == 1 and current_team_color != 'Team A') or (expected_team == 2 and current_team_color != 'Team B'):
                    euclidean_dist += 200

                combined_distance = euclidean_dist + (1 - avg_cosine_sim)


                distances.append(combined_distance)

            print(distances)
            if len(distances) != 0:
                min_value = min(distances)
                # print("min Value: "+str(min_value))
                # detection success and player exists in the frame
                if min_value < 50:
                    ind = distances.index(min_value)
                    detected[ind] = 1

                    # update player positions
                    tracked_distance[ind] = 0
                    prev_detections[ind].append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
                    player_positions[ind] = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                    tracker_list[ind].init(frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
                
                # New player entering the court
                else:
                    # Create vector of current detection
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    roi_vector = read_and_preprocess_image(roi)
                    max_sim = [0,0]
                    for i in range(len(detected)):
                        if detected[i] == 0 and len(previous_frame)==5:
                            j = -1
                            sum_sim = 0
                            while j > -6:
                                x,y,w,h = prev_detections[i][j]
                                roi_old = previous_frame[j][y:y+h, x:x+w]
                                roi_old_vector = read_and_preprocess_image(roi_old)
                                similarity = cosine_similarity(roi_vector, roi_old_vector)
                                sum_sim  += similarity
                                j -= 1

                            similarity = sum_sim/5
                            if similarity > max_sim[0]:
                                max_sim[0] = similarity
                                max_sim[1] = i
                    # print(max_sim)
                    if max_sim[0] > 0.6:
                        ind = max_sim[1]
                        tracked_distance[ind] = 0
                        prev_detections[ind].append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
                        player_positions[ind] = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                        tracker_list[ind].init(frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
                    
                    elif len(prev_detections) < (players*2):
                        scores.append({"makeList":[], "posList":[], "zoneList":[], "angleList": []})
                        tracked_distance.append(0)
                        player_positions.append(((int(x1), int(y1), int(x2-x1), int(y2-y1))))
                        tracker_list.append(createTracker())
                        prev_detections.append([(int(x1), int(y1), int(x2-x1), int(y2-y1))])
                        tracker_list[len(tracker_list)-1].init(frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
            
            if len(distances) == 0 and len(prev_detections) < (players*2):
                scores.append({"makeList":[], "posList":[], "zoneList":[], "angleList": []})
                tracked_distance.append(0)
                player_positions.append(((int(x1), int(y1), int(x2-x1), int(y2-y1))))
                tracker_list.append(createTracker())
                prev_detections.append([(int(x1), int(y1), int(x2-x1), int(y2-y1))])
                tracker_list[len(tracker_list)-1].init(frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
        
            

    
    # tracking
    for i in range(len(detected)):
        if detected[i] == 0 and tracked_distance[i] < 5:
            tracked_distance[i] += 1
            success, box = tracker_list[i].update(frame)
            if (success):
                print("Tracking")
                (x,y,w,h) = [int(a) for a in box]
                #changing previous prediction
                last_known_position = (int(x), int(y), int(w), int(h))
                prev_detections[i].append(last_known_position)
                player_positions[i] = last_known_position
                #draw
                #cv2.putText(frame, str(i), (int(x), int(y - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA) 
            else:
                print("Failed")

    return prev_detections, tracker_list, player_positions, tracked_distance, previous_frame, scores

def read_and_preprocess_image(image, size=(224, 224)):
    image = cv2.resize(image, size)
    # Flatten the image to a 1D vector
    image = image.flatten()
    # Normalize the vector
    image = image / np.linalg.norm(image)
    return image


def cosine_similarity(vec1, vec2):
    # Compute the dot product between the two vectors
    dot_product = np.dot(vec1, vec2)
    # Compute the norms of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # Compute the cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def predict_position(detections, weight=0.):
    """
    Predict the next position based on the last 5 detections using weighted momentum.
    """
    if len(detections) < 2:
        return detections[-1]

    velocity_x = 0
    velocity_y = 0
    num_positions = min(5, len(detections) - 1)

    for i in range(-num_positions, -1):
        x1, y1 = Center_of_Bounded_Box(detections[i])
        x2, y2 = Center_of_Bounded_Box(detections[i + 1])
        velocity_x += (x2 - x1)
        velocity_y += (y2 - y1)

    avg_velocity_x = (velocity_x / num_positions) * weight
    avg_velocity_y = (velocity_y / num_positions) * weight

    last_x, last_y= Center_of_Bounded_Box(detections[-1])
    predicted_x = last_x + avg_velocity_x
    predicted_y = last_y + avg_velocity_y
       
    return int(predicted_x - detections[-1][2]//2), int(predicted_y - detections[-1][3]//2), int(detections[-1][2]), int(detections[-1][3])

def calculate_average_similarity(roi_vector, prev, previous_frame):
    """Calculate the average cosine similarity of the ROI with previous frames for a specific player."""
    j = -1
    sum_cosine_sim = 0
    num_positions = min(5, len(prev))
    while j > -num_positions - 1:
        prev_x, prev_y, prev_w, prev_h = prev[j]
        roi_old = previous_frame[j][prev_y:prev_y + prev_h, prev_x:prev_x + prev_w]
        roi_old_vector = read_and_preprocess_image(roi_old)
        sum_cosine_sim += cosine_similarity(roi_vector, roi_old_vector)
        j -= 1

    avg_cosine_sim = sum_cosine_sim / num_positions
    return avg_cosine_sim