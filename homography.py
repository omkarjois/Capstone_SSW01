import cv2
import numpy as np
from collections import Counter


def extract_dominant_color(roi, k=2):
    # Convert the ROI to RGB color space
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixel_values = roi_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Apply K-means clustering to find two dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count the number of pixels assigned to each cluster
    _, counts = np.unique(labels, return_counts=True)
    
    # Identify the most dominant color (larger cluster)
    dominant_color = centers[np.argmax(counts)].astype(int)
    
    return dominant_color

def new_homography():
    
    src = [[649,613], [1030, 567], [948, 765], [1414, 671]]
    dest = [[169, 0], [328, 0], [169, 191], [328, 191]]
    src_pts = np.array(src).reshape(-1, 1, 2)
    dst_pts = np.array(dest).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def classify_team(color, team_a_color, team_b_color):
    # Calculate Euclidean distance to each team's color and classify based on closest match
    dist_to_a = np.linalg.norm(np.array(color) - np.array(team_a_color))
    dist_to_b = np.linalg.norm(np.array(color) - np.array(team_b_color))
    
    if dist_to_a < dist_to_b:
        return 'Team A'
    else:
        return 'Team B'

def get_position_on_court(x, y, H):
    src_pts = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = cv2.perspectiveTransform(src_pts, H)
    return dst_pts[0][0]  # Return as a tuple (x, y)

def group_by_team_and_position(detections, image, H, team_a_color, team_b_color, player_fixed_team_index, team_frame_count, player_team_history):
    team_groups = {'Team A': [], 'Team B': []}
    if len(detections) == 4:
        team_frame_count += 1
    
    for i in detections:
        if i != 0:
            x,y,w,h = i
            # Extract the ROI (torso region)
            roi = image[y+(h//6) : y+(h//2), int(x+w*0.2) : int(x+w*0.8)]
            
            # Find the dominant color in the ROI
            dominant_color = extract_dominant_color(roi)
            dominant_color = tuple(dominant_color)
            # print(dominant_color)
            
            # Classify into a team based on jersey color
            team = classify_team(dominant_color, team_a_color, team_b_color)
            if team_frame_count < 30 and len(detections) == 4:
                if team == 'Team A':
                    player_team_history[detections.index(i)].append(1)
                else:
                    player_team_history[detections.index(i)].append(2)
            elif team_frame_count >= 30:
                j = 0
                for j in range(4):
                    player_fixed_team_index[j] = Counter(player_team_history[j]).most_common(1)[0][0]
                
            # Calculate the bottom center point of the bounding box (foot position)
            foot_x = x + w // 2
            foot_y = y + h
            
            # Get the position on the court (top-down view)
            position_on_court = get_position_on_court(foot_x, foot_y, H)
            
            # Add the player to the appropriate team group with their court position
            team_groups[team].append({
                'player_id': detections.index(i),
                'bounding_box': (x, y, w, h),
                'court_position': position_on_court # Need sto be updated
            })
    
    return team_groups, player_fixed_team_index, player_team_history, team_frame_count