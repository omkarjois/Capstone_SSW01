#imports
import random
import math
from ultralytics import YOLO
import cv2
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import time as time


#file imports
from functions import check_overlap_percentage, createTracker, Generate_points, Center_of_Bounded_Box, Distance, find_launch_angle_from_quadratic, find_quadratic_equation, getAngle, getRange, getZone, do_intersect, draw_combined_heatmaps
from human_track import Track_Humans
from ball_track import TrackBallPos
from rim_track import TrackRingPos
from homography import group_by_team_and_position, new_homography, get_position_on_court

# Video path
video_path_out = 'out.mp4'.format('')

# Open CV preprocessing
cap = cv2.VideoCapture('/Users/omkarjois/Documents/PocketCoach/Playi/test_videos/d1_1080.mp4')
ret, frame = cap.read()

# Writing the file out
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Model
model_path = '/Users/omkarjois/Documents/PocketCoach/Playi/Capstone_SSW01/models/Akshay.pt'
model = YOLO(model_path)

#Main game function
def Shooting_game(ret, frame):
    #Variables
    #ball tracker
    tracker_ball = cv2.TrackerCSRT.create()
    #rim tracker
    tracker_rim = cv2.TrackerCSRT.create()
    #player trackers
    tracker_list = []
    players_with_data = {}
    complete_players_with_data = []

    #Rim variables
    threshold_rim = 0.4
    first_detect_rim = 0
    rim_pos = 0
    previous_detections_rim = []

    #Human variables
    init = 0
    prev_detections = []
    threshold_human = 0.2

    #Ball variables
    threshold_ball = 0.37
    first_detect_ball = 0
    previous_detections_ball = []
    
    ball_in = 0
    ball_with_player = 0
    ball_path = []
    player_release = 0
    player_positions = []

    #scoring params
    basket_count = 0
    basket_attempt = 0
    scores = []

    list_of_arcs = []
    release_time_list = []
    number_of_frames = 0
    was_with_player = 0

    #Human position
    shot_position_list = []
    player_pos_shot = (0,0,0,0)
    court_pos = (0,0)
    final_pos_list = []

    team1 = 0
    team2 = 0
    team = ""
    amount = 0

    #time for shot
    shot_frames = 0
    shot_times = []
    player = 0

    frames = 0
    frame_timer = 1
    zone = 0
    previous_frame = []
    tracked_distance = []
    top_pos = []
    bottom_pos = []
    rim_frame = []
    exited = 0

    player_team_history = [[], [], [], []]
    player_fixed_team_index = [0,0,0,0]

    overlap_end = 0
    overlap_start = 0
    ball_position = 0
    team_frame_count = 0

    # Initialize the tracker for human
    players = 2

    #Main game loop
    while ret:
        ball_with_player = 0
        # Detections
        results = model(frame)[0]

        if (frames % 20 == 0):
            old_rim = rim_pos
            frame, rim_pos, previous_detections_rim, first_detect_rim, tracker_rim = TrackRingPos(results, frame, threshold_rim, previous_detections_rim, first_detect_rim, tracker_rim)
            if rim_pos == 0:
                rim_pos = old_rim

        #Run game logic
        top_pos , bottom_pos = Generate_points(rim_pos)

        # Human Detections
        prev_detections, tracker_list, player_positions, tracked_distance, previous_frame, scores = Track_Humans(threshold_human, results, prev_detections, tracker_list, frame, previous_frame, tracked_distance, scores, players, player_fixed_team_index)

        if player_fixed_team_index == [0,0,0,0]:
            players_with_data, player_fixed_team_index, player_team_history, team_frame_count = group_by_team_and_position(player_positions, frame, new_homography(), (0,255,0), (0,0,255), player_fixed_team_index, team_frame_count, player_team_history)
            complete_players_with_data.append(players_with_data)
            print(player_team_history)
            print(player_fixed_team_index)
        else:
            print(player_positions)
            team_groups = {'Team A': [], 'Team B': []}
            for index, player in enumerate(player_positions):
                if player != 0:
                    if player_fixed_team_index[index] == 1:
                        x,y,w,h = player
                        foot_x = x + w // 2
                        foot_y = y + h
                        team_groups['Team A'].append({
                            'player_id': index,
                            'bounding_box': (x, y, w, h),
                            'court_position': get_position_on_court(foot_x, foot_y, new_homography())
                        })
                    else:
                        x,y,w,h = player
                        foot_x = x + w // 2
                        foot_y = y + h
                        team_groups['Team B'].append({
                            'player_id': index,
                            'bounding_box': (x, y, w, h),
                            'court_position': get_position_on_court(foot_x, foot_y, new_homography())
                        })

            players_with_data = team_groups
            complete_players_with_data.append(players_with_data)

        # Ball track 
        first_detect_ball, frame, previous_detections_ball, tracker_ball, ball_position = TrackBallPos(results, first_detect_ball, frame, previous_detections_ball, tracker_ball, threshold_ball)

        if ball_position != 0 and ball_position[2]*ball_position[3] > 10000:
            print("Big ass ball!!!!!!!!")
            ball_position = 0

        if ball_position != 0:
            if ball_in == 0 and check_overlap_percentage(ball_position, rim_pos) > 0: 
                frame_timer = 10

            # Check for basket
            if ball_in == 1:
                if check_overlap_percentage(ball_position, top_pos) > 0 and (ball_position[1]+ball_position[3]) < rim_pos[1]:
                    x,y,w,h = ball_position
                    overlap_start = [int(x+(w/2)), int(y+(h/2))]

                if exited == 0 and ball_position[1]>rim_pos[1]+rim_pos[3]:
                    x, y, w, h = ball_position
                    x1, y1, w1, h1 = rim_pos
                    overlap_end = [x + (w // 2), y + (h // 2)]
                    exited = 1
                    ball_in = 0
                    
                    print(overlap_start, overlap_end, (x1+(w1*0.1), y1), (x1+(w1*0.9), y1))
                    if do_intersect(overlap_start, overlap_end, (x1+(w1*0.1), y1), (x1+(w1*0.9), y1)):
                        frame_timer = 9
                        basket_count += 1
                        if getZone(court_pos[0], court_pos[1]) in [1,6,9,10,11]:
                            amount = 3
                        else:
                            amount = 2

                        if team == "Team A":
                            team1 += amount
                        else:
                            team2 += amount

                        scores[player_release]['makeList'][-1] = 1
                    
            #if ball is with player
            for i in player_positions:
                if i != 0:
                    if check_overlap_percentage(ball_position, i) > 0:
                        if was_with_player == 1:
                            was_with_player = 0
                            number_of_frames = 1
                        else:
                            number_of_frames += 1
                        ball_with_player = 1
                        player_release = player_positions.index(i)
                        ball_in = 0
                        ball_path = []

            # if ball in the air
            if ball_with_player == 0:
                was_with_player = 1

                if len(ball_path) == 0 and len(player_positions) != 0:
                    shot_frames = 1
                    player_pos_shot = player_positions[player_release]
                    if player_pos_shot != 0:
                        x, y, w, h = player_pos_shot
                        court_pos = get_position_on_court(x+w//2, y+h , new_homography())

                        for i in players_with_data['Team A']:
                            if i['player_id'] == player_release:
                                team = "Team A"
                        for i in players_with_data['Team B']:
                            if i['player_id'] == player_release:
                                team = "Team B"    

                shot_frames += 1
                ball_path.append(ball_position)

            # if ball reaches rings
            if ball_in == 0 and ball_with_player == 0 and frame_timer < 0:
                if check_overlap_percentage(ball_position, top_pos):
                    x,y,w,h = ball_position

                    overlap_start = [int(x+(w/2)), int(y+(h/2))]
                    overlap_end = overlap_start
                    
                    ball_in = 1
                    exited = 0
                    basket_attempt += 1
                    list_of_arcs.append(ball_path)
                    ball_path = []

                    #update release time
                    release_time_list.append(number_of_frames)
                    number_of_frames = 0
                    was_with_player = 0

                    frame_timer = 5

                    #update the position of release
                    shot_position_list.append(player_pos_shot)
                    final_pos_list.append(court_pos)

                    scores[player_release]['makeList'].append(0)
                    scores[player_release]['posList'].append(court_pos)
                    scores[player_release]['zoneList'].append(getZone(court_pos[0], court_pos[1]))

        cv2.putText(frame, str(team1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 150, 50), 4)
        cv2.putText(frame, "TEAM", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 50), 2)
        cv2.putText(frame, "BLUE", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 50), 2)

        cv2.putText(frame, str(team2), (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.putText(frame, "TEAM", (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "GREEN", (180, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.putText(frame, str(player_release), (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255, 255), 4)
        # cv2.putText(frame, "Player Release", (140, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        overlay = frame.copy()

        if ball_position!= 0:
            x,y,w,h = ball_position
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0), 4)
            cv2.rectangle(frame, (int(top_pos[0]), int(top_pos[1])), (int(top_pos[0]+top_pos[2]), int(top_pos[1]+top_pos[3])), (255, 255, 0), 4)
            cv2.rectangle(frame, (int(rim_pos[0]), int(rim_pos[1])), (int(rim_pos[0]+rim_pos[2]), int(rim_pos[1]+rim_pos[3])), (255, 255, 0), 4)
            cv2.rectangle(overlay, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0), 4)
            cv2.rectangle(overlay, (int(top_pos[0]), int(top_pos[1])), (int(top_pos[0]+top_pos[2]), int(top_pos[1]+top_pos[3])), (255, 255, 0), 4)
            cv2.rectangle(overlay, (int(rim_pos[0]), int(rim_pos[1])), (int(rim_pos[0]+rim_pos[2]), int(rim_pos[1]+rim_pos[3])), (255, 255, 0), 4)

        if overlap_end!= 0 and overlap_start!=0:
            cv2.line(frame, overlap_start, overlap_end, (255,0,0), 2)
            cv2.line(overlay, overlap_start, overlap_end, (255,0,0), 2)
            x1, y1, w1, h1 = rim_pos
            cv2.line(frame,(int(x1+(w1*0.1)), int(y1)), (int(x1+(w1*0.9)), int(y1)) , (255,0,0), 2)
            cv2.line(overlay,(int(x1+(w1*0.1)), int(y1)), (int(x1+(w1*0.9)), int(y1)) , (255,0,0), 2)


        # Get the dimensions of the image to be overlaid
        image = cv2.imread("/Users/omkarjois/Documents/PocketCoach/Playi/Capstone_SSW01/court_black.jpg")
        image = cv2.resize(image, (int(166*3/2), int(157*3/2)))
        image_h, image_w, _ = image.shape


        # Define the top-left corner coordinates of the ROI in the frame
        top, left = 0, W-image_w  # You can adjust these values

        # Check if the image fits within the frame boundaries
        if top + image_h <= frame.shape[0] and left + image_w <= frame.shape[1]:
            # print("Draw")
            # Copy the image data into the ROI of the frame
            # frame[top:top+image_h, left:left+image_w] = image
            overlay[top:top+image_h, left:left+image_w] = image

        i = 0
        # print(players_with_data)
        for i in range(len(players_with_data['Team A'])):
            _player = players_with_data['Team A'][i]

            x,y,w,h = _player['bounding_box']
            cv2.putText(frame, str(_player['player_id']), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x+3), int(y+3)), (int(x+w), int(y+h)), (0, 255, 0), 4)
            cv2.putText(overlay, str(_player['player_id']), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(overlay, (int(x+3), int(y+3)), (int(x+w), int(y+h)), (0, 255, 0), 4)

            x,y = _player['court_position']
            x = x/2
            y = y/2
            # print(i)
            x = x+W-image_w
            cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)
            cv2.circle(overlay, (int(x), int(y)), 4, (0,255,0), -1)
        
        i = 0
        for i in range(len(players_with_data['Team B'])):
            _player = players_with_data['Team B'][i]

            x,y,w,h = _player['bounding_box']
            cv2.putText(frame, str(_player['player_id']), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 50), 2)
            cv2.rectangle(frame, (int(x+3), int(y+3)), (int(x+w), int(y+h)), (255, 150, 50), 4)
            cv2.putText(overlay, str(_player['player_id']), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 50), 2)
            cv2.rectangle(overlay, (int(x+3), int(y+3)), (int(x+w), int(y+h)), (255, 150, 50), 4)

            x,y = _player['court_position']
            x = x/2
            y = y/2
            # print(i)
            x = x+W-image_w
            cv2.circle(frame, (int(x), int(y)), 4, (255, 150, 50), -1)
            cv2.circle(overlay, (int(x), int(y)), 4, (255, 150, 50), -1)


        # for i in list_of_arcs:
        #     points = np.array(i)
        #     unique_x, averaged_y = np.unique(points[:, 0], return_inverse=True)
        #     averaged_y = np.bincount(averaged_y, weights=points[:, 1]) / np.bincount(averaged_y)

        #     # Create a spline interpolation
        #     spl = make_interp_spline(unique_x, averaged_y, k=3)

        #     # Generate a smoother curve
        #     x_smooth = np.linspace(unique_x.min(), unique_x.max(), 200)
        #     y_smooth = spl(x_smooth)

        #     for j in range(len(x_smooth)-1):
        #         cv2.line(frame, (int(x_smooth[j]), int(y_smooth[j])), (int(x_smooth[j+1]), int(y_smooth[j+1])), (0, 0, 255 ), 2)


            

        #wait for 150 frames after the rim is found
        if rim_pos != 0:
            frame_timer -= 1
            frames += 1
        

        image_new = cv2.addWeighted(frame, 0.4, overlay, 1 - 0.4, 0)


        cv2.imshow("image", image_new)
        cv2.waitKey(1)

        out.write(image_new)
        ret, frame = cap.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    print(scores)

    # import tkinter as tk
    # from PIL import Image, ImageTk, ImageDraw
    # import numpy as np

    # # Main function to create each player's shot analysis window
    # def create_player_window(root, player_data, player_num):
    #     # Load the court image
    #     court_img = Image.open("/Users/omkarjois/Documents/PocketCoach/Playi/Capstone_SSW01/court_black.jpg")
    #     draw = ImageDraw.Draw(court_img)

    #     make_list = player_data["makeList"]
    #     pos_list = player_data["posList"]
    #     zone_list = player_data["zoneList"]
    #     angle_list = player_data["angleList"]

    #     # Determine the player's team based on the index (0,1 -> Team 1, 2,3 -> Team 2)
    #     team = "Team 1" if player_num in [1, 2] else "Team 2"

    #     # Initialize counters for calculating 2-point and 3-point accuracy
    #     two_point_attempts = 0
    #     two_point_makes = 0
    #     three_point_attempts = 0
    #     three_point_makes = 0

    #     # Draw shots on court
    #     for made, pos, zone in zip(make_list, pos_list, zone_list):
    #         x, y = pos
    #         if made:
    #             draw.ellipse((x-5, y-5, x+5, y+5), fill="green")  # Made shot (green dot)
    #             if zone in [1, 6, 9, 10, 11]:  # 3-point zone
    #                 three_point_attempts += 1
    #                 three_point_makes += 1
    #             else:  # 2-point zone
    #                 two_point_attempts += 1
    #                 two_point_makes += 1
    #         else:
    #             draw.line((x-5, y-5, x+5, y+5), fill="red", width=3)  # Missed shot (red X)
    #             draw.line((x-5, y+5, x+5, y-5), fill="red", width=3)
    #             if zone in [1, 6, 9, 10, 11]:  # 3-point zone
    #                 three_point_attempts += 1
    #             else:  # 2-point zone
    #                 two_point_attempts += 1

    #     # Calculate accuracy and average angle for the player
    #     two_point_accuracy = (two_point_makes / two_point_attempts * 100) if two_point_attempts > 0 else 0
    #     three_point_accuracy = (three_point_makes / three_point_attempts * 100) if three_point_attempts > 0 else 0
    #     average_angle = np.mean(angle_list) if angle_list else 0

    #     # Create a new Toplevel window for each player
    #     player_window = tk.Toplevel(root)
    #     player_window.title(f"Player {player_num} - Shot Analysis")

    #     # Convert Image to Tkinter-compatible format and display
    #     court_img_tk = ImageTk.PhotoImage(court_img)
    #     canvas = tk.Canvas(player_window, width=court_img.width, height=court_img.height + 120, bg="white")
    #     canvas.pack()
    #     canvas.create_image(0, 0, anchor="nw", image=court_img_tk)

    #     # Display player stats below the image
    #     text = f"Team: {team}\nPlayer {player_num} Statistics:\n\n" \
    #         f"2-Point Accuracy: {two_point_accuracy:.2f}%\n" \
    #         f"3-Point Accuracy: {three_point_accuracy:.2f}%\n" \
    #         f"Average Angle: {average_angle:.2f}°"
    #     canvas.create_text(court_img.width / 2, court_img.height + 50, text=text, font=("Arial", 12, "bold"), fill="black")

    #     # Keep a reference to prevent garbage collection
    #     player_window.image = court_img_tk

    # # Main function to display all players' windows
    # def show_all_players(scores):
    #     # Create the main root window that will manage all Toplevel windows
    #     root = tk.Tk()
    #     root.withdraw()  # Hide the main window, as we're only using Toplevel windows

    #     for i, player_data in enumerate(scores):
    #         create_player_window(root, player_data, player_num=i+1)

    #     # Run the Tkinter main

    #     root.mainloop()

    # # Run the function to display all players' stats
    # show_all_players(scores)

    # court_image_path = '/Users/omkarjois/Documents/PocketCoach/Playi/Capstone_SSW01/court_black.jpg'  # Path to your court image
    # output_image_path = '/Users/omkarjois/Documents/PocketCoach/Playi/Capstone_SSW01/'  # Base path for saving heatmap images
    # draw_combined_heatmaps(complete_players_with_data, court_image_path, output_image_path)

    # # for i in range(len(final_pos_list)):
    # #     #cv2.putText(img, str(i), final_pos_list[i], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    # #     #cv2.circle(img,final_pos_list[i],5,[255,0,0],-1)
    # #     range1 = getRange(final_pos_list[i])
    # #     zone = getZone(final_pos_list[i][0], final_pos_list[i][1])
    # #     if zone in (14, 11, 10, 13, 8):
    # #         time_ = shot_times[i]+8
    # #         time_ = time_/30
    # #     else:
    # #         time_ = (shot_times[i]+3)/30
    # #     angle = getAngle(range1, time_)
    # #     print("Shot "+str(i+1)+": "+ str(zone), str(angle))
    
    
    # cv2.destroyAllWindows()
    # #cv2.imshow('Image', img)
    # #cv2.waitKey() 

Shooting_game(ret, frame)

cap.release()
out.release()

