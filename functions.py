import math
import cv2
import random
import numpy as np

def Center_of_Bounded_Box(bounded_box):
    return int(bounded_box[0]+(bounded_box[2]/2)), int(bounded_box[1]+(bounded_box[3]/2))

def Distance(point1 , point2):
    x_dist = point1[0] - point2[0]
    y_dist = point1[1] - point2[1]
    dist = math.sqrt(x_dist**2 + y_dist**2)
    return dist

def createTracker():
    tracker = cv2.TrackerCSRT.create()
    return tracker

def calculate_overlap_area(box1, box2):
    """
    Calculate the overlapping area of two bounded boxes.

    Parameters:
    - box1: Tuple (x, y, width, height) representing the coordinates of the first box.
    - box2: Tuple (x, y, width, height) representing the coordinates of the second box.

    Returns:
    - Overlapping area as a float.
    """
    x_overlap = max(0, min(box1[0] + box1[2], box2[0] + box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1]))
    overlap_area = x_overlap * y_overlap
    return overlap_area

def check_overlap_percentage(box1, box2, threshold=0.7):
    """
    Check if the overlapping area of two bounded boxes is more than a specified threshold.

    Parameters:
    - box1: Tuple (x, y, width, height) representing the coordinates of the first box.
    - box2: Tuple (x, y, width, height) representing the coordinates of the second box.
    - threshold: Threshold percentage for overlap (default is 0.7).

    Returns:
    - True if overlapping area is more than the threshold, else False.
    """
    area_box1 = box1[2] * box1[3]  # width * height
    area_box2 = box2[2] * box2[3]  # width * height

    total_area = area_box1 + area_box2
    overlap_area = calculate_overlap_area(box1, box2)

    overlap_percentage = overlap_area / total_area

    return overlap_percentage

# Example usage:
box1 = (1, 1, 4, 4)  # (x, y, width, height)
box2 = (3, 3, 4, 4)

result = check_overlap_percentage(box1, box2)

def do_intersect(p1, p2, q1, q2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        elif val > 0:
            return 1  # clockwise
        else:
            return 2  # counterclockwise

    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    # p1, p2, q1 are collinear and q1 lies on segment p1p2
    if o1 == 0 and on_segment(p1, q1, p2):
        return True

    # p1, p2, q2 are collinear and q2 lies on segment p1p2
    if o2 == 0 and on_segment(p1, q2, p2):
        return True

    # q1, q2, p1 are collinear and p1 lies on segment q1q2
    if o3 == 0 and on_segment(q1, p1, q2):
        return True

    # q1, q2, p2 are collinear and p2 lies on segment q1q2
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False

def accumulate_heatmaps(frames_list, court_size=(500, 300), bin_size=20):
    # Initialize a dictionary to store accumulated heatmaps for each team
    accumulated_heatmaps = {}

    for frame_data in frames_list:
        for team, players in frame_data.items():
            if team not in accumulated_heatmaps:
                accumulated_heatmaps[team] = np.zeros((court_size[1] // bin_size, court_size[0] // bin_size))
            
            for player in players:
                x, y = player['court_position']
                x_bin = int(x // bin_size)
                y_bin = int(y // bin_size)
                
                if 0 <= x_bin < accumulated_heatmaps[team].shape[1] and 0 <= y_bin < accumulated_heatmaps[team].shape[0]:
                    accumulated_heatmaps[team][y_bin, x_bin] += 1
    
    # Apply Gaussian blur for better visualization
    for team in accumulated_heatmaps:
        accumulated_heatmaps[team] = cv2.GaussianBlur(accumulated_heatmaps[team], (5, 5), 0)
    
    return accumulated_heatmaps

def overlay_heatmap_on_image(heatmap, court_image, alpha=0.6, colormap=cv2.COLORMAP_JET):
    heatmap_resized = cv2.resize(heatmap, (court_image.shape[1], court_image.shape[0]))
    heatmap_colored = cv2.applyColorMap((255*heatmap_resized/np.max(heatmap_resized)).astype(np.uint8), colormap)
    
    overlay = cv2.addWeighted(court_image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

def draw_combined_heatmaps(frames_list, court_image_path, output_image_path, court_size=(500, 300), bin_size=20):
    court_image = cv2.imread(court_image_path)
    accumulated_heatmaps = accumulate_heatmaps(frames_list, court_size=court_image.shape[:2], bin_size=bin_size)
    
    for team, heatmap in accumulated_heatmaps.items():
        overlay_image = overlay_heatmap_on_image(heatmap, court_image)
        cv2.putText(overlay_image, f"Heatmap for {team}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(f"{output_image_path}_{team}.png", overlay_image)
        cv2.imshow(f"Heatmap for {team}", overlay_image)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def Generate_points(rim_pos):
    #Initialize points lists
    (x,y,w,h) = rim_pos
    top_pos = [int(x-(w*0.3)-35), int(y-(h*0.4)-35),int(w+(w*0.6)+70), int((h*0.4)+70)]
    bottom_pos = []
    
    return top_pos, bottom_pos


def find_quadratic_equation(points):
    n = len(points)

    center_points = []

    for i in points:
        center_points.append(Center_of_Bounded_Box(i))

    if n < 3:
        return (1,0,0)
        raise ValueError("At least three points are required to determine a unique quadratic equation.")

    x_values = [point[0] for point in center_points]
    y_values = [point[1] for point in center_points]

    # Create the Vandermonde matrix
    X = np.column_stack((np.square(x_values), x_values, np.ones(n)))
    Y = np.array(y_values)

    # Use NumPy's least squares function to find coefficients
    coefficients, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    # Coefficients a, b, and c
    a, b, c = coefficients

    # Return the quadratic equation as a tuple (a, b, c)
    equation = (a, b, c)
    return equation



def solve_quadratic(a, b, c, y):
    discriminant = b**2 - 4*a*(c - y)
    
    if discriminant < 0:
        return None  # No real roots
    
    sqrt_discriminant = math.sqrt(discriminant)
    
    x1 = (-b + sqrt_discriminant) / (2 * a)
    
    return x1

def find_launch_angle_from_quadratic(a, b, c, y):
    # Ensure the coefficient of x^2 is not zero to avoid division by zero
    if a == 0:
        raise ValueError("The coefficient of x^2 must be non-zero.")
    
    x1 = solve_quadratic(a, b, c, y)

    if x1 != None:
        # Calculate the launch angle in radians
        launch_angle1 = math.atan(2*a*x1+b)


        # Convert the angle to degrees
        launch_angle_degrees1 = math.degrees(launch_angle1)


        return launch_angle_degrees1
    else: 
        return 0

list1 = [(468, 340, 34, 38), (468, 340, 34, 38), (477, 330, 34, 38), (477, 330, 34, 38), (486, 320, 34, 38), (486, 320, 34, 38), (496, 311, 34, 38), (496, 311, 34, 38), (505, 302, 34, 38), (505, 302, 34, 38), (514, 293, 34, 38), (514, 293, 34, 38), (523, 286, 34, 38), (523, 286, 34, 38), (531, 279, 34, 38), (531, 279, 34, 38), (541, 272, 34, 38), (541, 272, 34, 38), (548, 265, 34, 38), (548, 265, 34, 38), (557, 259, 34, 38), (557, 259, 34, 38), (567, 255, 34, 38), (567, 255, 34, 38), (576, 251, 34, 38), (576, 251, 34, 38), (585, 247, 34, 38), (585, 247, 34, 38), (594, 244, 34, 38), (594, 244, 34, 38), (603, 242, 34, 38), (603, 242, 34, 38), (611, 239, 34, 38), (611, 239, 34, 38), (620, 238, 34, 38), (620, 238, 34, 38), (630, 238, 34, 38), (630, 238, 34, 38), (639, 238, 34, 38), (639, 238, 34, 38), (648, 238, 34, 38), (648, 238, 34, 38), (658, 238, 34, 38), (658, 238, 34, 38), (667, 239, 34, 38), (667, 239, 34, 38), (676, 242, 34, 38), (676, 242, 34, 38), (685, 245, 34, 38), (685, 245, 34, 38), (695, 249, 34, 38), (695, 249, 34, 38), (705, 254, 34, 38), (705, 254, 34, 38), (715, 258, 34, 38), (715, 258, 34, 38), (725, 264, 34, 38), (725, 264, 34, 38), (734, 271, 34, 38), (734, 271, 34, 38), (744, 277, 34, 38), (744, 277, 34, 38), (754, 285, 34, 38), (754, 285, 34, 38), (764, 292, 34, 38), (764, 292, 34, 38), (774, 301, 34, 38), (774, 301, 34, 38)]

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def getRange(player_pos):
    rim_pos = (250, 54)

    scaling_factor = 15/500

    dist = calculate_distance(rim_pos[0], rim_pos[1], player_pos[0], player_pos[1])

    return scaling_factor *dist


def getAngle(range,time):
    # Input: Range and Time of flight of projectile
    # Output: Angle of release in degrees
    temp = (time**2)/range
    radians = math.tanh(temp*9.8/2)
    degrees = radians * 180 / math.pi
    return degrees

'''a, b, c = find_quadratic_equation(list1)
print(a,b,c)
a,b = find_launch_angle_from_quadratic(a,b,c,list1[0][1])
print(a,b)'''

def getZone(x,y):
    ring_x = 250
    ring_y = 54

    dist = ((ring_x-x)**2 + (ring_y-y)**2)**0.5

    if dist > 234:
        if y>140:
            #3pt left
            if x <= 170:
                return 9
            #3pt centre
            elif x > 170 and x<=329:
                return 10
            #3pt right
            elif x > 329:
                return 11
    
    #3pt left corner
    if x <= 33 and y <= 140:
        return 1
    #left corner mid-range
    elif x <= 170 and y <= 85:
        return 2
    #left under the basket
    elif x <= 250 and y <= 85:
        return 3
    #right under the basket
    elif x <= 329 and y <= 85:
        return 4
    #right corner mid-range
    elif x <= 467 and y <= 85:
        return 5
    #right corner 3-pt
    elif x >= 467 and y <= 140:
        return 6
    #centre left mid range
    elif x <= 250 and y <= 190 and x >= 170:
        return 7
    #centre right mid range
    elif x <= 329 and y <= 190 and x >= 250:
        return 8
    # left mid range
    if x >= 33 and x <= 170 and y >= 85:
        return 12
    # centre mid range
    if x >= 170 and x <= 329 and y >= 190:
        return 13
    #right mid range
    if x >= 329 and x <= 467 and y>= 85:
        return 14
    