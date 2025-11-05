"""
main.py

Script principal del proyecto de detección de poses.

Descripción:
- Ejecuta la detección de personas usando YOLOv11.
- Triangula las posiciones de las articulaciones a partir de múltiples cámaras.
- Estima poses 3D usando el modelo st-gcn++.
- Genera los archivos de calibración de cámaras (camera_calibration/) para poder reutilizar la calibración.

Entradas:
- Señales de vídeo desde múltiples cámaras para procesar.
- Modelos preentrenados en 'PoseDetection/Models':
    - st-gcn++: archivo .pth y config_inference.py
    - Yolo11-Pose: archivo .pt
- Configuración del modelo (archivo config_inference.py).

Salidas:
- Archivos .pkl en 'camera_calibration/' que contienen:
    - Parámetros de cada cámara: `parameters_index_0.pkl`, `parameters_index_1.pkl`, ..., `parameters_index_N.pkl` (uno por cada cámara).
    - Matrices de proyección combinadas: `projection_camerasX.pkl` (donde X depende del número de cámaras).
"""

import os
import threading # For running multiple threads
import time 
from ultralytics import YOLO # Import the YOLO classsss
from multiprocessing import Manager, Process # For managing shared memory and parallel processes
import cv2 
import numpy as np 
import pickle  # For saving and loading calibration parameters
import open3d as o3d # For visualizationq
import copy
from typing import List, Tuple
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from pyskl.models import build_model
from pyskl.utils import cache_checkpoint
from pyskl.apis import init_recognizer, inference_recognizer
from scipy.spatial.transform import Rotation as R

# Load the YOLO model
model = YOLO("PoseDetection/Models/Yolo11-Pose/yolo11m-pose.pt")


CHESSBOARD_SIZE = (9, 6)
NUMBER_OF_CAMERAS = 3
VISUALIZATION = True # Set to True to enable 3D visualization


def get_camera_serial_number(camera_index):
    '''
    """
    Retrieve the serial number (or unique device ID) of a specific camera based on its index.

    Parameters:
        camera_index (int): The OpenCV camera index (0, 1, 2, ...)

    Returns:
        str: Serial number or unique identifier of the camera, or None if not found.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow to get device info
    if not cap.isOpened():
        print(f"Could not access camera at index {camera_index}")
        return None

    device_path = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Attempt to get a unique identifier (not always reliable)
    print(f"Device path: {device_path}")
    cap.release()

    if device_path is None:
        print(f"Could not retrieve device path for camera {camera_index}")
        return None

    c = wmi.WMI()
    for usb in c.Win32_PnPEntity():
        if usb.Name and "camera" in usb.Name.lower():  # Ensure usb.Name is not None
            if usb.PNPDeviceID and device_path in usb.PNPDeviceID:
                return usb.PNPDeviceID.split("\\")[-1]  # Extract unique ID

    print(f"No serial number found for camera at index {camera_index}")
    return None ''' #TODO

    return "index_" + str(camera_index)

def get_camera_calibration_parameters(camera_index=0, num_images=30, chessboard_size=CHESSBOARD_SIZE):
    """
    Get the camera matrix and distortion coefficients by either calibrating the camera or loading from a file.

    Parameters:
        camera_index (int): Index of the camera for live calibration.
        num_images (int): Number of calibration images to capture.

    Returns:
        camera_matrix (numpy.ndarray): The intrinsic camera matrix.
        dist_coeffs (numpy.ndarray): The distortion coefficients.
    """
    serial_number = get_camera_serial_number(camera_index)
    
    if serial_number is None:
        print(f"Could not determine camera serial number for camera {camera_index}. Using index instead.")
        serial_number = f"index_{camera_index}"

    param_file = f"camera_calibration\\parameters_{serial_number}.pkl"

    if param_file and os.path.exists(param_file): # Load calibration parameters from a file if available
        # Load calibration parameters from a file
        with open(param_file, 'rb') as file:
            params = pickle.load(file)
        print("Calibration parameters loaded from file.")
        return params['camera_matrix'], params['dist_coeffs']

    cap = cv2.VideoCapture(camera_index)

    # Prepare object points (3D coordinates of the chessboard corners in the world)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # List to store 3D object points and 2D image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    captured_images = 0

    while captured_images < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            return

        # Display the frame
        cv2.imshow("Camera Calibration - Press 'S' to Capture Image", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # If 's' key is pressed, capture the image
        if key == ord('s'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                print(f"Captured image {captured_images + 1}/{num_images}")

                # Refine corner locations
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

                # Store object points and image points
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw the corners on the image
                cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
                cv2.imshow("Captured Chessboard Corners", frame)

                captured_images += 1
            else:
                print("Chessboard corners not found. Try adjusting the chessboard or the camera angle.")

        # Exit when the 'q' key is pressed
        if key == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Perform calibration if enough images were captured
    if len(objpoints) >= num_images:
        print(f"Calibrating camera with {captured_images} images...")

        # Get the camera calibration parameters
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            print("Camera calibration successful.")

            # Save calibration parameters to file if a path is provided
            with open(param_file, 'wb') as file:
                pickle.dump({'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs}, file)
            print(f"Calibration parameters saved to {param_file}")

            return camera_matrix, dist_coeffs
        else:
            print("Camera calibration failed.")
            return None, None
    else:
        print("Not enough valid images captured for calibration.")
        return None, None

# Example calibration using the camera matrix and distortion coefficients
def undistort_keypoint(keypoint, camera_matrix, dist_coeffs):
    """
    Apply distortion correction to keypoints based on the camera matrix and distortion coefficients.
    """
    if keypoint == (0, 0): # Skip undistortion for invalid keypoints
        return (0, 0)

    keypoints = np.array([keypoint], dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(keypoints, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted[0][0]

def run_camera_inference(camera_index, camera_matrix, dist_coeffs, keypoint_dict):

    for results in model.predict(source=camera_index, show=True, verbose=False, stream=True):

        # Dictionary to store keypoints for the current frame
        keypoints_for_frame = {}

        # Process the results
        for result in results:
            if hasattr(result, "keypoints"):  # Check if keypoints are available
                keypoints_array = result.keypoints.data.cpu().numpy()  # Convert keypoints to NumPy array
                for person_id, person in enumerate(keypoints_array):  # Iterate over people detected
                    keypoints_for_frame[person_id] = []  # Initialize list for keypoints of the current person
                    for kp_idx, kp in enumerate(person):  # Iterate over keypoints with index
                        while len(keypoints_for_frame[person_id]) <= kp_idx:
                            keypoints_for_frame[person_id].append(None)  # Add placeholder `None` if list is too short
                        # Extract keypoint index, coordinates, and confidence
                        x = kp[0]  # x-coordinate
                        y = kp[1]  # y-coordinate
                        conf = kp[2]  # confidence

                        #print(f"Person ID: {person_id}, KeyPoint Index: {kp_idx}")
                        #print(f"Distorted KeyPoint: {(x, y)}, Confidence: {conf}")

                        # Undistort the keypoint
                        #x, y = undistort_keypoint((x, y), camera_matrix, dist_coeffs) Keep em distorted
                        #print(f"Undistorted KeyPoint: {(x, y)}")

                        # Store the undistorted keypoint in the dictionary
                        keypoints_for_frame[person_id][kp_idx] = (x, y, conf)
        
        # Update the shared dictionary with the keypoints for the current frame
        #keypoint_dict[camera_index] = keypoints_for_frame
        keypoint_dict[camera_index] = copy.deepcopy(keypoints_for_frame)
        debug = dict(keypoint_dict) 

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def match_people(keypoint_dict, projection_matrices):
    # Initialize an empty dictionary to store matched people
    matched_people = {}

    # Create a temporary dictionary to collect data
    temp_matched_people = {}
    debug = dict(keypoint_dict)

    # Iterate over the keypoints data
    for camera_index in dict(keypoint_dict):
        for person_id in dict(keypoint_dict)[camera_index]:
            if person_id == 0:
                if person_id not in temp_matched_people:
                    temp_matched_people[person_id] = {}
                for kp_index, kp in enumerate(keypoint_dict[camera_index][person_id]):
                    if kp_index not in temp_matched_people[person_id]:
                        temp_matched_people[person_id][kp_index] = {}
                    temp_matched_people[person_id][kp_index][camera_index] = kp

    # After iteration is complete, update matched_people
    matched_people.update(temp_matched_people)

    # For debugging: print the matched people
    #print(matched_people)
    #print()

    return matched_people

def get_camera_center(P2):
    """Extract the 3D camera center from a 3x4 projection matrix."""
    #print("P2:", P2)  # Debugging: Check the shape of P2
    P = np.array(P2)

    #print("Shape of P:", P.shape)  # Make sure it prints (3, 4)

    M = P[:, :3]  # Matriz [K * R]
    t = P[:, 3]   # Vector de traslación proyectado (K * t)
    return(-np.linalg.inv(M) @ t)

def get_ray_direction(P2, point_2d: Tuple[float, float]):
    """Get normalized ray direction in world coordinates from image point."""
    P = np.array(P2)
    A = P[:, :3]
    x = np.array([point_2d[0], point_2d[1], 1.0])
    ray = np.linalg.inv(A) @ x
    return ray / np.linalg.norm(ray)

def weighted_triangulation(keypoints, confidences, projection_matrices):
    """
    Triangulate a 3D point using confidence-weighted midpoint method.

    Parameters:
        keypoints: Nx2 array of (x, y) image coordinates (dict or list of tuples)
        confidences: N-length array of confidence values for each point
        projection_matrices: List of 3x4 projection matrices, same length as keypoints/confidences

    Returns:
        point_3d: 3D coordinates as a numpy array of shape (3,)
    """
    # Validate that keypoints and confidences have the same length
    if len(keypoints) != len(confidences):
        raise ValueError(f"Keypoints and confidences must have the same length: "
                         f"{len(keypoints)} vs {len(confidences)}")

    assert len(keypoints) == len(confidences), "Input lengths mismatch"

    A = np.zeros((3, 3))
    b = np.zeros(3)
    valid_views = 0

    for xy, conf, P in zip(keypoints.values(), confidences.values(), projection_matrices.values()):  # .values() here

        # Check if xy is a tuple of (x, y)
        if not isinstance(xy, tuple) or len(xy) != 2:
            raise ValueError(f"Each keypoint must be a tuple of two values: {xy}")
        
        if conf <= 0.4:
            continue  # skip low-confidence
        if xy[0] == 0 and xy[1] == 0:
            continue  # skip dummy points

        weight = conf ** 2

        #print(f"Keypoints: {xy}, Confidences: {conf}, Projection Matrices: {P}")
        C = get_camera_center(P) 
        d = get_ray_direction(P, xy).reshape(3, 1) 
        
        #print(f"Camera Center: {C}, Ray Direction: {d}")
        I = np.eye(3)
        W = weight * (I - d @ d.T)
        A += W
        b += W @ C
        valid_views += 1

    if valid_views < 2:
        return np.array([0.0, 0.0, 0.0])  # Not enough views to triangulate reliably

    point_3d = np.linalg.solve(A, b)
    # Flip Y-axis
    point_3d[0] = -point_3d[0] 
    point_3d[1] = -point_3d[1]
    return point_3d.flatten()


def update_points_lines(keypoints_3d):
    """
    Update the Open3D geometries (point cloud and line set) using the latest keypoints.
    Expects keypoints_3d to be a dictionary whose values are (x, y, z) coordinates.
    """
    global pcd, line_set, lines
    if keypoints_3d is None or len(keypoints_3d) == 0:
        return

    # Convert the dictionary values to an (n,3) numpy array.
    npkeypoints = np.array(list(keypoints_3d.values())).reshape(-1, 3)

    # Update the point cloud
    pcd.points = o3d.utility.Vector3dVector(npkeypoints)

    # Filter lines: only use a line if neither endpoint is the placeholder (0,0,0)
    valid_lines = []
    for line in lines:
        p1 = npkeypoints[line[0]]
        p2 = npkeypoints[line[1]]
        if not (np.all(p1 == 0) or np.all(p2 == 0)):
            valid_lines.append(line)
    if valid_lines:
        line_set.points = o3d.utility.Vector3dVector(npkeypoints)
        line_set.lines = o3d.utility.Vector2iVector(np.array(valid_lines))
    
def start_visualization(keypoints_3d):
    '''
    Starts a real-time 3D visualization loop using Open3D.
    
    Parameters:
        get_keypoints_func: Function that returns a (17, 3) NumPy array of keypoints.
                            Points that are [0, 0, 0] are ignored.
    '''
    global pcd, line_set, lines

    # Duration of each frame in seconds
    frame_time = 1 / 30.0  # 30 FPS

    # Define the connectivity for the skeleton (same as your VisPy lines)
    lines = np.array([
        [0, 1], [0, 2], [0, 5], [0, 6],
        [1, 2], [1, 3], [2, 4], [5, 6],
        [5, 7], [5, 11], [6, 12], [6, 8],
        [7, 9], [8, 10], [11, 12], [11, 13],
        [12, 14], [13, 15], [14, 16]
    ])

    keypoints = np.array([np.array([     2.4223,     -3.1009,      39.203]), 
        np.array([     3.2569,     -3.7365,      39.614]), 
        np.array([     1.8148,     -3.9389,      39.566]), 
        np.array([      4.488,     -3.3529,      41.946]), 
        np.array([    0.61278,     -4.1373,      41.883]), 
        np.array([     7.2838,     0.86162,      41.843]), 
        np.array([    -2.3216,     0.50413,      44.315]), 
        np.array([     8.6338,      4.9017,      37.397]), 
        np.array([    -6.1763,      4.1871,       40.12]), 
        np.array([      3.128,      4.0059,      34.757]), 
        np.array([    -2.4501,      3.8709,      35.829]), 
        np.array([      5.961,      11.143,      43.601]), 
        np.array([  -0.007574,      11.017,      44.764]), 
        np.array([     4.4909,      6.6938,      37.398]), 
        np.array([    -2.0696,      6.6634,      39.453]), 
        np.array([     2.5641,      13.798,      42.889]), 
        np.array([     2.1497,       12.48,      40.136])])

    # Initialize Open3D geometries
    pcd = o3d.geometry.PointCloud()
    line_set = o3d.geometry.LineSet()
    pcd.points = o3d.utility.Vector3dVector(keypoints)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Create Open3D visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live Keypoints", width=800, height=600)
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)

    # Optional: render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 5.0
    opt.line_width = 2.0

    def update_scene(keypoints):
        valid_mask = ~(np.all(keypoints == 0, axis=1))
        valid_points = keypoints[valid_mask]

        if len(valid_points) == 0:
            return  # Nothing to draw this frame

        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(valid_mask)[0])}

        valid_lines = []
        for l in lines:
            if valid_mask[l[0]] and valid_mask[l[1]]:
                valid_lines.append([idx_map[l[0]], idx_map[l[1]]])
        valid_lines = np.array(valid_lines)

        # Set colors: assign fixed color per keypoint index
        fixed_colors = np.array([
            [1, 0, 0],   # 0 - Red
            [0, 1, 0],   # 1 - Green
            [0, 0, 1],   # 2 - Blue
            [1, 1, 0],   # 3 - Yellow
            [1, 0, 1],   # 4 - Magenta
            [0, 1, 1],   # 5 - Cyan
            [0.5, 0, 0], # 6 - Dark Red
            [0, 0.5, 0], # 7 - Dark Green
            [0, 0, 0.5], # 8 - Dark Blue
            [0.5, 0.5, 0], # 9 - Olive
            [0.5, 0, 0.5], # 10 - Purple
            [0, 0.5, 0.5], # 11 - Teal
            [0.75, 0.25, 0.5], # 12 - Pinkish
            [0.5, 0.75, 0.25], # 13 - Lime-ish
            [0.25, 0.5, 0.75], # 14 - Sky blue
            [0.75, 0.5, 0.25], # 15 - Orange
            [0.25, 0.75, 0.5], # 16 - Aqua
        ])

        # Map the original color set to only the valid points
        valid_indices = np.where(valid_mask)[0]
        colors = np.array([fixed_colors[i] for i in valid_indices])

        pcd.points = o3d.utility.Vector3dVector(valid_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)  # <-- Set per-point colors

        line_set.points = o3d.utility.Vector3dVector(valid_points)
        line_set.lines = o3d.utility.Vector2iVector(valid_lines)

        vis.update_geometry(pcd)
        vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()

    try:
        while True:
            start_time = time.time()  # Start time of frame
            
            # Get the first dictionary inside the list
            keypoints_dicts = list(keypoints_3d.values())
            
            if keypoints_dicts:  # Check if list is not empty
                keypoints_dict = keypoints_dicts[0]  # Get the first dictionary
                # Now sort the keypoints by key and stack into a (17, 3) array
                keypoints_array = np.array([keypoints_dict[i] for i in range(17)])
                
                if keypoints_array.shape == (17, 3):
                    update_scene(keypoints_array)
            
            elapsed_time = time.time() - start_time
            remaining_time = frame_time - elapsed_time  # Time left to maintain the FPS
            time.sleep(max(0, remaining_time))  # Sleep only if there's time left

    except KeyboardInterrupt:
        vis.destroy_window()

def start_visualization_16p(keypoints_3d):
    '''
    Starts a real-time 3D visualization loop using Open3D.
    
    Parameters:
        get_keypoints_func: Function that returns a (16, 3) NumPy array of keypoints.
                            Points that are [0, 0, 0] are ignored.
    '''
    global pcd, line_set, lines

    # Duration of each frame in seconds
    frame_time = 1 / 30.0  # 30 FPS

    # Define the connectivity for the skeleton (same as your VisPy lines)
    lines = np.array([
        (0, 1), (1, 15), (15, 2),          # Spine
        (15, 3), (3, 4), (4, 5),           # Left Arm
        (15, 6), (6, 7), (7, 8),           # Right Arm
        (0, 9), (9, 10), (10, 11),         # Left Leg
        (0, 12), (12, 13), (13, 14)        # Right Leg
    ])

    keypoints = np.array([
                        [ 2.9745, 11.075, 44.182],
                        [ 2.7278, 5.8791, 43.63],
                        [ 2.5541, -3.7495, 41.914],
                        [ 7.2836, 0.86191, 41.845],
                        [ 8.6337, 4.9042, 37.397],
                        [ 3.1282, 4.0055, 34.757],
                        [ -2.3215, 0.50336, 44.314],
                        [ -6.1765, 4.1893, 40.116],
                        [ -2.4496, 3.8719, 35.829],
                        [ 5.9575, 11.136, 43.599], 
                        [ 4.5654, 6.7026, 37.402], 
                        [ 2.5471, 13.699, 42.952],
                        [ -0.0084665, 11.015, 44.764],
                        [ -2.053, 6.714, 39.498],
                        [ 2.2332, 12.747, 40.101],
                        [ 2.4811, 0.68263, 43.079]
    ])

    # Initialize Open3D geometries
    pcd = o3d.geometry.PointCloud()
    line_set = o3d.geometry.LineSet()

    # Create Open3D visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live Keypoints", width=800, height=600)
    pcd.points = o3d.utility.Vector3dVector(keypoints)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)

    # Optional: render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 5.0
    opt.line_width = 2.0

    def update_scene(keypoints):

        scaling_factor = 50
        keypoints *= scaling_factor

        valid_mask = ~(np.all(keypoints == 0, axis=1))
        valid_mask[0] = True  # Ensure point at index 0 is always included
        valid_points = keypoints[valid_mask]

        if len(valid_points) == 0:
            return  # Nothing to draw this frame

        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(valid_mask)[0])}

        valid_lines = []
        for l in lines:
            if valid_mask[l[0]] and valid_mask[l[1]]:
                valid_lines.append([idx_map[l[0]], idx_map[l[1]]])
        valid_lines = np.array(valid_lines)

        # Set colors: assign fixed color per keypoint index
        fixed_colors = np.array([
            [1, 0, 0],   # 0 - Red
            [0, 1, 0],   # 1 - Green
            [0, 0, 1],   # 2 - Blue
            [1, 1, 0],   # 3 - Yellow
            [1, 0, 1],   # 4 - Magenta
            [0, 1, 1],   # 5 - Cyan
            [0.5, 0, 0], # 6 - Dark Red
            [0, 0.5, 0], # 7 - Dark Green
            [0, 0, 0.5], # 8 - Dark Blue
            [0.5, 0.5, 0], # 9 - Olive
            [0.5, 0, 0.5], # 10 - Purple
            [0, 0.5, 0.5], # 11 - Teal
            [0.75, 0.25, 0.5], # 12 - Pinkish
            [0.5, 0.75, 0.25], # 13 - Lime-ish
            [0.25, 0.5, 0.75], # 14 - Sky blue
            [0.75, 0.5, 0.25], # 15 - Orange
        ])

        # Map the original color set to only the valid points
        valid_indices = np.where(valid_mask)[0]
        colors = np.array([fixed_colors[i] for i in valid_indices])

        pcd.points = o3d.utility.Vector3dVector(valid_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)  # <-- Set per-point colors

        line_set.points = o3d.utility.Vector3dVector(valid_points)
        line_set.lines = o3d.utility.Vector2iVector(valid_lines)

        vis.update_geometry(pcd)
        vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()

    try:
        while True:
            start_time = time.time()  # Start time of frame
            
            # Get the first dictionary inside the list
            keypoints_dicts = list(keypoints_3d.values())
            
            if keypoints_dicts:  # Check if list is not empty
                keypoints_dict = keypoints_dicts[0]  # Get the first dictionary
                # Now sort the keypoints by key and stack into a (16, 3) array
                keypoints_array = np.array([keypoints_dict[i] for i in range(16)])
                
                if keypoints_array.shape == (16, 3):
                    update_scene(keypoints_array)
            
            elapsed_time = time.time() - start_time
            remaining_time = frame_time - elapsed_time  # Time left to maintain the FPS
            time.sleep(max(0, remaining_time))  # Sleep only if there's time left

    except KeyboardInterrupt:
        vis.destroy_window()
    
def transform_coco_to_ntu(coco_keypoints):
    # Convert the string of COCO keypoints into a numpy array
    coco_keypoints = coco_keypoints.copy()

    # Initialize an array to store the NTU keypoints (16 keypoints, each with [x, y, z])
    ntu_keypoints = np.zeros((16, 3))

    # Define the mapping rules (index in NTU -> (index in COCO or 'midpoint' rule))
    transformation_map = [
        (0, [11, 12]),  # midpoint between COCO 11 and 12 (for NTU 0)
        (2, [3, 4]),    # midpoint between COCO 3 and 4
        (3, 5),         # COCO 5 -> NTU 5
        (4, 7),         # COCO 7 -> NTU 7
        (5, 9),         # COCO 9 -> NTU 9
        (6, 6),         # COCO 6 -> NTU 6
        (7, 8),         # COCO 8 -> NTU 8
        (8, 10),        # COCO 10 -> NTU 10
        (9, 11),        # COCO 11 -> NTU 11
        (10, 13),       # COCO 13 -> NTU 13
        (11, 15),       # COCO 15 -> NTU 15
        (12, 12),       # COCO 12 -> NTU 12
        (13, 14),       # COCO 14 -> NTU 14
        (14, 16),       # COCO 16 -> NTU 16
        (15, [5, 6]),   # midpoint between COCO 5 and 6
    ]

    # Apply the transformations based on the mapping
    for ntu_idx, coco_idx in transformation_map:
        if isinstance(coco_idx, list):
            k1 = np.array(coco_keypoints.get(coco_idx[0], np.zeros(3)))
            k2 = np.array(coco_keypoints.get(coco_idx[1], np.zeros(3)))
            ntu_keypoints[ntu_idx] = (k1 + k2) / 2
        else:
            ntu_keypoints[ntu_idx] = np.array(coco_keypoints.get(coco_idx, np.zeros(3)))

    # Special case for NTU keypoint 1 (midpoint of NTU 0 and NTU 20)
    ntu_keypoints[1] = (ntu_keypoints[0] + ntu_keypoints[15]) / 2

    return ntu_keypoints

def normalize_frame_with_base_center(keypoints):
    keypoints = np.array(keypoints)
    
    # Paso 1: Centrar en el punto 0 (base de la columna vertebral)
    base_spine = keypoints[0]
    centered_keypoints = keypoints - base_spine
    
    # Paso 2: Escalar entre -1 y 1
    min_vals = np.min(centered_keypoints, axis=0)
    max_vals = np.max(centered_keypoints, axis=0)
    
    scale_factors = np.max(np.abs([min_vals, max_vals]), axis=0)
    
    uniform_scale = np.max(scale_factors)
    if uniform_scale == 0:
        uniform_scale = 1  # Avoid division by zero

    normalized_keypoints = centered_keypoints / uniform_scale
    
    return normalized_keypoints

def align_vertical(keypoints):
    keypoints = np.array(keypoints)
    
    # Step 1: Center at keypoint 0 (base spine)
    base = keypoints[0]
    centered = keypoints - base

    # Step 2: Get the direction vector from 0 to 15
    spine_vec = centered[15]  # Since base is 0, direction is just keypoint[15] after centering

    # Normalize the spine vector
    spine_vec_norm = spine_vec / (np.linalg.norm(spine_vec) + 1e-6)

    # Target direction is the global Y-axis
    target_vec = np.array([0, 1, 0])

    # Compute the rotation axis (cross product) and angle (dot product)
    axis = np.cross(spine_vec_norm, target_vec)
    angle = np.arccos(np.clip(np.dot(spine_vec_norm, target_vec), -1.0, 1.0))

    if np.linalg.norm(axis) < 1e-6:
        rotation_matrix = np.eye(3)  # No rotation needed
    else:
        axis = axis / np.linalg.norm(axis)
        rotation = R.from_rotvec(axis * angle)
        rotation_matrix = rotation.as_matrix()

    # Apply rotation
    rotated = np.dot(centered, rotation_matrix.T)

    return rotated

def align_shoulders(keypoints):
    keypoints = np.array(keypoints)

    # Shoulder vector: from left shoulder (3) to right shoulder (6)
    shoulder_vec = keypoints[6] - keypoints[3]

    # Project shoulder vector onto XZ plane (ignore Y component)
    shoulder_vec_proj = np.array([shoulder_vec[0], 0, shoulder_vec[2]])
    norm = np.linalg.norm(shoulder_vec_proj)
    if norm < 1e-6:
        return keypoints  # Skip if shoulders collapsed
    shoulder_vec_proj /= norm

    # Target vector is X-axis
    target_vec = np.array([1, 0, 0])

    # Compute rotation around Y-axis to align projection with X-axis
    angle = np.arccos(np.clip(np.dot(shoulder_vec_proj, target_vec), -1.0, 1.0))
    cross = np.cross(shoulder_vec_proj, target_vec)
    if cross[1] < 0:
        angle = -angle  # Determine direction

    # Apply Y-axis rotation
    rot_y = R.from_euler('y', angle).as_matrix()
    rotated = np.dot(keypoints, rot_y.T)

    return rotated

def fill_missing_keypoints_coco(keypoints):
    """
    Fill missing COCO17 keypoints by symmetry and neighbor extrapolation.
    
    Parameters:
        keypoints: np.ndarray of shape (17, 2) - 2D keypoints with 0s if missing.

    Returns:
        np.ndarray of shape (17, 2) - with missing joints estimated if possible.
    """
    keypoints = keypoints.copy()
    
    # COCO17 indices for readability
    NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
    LSHOULDER, RSHOULDER = 5, 6
    LELBOW, RELBOW = 7, 8
    LWRIST, RWRIST = 9, 10
    LHIP, RHIP = 11, 12
    LKNEE, RKNEE = 13, 14
    LANKLE, RANKLE = 15, 16

    def is_valid(idx): return not np.all(keypoints[idx] == 0)

    def mirror(from_idx, ref_idx, to_idx):
        """Mirror one point across a reference midpoint."""
        if is_valid(from_idx) and is_valid(ref_idx) and not is_valid(to_idx):
            offset = keypoints[from_idx] - keypoints[ref_idx]
            keypoints[to_idx] = keypoints[ref_idx] - offset

    def extend_joint(a, b, out_idx, scale=1.0):
        """Predict out_idx by extending the vector from a -> b."""
        if is_valid(a) and is_valid(b) and not is_valid(out_idx):
            direction = keypoints[b] - keypoints[a]
            keypoints[out_idx] = keypoints[b] + direction * scale

    # Mirror left/right parts if only one side is visible
    mirror(LSHOULDER, NOSE, RSHOULDER)
    mirror(RSHOULDER, NOSE, LSHOULDER)

    mirror(LELBOW, LSHOULDER, RELBOW)
    mirror(RELBOW, RSHOULDER, LELBOW)

    mirror(LWRIST, LELBOW, RWRIST)
    mirror(RWRIST, RELBOW, LWRIST)

    mirror(LHIP, NOSE, RHIP)
    mirror(RHIP, NOSE, LHIP)

    mirror(LKNEE, LHIP, RKNEE)
    mirror(RKNEE, RHIP, LKNEE)

    mirror(LANKLE, LKNEE, RANKLE)
    mirror(RANKLE, RKNEE, LANKLE)

    # Extrapolate wrists from elbows and shoulders
    extend_joint(LSHOULDER, LELBOW, LWRIST)
    extend_joint(RSHOULDER, RELBOW, RWRIST)

    # Extrapolate ankles from knees and hips
    extend_joint(LHIP, LKNEE, LANKLE)
    extend_joint(RHIP, RKNEE, RANKLE)

    return keypoints


def process_triangulation(keypoint_dict, projection_matrices, keypoints_3d, frame_buffer):

    matched_people = match_people(keypoint_dict, projection_matrices)

    # Create a dictionary to store the 3D coordinates of people
    people_3d_coords = {}
    needed_projection_matrices = {}
    needed_keypoints = {}
    needed_confidences = {}

    # Debugging: Check the structure of matched_people
    #print(f"Matched People: {matched_people}")
    #print()

    if len(matched_people) == 0 or len(matched_people[0]) == 0 or len(matched_people[0][0]) == 0:
        return

    # Iterate over the matched people
    for person_id, kpts_dict in matched_people.items():

        if not isinstance(kpts_dict, dict):
            #print(f"Expected a dictionary for person {person_id}, but got {type(kpts_dict)}")
            continue  # Skip if the structure is not as expected

        for kpts_index, kpts in kpts_dict.items():  # Now iterating over the dictionary of keypoints
            #print(f"Processing person {person_id}, keypoint index {kpts_index} with keypoints {kpts}")
            needed_projection_matrices.clear()
            needed_keypoints.clear()
            needed_confidences.clear()

            for camera_index, kp in kpts.items():  # Assuming kpts is a dictionary with camera_index as the key
                x, y, conf = kp
                if (x, y) != (0, 0): # Skip invalid keypoints
                    needed_keypoints[camera_index] = (x, y)
                    needed_confidences[camera_index] = conf
                    if camera_index not in needed_projection_matrices:
                        needed_projection_matrices[camera_index] = projection_matrices[camera_index]

            # Perform weighted triangulation if enough keypoints are available
            if len(needed_keypoints) >= 2:
                # Perform weighted triangulation
                #print(f"Performing weighted triangulation for person {person_id}, keypoint index {kpts_index}, with keypoints {needed_keypoints}")
                X = weighted_triangulation(needed_keypoints, needed_confidences, needed_projection_matrices)
                if person_id not in people_3d_coords:
                    people_3d_coords[person_id] = {} #Create the person if it ahs no keypoints yet

                people_3d_coords[person_id][kpts_index] = X
                #print(f"Person ID: {person_id}, KeyPoint Index: {kpts_index}, 3D Coordinates: {X}")
            else:
                if person_id not in people_3d_coords:
                    people_3d_coords[person_id] = {}

                people_3d_coords[person_id][kpts_index] = [0, 0, 0]  # Placeholder for missing keypoints

    people_3d_coords_16p = people_3d_coords.copy()
    #print(f"3D keypoints before transformation: {people_3d_coords}")
    #print(f"3D keypoints before transformation (16p): {people_3d_coords_16p}")

    for person_id, coords in people_3d_coords_16p.items():
        # Change from COCO to semi-NTU format
        people_3d_coords_16p[person_id] = transform_coco_to_ntu(people_3d_coords_16p[person_id])
        #print(f"Transformed 3D keypoints for person {person_id}: {people_3d_coords_16p[person_id]}")

        # Normalize
        people_3d_coords_16p[person_id] = normalize_frame_with_base_center(people_3d_coords_16p[person_id])
        #print(f"Normalized 3D keypoints for person {person_id}: {people_3d_coords_16p[person_id]}")

        # Put the column vertical
        people_3d_coords_16p[person_id] = align_vertical(people_3d_coords_16p[person_id])
        #print(f"Aligned vertical 3D keypoints for person {person_id}: {people_3d_coords_16p[person_id]}")

        # Align shoulder to the x axis
        people_3d_coords_16p[person_id] = align_shoulders(people_3d_coords_16p[person_id])
        #print(f"Aligned shoulders 3D keypoints for person {person_id}: {people_3d_coords_16p[person_id]}")

    # Update the keypoints_3d dictionary
    keypoints_3d.clear()
    keypoints_3d.update(people_3d_coords)
    #print(f"Updated 3D keypoints: {keypoints_3d}")

    # Add the current frame's processed 3D keypoints to the rolling buffer (frame_buffer)
    frame_buffer.append(people_3d_coords_16p)
    
    # Optionally, you can print or log the last few frames
    #print(f"Stored {len(frame_buffer)} frames in buffer.")
    #print(f"Last frame 3D keypoints: {people_3d_coords_16p}")
    
    # Ensure that the buffer only stores the last 100 frames
    if len(frame_buffer) > 100:
        frame_buffer.pop(0) # Remove the oldest frame if buffer exceeds 100 frames

def call_triangulation(keypoint_dict, projection_matrices, keypoints_3d, frame_buffer):
    def run_triangulation():
        while True:
            try:
                starttime = time.monotonic()
                process_triangulation(keypoint_dict, projection_matrices, keypoints_3d, frame_buffer)

                elapsed_time = time.monotonic() - starttime
                time.sleep(max(0, 0.03333333 - elapsed_time))
            except Exception as e:
                print(f"Error in triangulation loop: {e}")
                break  # Exit the loop on unhandled exception


    def monitor_thread(triangulation_thread):
        while True:
            if not triangulation_thread.is_alive():
                print("Triangulation thread has died.")
            time.sleep(1)

    # Start the triangulation process in a separate thread
    triangulation_thread = threading.Thread(target=run_triangulation, daemon=True)
    triangulation_thread.start()

    triangulation_monitor = threading.Thread(target=monitor_thread, args=(triangulation_thread,), daemon=True)
    triangulation_monitor.start()

    triangulation_thread.join()  # Wait for the triangulation thread to finish
    triangulation_monitor.join()  # Wait for the monitor thread to finish

def calibrate_camera(camera_index, calibration_results):
    """
    Calibrate the camera and store the results in the shared dictionary.
    """
    result = get_camera_calibration_parameters(camera_index=camera_index)

    if result is not None:
        camera_matrix, dist_coeffs = result
        calibration_results[camera_index] = (camera_matrix, dist_coeffs)
        print(f"Camera {camera_index} calibration complete.")
    else:
        print(f"Camera {camera_index} calibration failed.")

def capture_frame(camera_index):
    """
    Capture a frame from the specified camera index.
    """
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to capture frame from camera {camera_index}.")
    
    return frame

def manual_point_selection(image):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select Points", image)

    cv2.imshow("Select Points", image)
    cv2.setMouseCallback("Select Points", click_event)
    cv2.waitKey(0)

    return points

def stereo_calibrate(camera_indices, calibration_results, chessboard_size=CHESSBOARD_SIZE):
    """
    Perform stereo calibration dynamically with live camera feeds. 
    Press keys to toggle between chessboard and manual point selection, and capture images.

    Parameters:
        camera_indices (list): List of two camera indices for stereo calibration.
        calibration_results (dict): Dictionary with camera calibration results.
        param_file (str): File to save calibration parameters (optional).

    Returns:
        retval, R, T, E, F: Stereo calibration parameters
        - retval: Success flag.
        - R: Rotation matrix (extrinsic).
        - T: Translation vector (extrinsic).
        - E: Essential matrix.
        - F: Fundamental matrix.
    """

    # Extract camera matrices and distortion coefficients
    camera_matrix_1, dist_coeffs_1 = calibration_results[camera_indices[0]]
    camera_matrix_2, dist_coeffs_2 = calibration_results[camera_indices[1]]

    # Prepare containers for object and image points
    objpoints = []  # 3D points in real-world space
    imgpoints1 = []  # 2D points in camera 1 image plane
    imgpoints2 = []  # 2D points in camera 2 image plane

    # Initialize chessboard parameters
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    cap1 = cv2.VideoCapture(camera_indices[0])
    cap2 = cv2.VideoCapture(camera_indices[1])

    mode = None  # Mode (chessboard/manual)
    print("Press 'c' for Chessboard Mode or 'm' for Manual Mode and then 's' to Save Pair and'q' to Quit.")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            print("Failed to capture frames from one or both cameras.")
            break

        # Display frames
        combined_frame = np.hstack((frame1, frame2))
        cv2.imshow("Stereo Calibration (Press 'c', 'm', 's', 'q')", combined_frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            mode = "chessboard"
            print("Mode: Chessboard detection")

        elif key == ord('m'):
            mode = "manual"
            print("Mode: Manual point selection")

        elif key == ord('s'):
            
            if mode == "chessboard":
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                # Detect chessboard corners
                ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
                ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

                if ret1 and ret2:
                    # Refine corner points
                    cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), 
                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001))
                    cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), 
                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001))

                    # Draw chessboard corners
                    frame1_copy = frame1.copy()
                    frame2_copy = frame2.copy()
                    cv2.drawChessboardCorners(frame1_copy, chessboard_size, corners1, ret1)
                    cv2.drawChessboardCorners(frame2_copy, chessboard_size, corners2, ret2)

                    # Combine frames side by side
                    combined_frame = np.hstack((frame1_copy, frame2_copy))

                    # Show combined frame and wait for user input
                    cv2.imshow("Chessboard Detection (Press 'y' to save, 'n' to discard)", combined_frame)
                    key = cv2.waitKey(0) & 0xFF  # Wait for a key press

                    if key == ord('y'):
                        objpoints.append(objp)
                        imgpoints1.append(corners1)
                        imgpoints2.append(corners2)
                        print(f"Captured chessboard pair. Total pairs: {len(objpoints)}")
                    elif key == ord('n'):
                        print("Chessboard corners not saved.")
                    cv2.destroyWindow("Chessboard Detection (Press 'y' to save, 'n' to discard)")

                else:
                    print("Chessboard not found. Try again.")

            elif mode == "manual":
                selected_points1 = manual_point_selection(frame1)
                for idx, point in enumerate(selected_points1):
                    # Draw a circle for each selected point
                    cv2.circle(frame1, tuple(int(i) for i in point), 5, (0, 255, 0), -1)
                    
                    # Add the point index as a label near the point
                    cv2.putText(frame1, str(idx + 1), 
                                tuple(int(i) for i in point), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                cv2.imshow("Manual Selection - Camera 1", frame1)
                print("Selected points on Camera 1. Now select points on Camera 2.")
                
                selected_points2 = manual_point_selection(frame2)

                for point in selected_points2:
                    cv2.circle(frame2, tuple(int(i) for i in point), 5, (0, 255, 0), -1)
                cv2.imshow("Manual Selection - Camera 2", frame2)

                cv2.destroyAllWindows()

                if len(selected_points1) < 6:
                    print("At least 6 points are required for stereo calibration.")
                elif len(selected_points1) == len(selected_points2):
                    objpoints.append(np.zeros((len(selected_points1), 3), dtype=np.float32))  # Placeholder 3D points
                    imgpoints1.append(np.array(selected_points1, dtype=np.float32))
                    imgpoints2.append(np.array(selected_points2, dtype=np.float32))
                    selected_points1 = None  # Reset for the next pair
                    selected_points2 = None

                    print(f"Captured manual points pair. Total pairs: {len(objpoints)}")
                else:
                    print("Point counts mismatch. Try again.")  

            else:
                    print("Select a mode first.")

        elif key == ord('q'):
            print("Exiting stereo calibration.")

            # Debugging: Check if enough pairs are collected
            if len(objpoints) < 1:
                print("Not enough data for stereo calibration.")
                return False, None, None, None, None

            # Perform stereo calibration if enough pairs are collected
            print(f"Number of point pairs: {len(objpoints)}")
            print("Performing stereo calibration...")

            try:
                retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                    objpoints, imgpoints1, imgpoints2,
                    camera_matrix_1, dist_coeffs_1,
                    camera_matrix_2, dist_coeffs_2,
                    frame1.shape[:2][::-1],  # Image size (width, height)
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
                    flags=cv2.CALIB_FIX_INTRINSIC
                )
                
                # If calibration succeeds, return the results
                cap1.release()
                cap2.release()
                cv2.destroyAllWindows()

                if retval:
                    print("Stereo calibration successful!")
                    return retval, R, T, E, F
                else:
                    print("Stereo calibration failed.")
                    return False, None, None, None, None

            except cv2.error as e:
                # If an error occurs (e.g., insufficient points), print the error message
                print(f"Error in stereo calibration: {e}. Collect more points and try again.")   

def get_default_projection_matrix(camera_matrix):
    """
    Create a default projection matrix using the camera matrix.
    This is a simple assumption where the rotation is identity and translation is zero.
    
    Parameters:
        camera_matrix (numpy.ndarray): The intrinsic camera matrix.

    Returns:
        projection_matrix (numpy.ndarray): A 3x4 projection matrix.
    """
    # Assuming no rotation and no translation (identity rotation and zero translation)
    translation_vector = np.zeros((3, 1))  # Zero translation
    
    # Create the projection matrix [K | 0] (assuming no distortion and perfect calibration)
    projection_matrix = np.hstack((camera_matrix, translation_vector))  # 3x4 matrix

    return projection_matrix

def calculate_projection_matrixes(camera_indices, calibration_results):
    """
    Calculate the projection matrix dynamically by selecting a base camera
    and calculating for a target camera. Convert the matrix to be relative to camera 0.
    """
    projection_matrices = {}
    param_file = "camera_calibration/projection_cameras" + str(len(camera_indices)) + ".pkl"

    if param_file and os.path.exists(param_file):
        # Load projection matrices from a file
        with open(param_file, 'rb') as file:
            params = pickle.load(file)
        projection_matrices = params
        print("Projection matrices loaded from file.")
    else:
        # Ensure camera 0 has a default projection matrix
        camera_matrix_0, _ = calibration_results[0]
        projection_matrices[0] =  np.hstack((camera_matrix_0, np.zeros((3, 1))))  # Default projection for camera 0

        # Dynamic calculation loop
        while len(projection_matrices) < len(camera_indices):
            print("Available cameras with projection matrices:", list(projection_matrices.keys()))

            # Base camera selection with confirmation
            while True:
                base_camera = int(input("Select a base camera from the above list: "))
                if base_camera in camera_indices:
                    if base_camera not in projection_matrices:
                        print(f"Base camera {base_camera} does not have a projection matrix. Try again.")
                        continue
                    else:
                        if confirm_camera_view(base_camera):
                            break
                        else:
                            print("Camera view confirmation failed. Try again.")
                else:
                    print("Camera doesn't exist. Try again.")

            print(f"Base camera selected: {base_camera}")
            print("Available cameras without projection matrices:", list(set(camera_indices) - set(projection_matrices.keys())))

            # Target camera selection with confirmation
            while True:
                target_camera = int(input("Select a target camera to calculate the matrix for: "))

                if target_camera in camera_indices:
                    if target_camera in projection_matrices:
                        print(f"Projection matrix for camera {target_camera} is already calculated.")
                    else:
                        if target_camera not in calibration_results:
                            print(f"Calibration results for camera {target_camera} are unavailable.")
                            continue
                        else:
                            if confirm_camera_view(target_camera):
                                break
                            else:
                                print("Camera view confirmation failed. Try again.")
                else:
                    print("Camera doesnt exist. Try again.")

            print(f"Target camera selected: {target_camera}")
            print(f"Calculating projection matrix for camera pair ({base_camera}, {target_camera}).")

            # Retrieve calibration data for base and target cameras
            camera_matrix_target, _ = calibration_results[target_camera]

            # Perform stereo calibration for relative R and T
            retval, R, T, E, F = stereo_calibrate((base_camera, target_camera), calibration_results)

            if retval:
                # Compute projection matrix for target camera relative to the base camera 
                P_base = projection_matrices[base_camera]  # 3x4 matrix of base camera (relative to camera 0)

                '''
                # Extract rotation and translation from base camera projection matrix
                R_base = P_base[:, :3]  # 3x3 rotation (already relative to camera 0)
                T_base = P_base[:, 3:]  # 3x1 translation (already relative to camera 0)

                # Compute new translation in camera 0’s coordinate system
                T_target_relative_to_0 = R_base @ T + T_base  # Apply transformation

                # Compute projection matrix for target camera
                P2 = np.hstack((camera_matrix_target @ (R_base @ R), camera_matrix_target @ T_target_relative_to_0))
                '''

                if base_camera != 0:
                    print ("Base camera is not camera 0. Not allowed atm.")
                    exit(1)

                # Step 1: Get intrinsics
                #_, K1, _, _ = cv2.decomposeProjectionMatrix(P_base)

                # Step 2: Build new extrinsics
                #Rt2 = np.hstack((R, T.reshape(3, 1)))
                # Step 3: Build new projection matrix
                RT2 = np.concatenate([R, T], axis = -1)
                P2 = calibration_results[target_camera][0] @ RT2

                projection_matrices[target_camera] = P2
                print(f"Projection matrix for camera {target_camera} calculated successfully.")

                
            else:
                print(f"Stereo calibration failed for camera pair ({base_camera}, {target_camera}).")

        with open(param_file, 'wb') as file:
            pickle.dump(projection_matrices, file)
        print(f"Projection matrices saved to {param_file}")

    return projection_matrices
            

def confirm_camera_view(camera_id):
    """
    Display the camera view for the given camera ID and confirm selection.
    Press 'y' to confirm, 'n' to go back to selection.
    """
    print(f"Displaying view for camera {camera_id}. Press 'y' to confirm or 'n' to select again.")

    cap = cv2.VideoCapture(camera_id)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {camera_id} View. Press y to confirm, n to go back", frame)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                cv2.destroyAllWindows()
                return True
            elif key == ord('n'):
                cv2.destroyAllWindows()
                return False
        

def camera_process_inference(camera_index, calibration_results, keypoint_dict):
    """
    Wait for calibration results and start inference.
    """
    # Wait until calibration results are available
    while camera_index not in calibration_results:
        pass  # Busy-waiting, can be optimized with better IPC mechanisms
    camera_matrix, dist_coeffs = calibration_results[camera_index]
    print(f"Starting inference for Camera {camera_index}.")
    run_camera_inference(camera_index, camera_matrix, dist_coeffs, keypoint_dict)

def normalize_3d_pose(kp3d, use_midhip=True):
    """
    Normalize 3D pose keypoints for skeleton-based action recognition.
    
    Args:
        kp3d: np.ndarray of shape (T, V, 3) - (frames, joints, xyz)
        use_midhip: if True, center on midpoint between left and right hips (COCO joints 11 & 12)
    
    Returns:
        torch.Tensor of shape (1, 3, T, V, 1) - ready for PySkl models
    """
    kp3d = kp3d.copy()  # Avoid modifying original
    
    T, V, C = kp3d.shape
    assert C == 3, "Input must have 3 coordinates (x,y,z)"

    # 1. Centering
    if use_midhip:
        # Mid-hip between left (11) and right (12) hips
        pelvis = (kp3d[:, 11, :] + kp3d[:, 12, :]) / 2.0
    else:
        pelvis = kp3d[:, 11, :]  # Only left hip

    kp3d_centered = kp3d - pelvis[:, None, :]  # (T, V, 3)

    # 2. Scaling based on shoulder width
    left_shoulder = kp3d[:, 5, :]  # Left shoulder
    right_shoulder = kp3d[:, 6, :] # Right shoulder
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=-1)  # (T,)

    # Avoid division by zero
    shoulder_dist = np.clip(shoulder_dist, a_min=1e-6, a_max=None)

    kp3d_normalized = kp3d_centered / shoulder_dist[:, None, None]  # (T, V, 3)

    # 3. Reorder to (N, C, T, V, M)
    kp3d_normalized = np.transpose(kp3d_normalized, (2, 0, 1))  # (3, T, V)
    kp3d_normalized = kp3d_normalized[None, ..., None]  # (1, 3, T, V, 1)

    # 4. Convert to PyTorch Tensor
    pose_tensor = torch.from_numpy(kp3d_normalized).float()

    return pose_tensor

def start_action_recognition(frame_buffer):
    """
    Start action recognition on the buffered frames.
    """
    # Load the config file and model checkpoint
    config_file = 'PoseDetection/Models/st-gcn++/config_inference.py'  # Update with the correct path to the config file
    checkpoint_file = 'PoseDetection/Models/st-gcn++/best_top1_acc_epoch_32.pth'  # Update with the correct path to the checkpoint file

    ACTION_LABELS = {
        0: "drink water (A1)",
        1: "eat meal (A2)",
        2: "brush teeth (A3)",
        3: "brush hair (A4)",
        4: "drop (A5)",
        5: "pick up (A6)",
        6: "throw (A7)",
        7: "sit down (A8)",
        8: "stand up (A9)",
        9: "clapping (A10)",
        10: "reading (A11)",
        11: "writing (A12)",
        12: "tear up paper (A13)",
        13: "put on jacket (A14)",
        14: "take off jacket (A15)",
        15: "put on a shoe (A16)",
        16: "take off a shoe (A17)",
        17: "put on glasses (A18)",
        18: "take off glasses (A19)",
        19: "put on a hat/cap (A20)",
        20: "take off a hat/cap (A21)",
        21: "cheer up (A22)",
        22: "hand waving (A23)",
        23: "kicking something (A24)",
        24: "reach into pocket (A25)",
        25: "hopping (A26)",
        26: "jump up (A27)",
        27: "phone call (A28)",
        28: "play with phone/tablet (A29)",
        29: "type on a keyboard (A30)",
        30: "point to something (A31)",
        31: "taking a selfie (A32)",
        32: "check time (A33)",
        33: "rub two hands (A34)",
        34: "nod head/bow (A35)",
        35: "shake head (A36)",
        36: "wipe face (A37)",
        37: "salute (A38)",
        38: "put palms together (A39)",
        39: "cross hands in front (A40)",
        40: "sneeze/cough (A41)",
        41: "staggering (A42)",
        42: "falling down (A43)",
        43: "headache (A44)",
        44: "chest pain (A45)",
        45: "back pain (A46)",
        46: "neck pain (A47)",
        47: "nausea/vomiting (A48)",
        48: "fan self (A49)",
        49: "punch/slap (A50)",
        50: "kicking (A51)",
        51: "pushing (A52)",
        52: "pat on back (A53)",
        53: "point finger (A54)",
        54: "hugging (A55)",
        55: "giving object (A56)",
        56: "touch pocket (A57)",
        57: "shaking hands (A58)",
        58: "walking towards (A59)",
        59: "walking apart (A60)",
        60: "put on headphone (A61)",
        61: "take off headphone (A62)",
        62: "shoot at basket (A63)",
        63: "bounce ball (A64)",
        64: "tennis bat swing (A65)",
        65: "juggle table tennis ball (A66)",
        66: "hush (A67)",
        67: "flick hair (A68)",
        68: "thumb up (A69)",
        69: "thumb down (A70)",
        70: "make OK sign (A71)",
        71: "make victory sign (A72)",
        72: "staple book (A73)",
        73: "counting money (A74)",
        74: "cutting nails (A75)",
        75: "cutting paper (A76)",
        76: "snap fingers (A77)",
        77: "open bottle (A78)",
        78: "sniff/smell (A79)",
        79: "squat down (A80)",
        80: "toss a coin (A81)",
        81: "fold paper (A82)",
        82: "ball up paper (A83)",
        83: "play magic cube (A84)",
        84: "apply cream on face (A85)",
        85: "apply cream on hand (A86)",
        86: "put on bag (A87)",
        87: "take off bag (A88)",
        88: "put object into bag (A89)",
        89: "take object out of bag (A90)",
        90: "open a box (A91)",
        91: "move heavy objects (A92)",
        92: "shake fist (A93)",
        93: "throw up cap/hat (A94)",
        94: "capitulate (A95)",
        95: "cross arms (A96)",
        96: "arm circles (A97)",
        97: "arm swings (A98)",
        98: "run on the spot (A99)",
        99: "butt kicks (A100)",
        100: "cross toe touch (A101)",
        101: "side kick (A102)",
        102: "yawn (A103)",
        103: "stretch oneself (A104)",
        104: "blow nose (A105)",
        105: "hit with object (A106)",
        106: "wield knife (A107)",
        107: "knock over (A108)",
        108: "grab stuff (A109)",
        109: "shoot with gun (A110)",
        110: "step on foot (A111)",
        111: "high-five (A112)",
        112: "cheers and drink (A113)",
        113: "carry object (A114)",
        114: "take a photo (A115)",
        115: "follow (A116)",
        116: "whisper (A117)",
        117: "exchange things (A118)",
        118: "support somebody (A119)",
        119: "rock-paper-scissors (A120)"
    }

    # Initialize the model from the configuration and checkpoint
    try:
        cfg = Config.fromfile(config_file)
        model = build_model(cfg.model)  # Build the model based on the configuration
        checkpoint = cache_checkpoint(checkpoint_file)
        load_checkpoint(model, checkpoint, map_location='cpu')
    except Exception as e:
        print(f"Error loading model or checkpoint: {e}")
        return

    model = model.cuda()  # Use GPU for inference
    model.cfg = cfg  # Set the configuration for the model
    model.eval()  # Set the model to evaluation mode

    connections = [
        (0, 1), (1, 15), (15, 2),          # Spine
        (15, 3), (3, 4), (4, 5),           # Left Arm
        (15, 6), (6, 7), (7, 8),           # Right Arm
        (0, 9), (9, 10), (10, 11),         # Left Leg
        (0, 12), (12, 13), (13, 14)        # Right Leg
    ]

    started = False  # Flag to indicate if action recognition has started

    def action_recognition(frame_buffer, model, connections):
        """
        Perform action recognition on the buffered frames.
        """
        nonlocal started

        keypoints_raw = list(frame_buffer)

        if not keypoints_raw:
            print("Frame buffer is empty. Skipping action recognition.")
            return

        try:
            # Handle dict-wrapped format like {0: np.array([...])}
            keypoints_list = [frame[0] for frame in keypoints_raw if 0 in frame]
        except Exception as e:
            print(f"Error processing frame buffer: {e}")
            return

        if not keypoints_list:
            print("No valid keypoint arrays found in frame buffer.")
            return

        try:
            keypoints_array = np.stack(keypoints_list)  # Shape: (num_frames, 16, 3)
        except Exception as e:
            print(f"Error stacking keypoints: {e}")
            return

        if keypoints_array.shape == (100, 16, 3):
            # Reshape the real-time keypoints into a list for ST-GCN++ input
            if not started:
                print("Action recognition started.")
                started = True

            # Add batch and people dimension: (1, 1, 100, 16, 3)
            keypoints_array = keypoints_array[np.newaxis, ...]  # (1, 100, 16, 3)

            # Skeleton data now has the expected shape
            skeleton_data = {
                'keypoint': keypoints_array,  # Shape (1, 100, 16, 3)
                'total_frames': 100, # Total number of frames in the sequence
                'start_index': 0,  # Start index for the sequence
                'label': -1 # Placeholder for label, not used in inference
            }

            # Run inference
            try:
                result = inference_recognizer(model, skeleton_data)
                print("Top-5 predictions:")
                for class_id, score in result:
                    if class_id in ACTION_LABELS:
                        action_name = ACTION_LABELS[class_id-1]
                        print(f"  {action_name}): {score:.2%}")
                    else:
                        print(f"  Unknown ({class_id+1}): {score:.2%}")
            except Exception as e:
                print(f"Error during inference: {e}")

        else:
            print(f"Unexpected keypoints shape: {keypoints_array.shape}. Expected (100, 16, 3).")
            return

    while True:
        try:
            starttime = time.monotonic()
            action_recognition(frame_buffer, model, connections)

            elapsed_time = time.monotonic() - starttime
            time.sleep(max(0, 0.03333333 - elapsed_time))  # Sleep to simulate 30 fps frame rate
        except Exception as e:
            print(f"Error in action recognition loop: {e}")
            break  # Exit the loop on unhandled exception



# Create a process for each camera
if __name__ == "__main__":

    with Manager() as manager: # Opción de shared_memory si muy lento

        keypoint_dict = manager.dict()
        keypoints_3d = manager.dict()
        frame_buffer = manager.list()
        

        # Define the number of cameras
        camera_indices = []
        for i in range(NUMBER_OF_CAMERAS):
            camera_indices.append(i)

        # Step 1: Calibrate all cameras
        calibration_results = {}

        for index in camera_indices:
            calibrate_camera(index, calibration_results)

        # Step 2: Get the projection matrices for each camera
        projection_matrices = calculate_projection_matrixes(camera_indices, calibration_results)

        # Step 3: Start inference for all cameras
        inference_processes = [
            Process(target=camera_process_inference, args=(index, calibration_results, keypoint_dict))
            for index in camera_indices #if index != 1 # Coment for camera exclusion, change the index to exclude that camera
        ]

        for process in inference_processes:
            process.start()

        # Step 4: Perform triangulation in a separate process
        triangulation_process = Process(target=call_triangulation, args=(keypoint_dict, projection_matrices, keypoints_3d, frame_buffer))
        triangulation_process.start()

        # Step 5: Start the visualization process if selected
        if VISUALIZATION:
            # Global objects that we update
            pcd = None
            line_set = None
            lines = None

            visualization_process = Process(target=start_visualization, args=(keypoints_3d,))
            visualization_process.start()

        # Step 6: Start action recognition for the last 100 frames
        recognition_process = Process(target=start_action_recognition, args=(frame_buffer,))
        recognition_process.start()

        for process in inference_processes:
            process.join()
        triangulation_process.join()
        if VISUALIZATION: visualization_process.join()
        recognition_process.join()
