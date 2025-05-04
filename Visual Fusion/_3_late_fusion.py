import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import glob
import open3d as o3d
import pandas as pd
from ultralytics import YOLO

import random

import time
import numpy as np
import cv2


def load_class_names(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def get_class_colors(class_names):
    random.seed(42)  # Make colors consistent across runs
    return {class_name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for class_name in class_names}


yolo = YOLO("yolov5s.pt")
class_names = load_class_names("Yolov4\\coco.names")
class_colors = get_class_colors(class_names)

class Object2D:
    def __init__(self, box2D):
        self.xmin = int(box2D[0])
        self.ymin = int(box2D[1])
        self.width = int(box2D[2])
        self.height = int(box2D[3])
        self.xmax = self.xmin + self.width
        self.ymax = self.ymin + self.height

        self.bbox = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        self.category = box2D[4]  # String class name
        self.confidence = float(box2D[5])

    def __repr__(self):
        return f"Object2D(Class: {self.category}, BBox: {self.bbox.tolist()}, Confidence: {self.confidence:.2f})"


def fill_2D_obstacles(pred_bboxes):
    return [Object2D(box) for box in pred_bboxes]

def run_obstacle_detection(img):
    start_time = time.time()
    candidates = yolo(img)

    _candidates = []
    result = img.copy()

    for candidate in candidates:
        boxes = candidate.boxes.xyxy.cpu().numpy()  # Extract bounding boxes (x1, y1, x2, y2)
        confs = candidate.boxes.conf.cpu().numpy()  # Extract confidence scores
        classes = candidate.boxes.cls.cpu().numpy()  # Extract class IDs

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1  # Convert (x1, y1, x2, y2) to (x, y, w, h)

            # Get class name from file (fallback to "Unknown" if class ID is invalid)
            class_name = class_names[int(cls)] if int(cls) < len(class_names) else f"Unknown({int(cls)})"
            class_color = class_colors.get(class_name, (0, 255, 0))

            # Append detection
            _candidates.append([x1, y1, w, h, class_name, conf])

            # Draw bounding box with class name
            label = f"{class_name}: {conf:.2f}"
            cv2.rectangle(result, (x1, y1), (x2, y2), class_color, 2)
            cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)

    exec_time = time.time() - start_time
    print(f"Inference Time: {exec_time:.2f} seconds")

    return result, np.array(_candidates)


class Object3d(object):
    """ 3d object label """
    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.bbox2d = np.zeros(shape=(2,2))
        self.bbox3d = np.zeros(shape=(4,2))



def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines if line.split(" ")[0]!="DontCare"]
    return objects

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
class LiDAR2Camera(object):
    def __init__(self, calib_file):
        self.P = np.reshape(self.read_calib_file(calib_file)["P2"], [3, 4])

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def project_to_image(self, pts_3d):
        # Convert to Homogeneous Coordinates
        n = pts_3d.shape[0]
        pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
        # Multiply with the P Matrix
        pts_2d = np.dot(pts_3d_extend, np.transpose(self.P))  # nx3
        # Convert Back to Cartesian
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def draw_projected_box3d(self, image, qs, color=(255, 0, 0), thickness=2):
        """ Draw 3d bounding box in image
            qs: (8,3) array of vertices for the 3d box in following order:
                1 -------- 0
            /|         /|
            2 -------- 3 .
            | |        | |
            . 5 -------- 4
            |/         |/
            6 -------- 7
        """
        qs = qs.astype(np.int32)
        for k in range(0, 4):
            # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            # use LINE_AA for opencv3
            # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k, k + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        return image

    def project_8p_to_4p(self, pts_2d):
        x0 = np.min(pts_2d[:, 0])
        x1 = np.max(pts_2d[:, 0])
        y0 = np.min(pts_2d[:, 1])
        y1 = np.max(pts_2d[:, 1])
        x0 = max(0, x0)
        # x1 = min(x1, proj.image_width)
        y0 = max(0, y0)
        # y1 = min(y1, proj.image_height)
        return np.array([x0, y0, x1, y1])

    def compute_box_3d(self, obj):
        """ Projects the 3d bounding box into the image plane.
            Returns:
                corners_2d: (8,2) array in left image coord.
                corners_3d: (8,3) array in rect camera coord.
        """
        # compute rotational matrix around yaw axis
        R = roty(obj.ry)
        # 3d bounding box dimensions
        l = obj.l
        w = obj.w
        h = obj.h

        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # rotate and translate 3d bounding box
        # corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
        corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
        corners_3d[2, :] = corners_3d[2, :] + obj.t[2]

        # only draw 3d bounding box for objs in front of the camera
        if np.any(corners_3d[2, :] < 0.1):
            corners_2d = None
            return corners_2d

        # project the 3d bounding box into the image plane
        corners_2d = self.project_to_image(np.transpose(corners_3d))

        return corners_2d

    def draw_projected_box2d(self, image, qs, color=(255, 0, 0), thickness=2):
        return cv2.rectangle(image, (int(qs[0]), int(qs[1])), (int(qs[2]), int(qs[3])), (255, 0, 0), 2)

    def get_image_with_bboxes(self, img, objects):
        img2 = np.copy(img)
        img3 = np.copy(img)
        for obj in objects:
            boxes = self.compute_box_3d(obj)
            if boxes is not None:
                obj.bbox3d = boxes
                obj.bbox2d = self.project_8p_to_4p(boxes)
                img2 = self.draw_projected_box2d(img2, obj.bbox2d)  # Draw the 2D Bounding Box
                img3 = self.draw_projected_box3d(img3, obj.bbox3d)  # Draw the 3D Bounding Box
        return img2, img3

def box_iou(box1, box2):
    """
    Computer Intersection Over Union cost
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)  # abs((box1[3] - box1[1])*(box1[2]- box1[0]))
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)  # abs((box2[3] - box2[1])*(box2[2]- box2[0]))
    union_area = (box1_area + box2_area) - inter_area

    # compute the IoU
    iou = inter_area / float(union_area)
    return iou

def associate(lidar_boxes, camera_boxes):
    """
    LiDAR boxes will represent the red bounding boxes
    Camera will represent the other bounding boxes
    Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
    """
    # Define a new IOU Matrix nxm with old and new boxes
    iou_matrix = np.zeros((len(lidar_boxes), len(camera_boxes)), dtype=np.float32)

    # Go through boxes and store the IOU value for each box
    # You can also use the more challenging cost but still use IOU as a reference for convenience (use as a filter only)
    for i, lidar_box in enumerate(lidar_boxes):
        for j, camera_box in enumerate(camera_boxes):
            iou_matrix[i][j] = box_iou(lidar_box, camera_box)

    # Call for the Hungarian Algorithm
    hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
    hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

    # Create new unmatched lists for old and new boxes
    matches, unmatched_camera_boxes, unmatched_lidar_boxes = [], [], []

    # Go through the Hungarian Matrix, if matched element has IOU < threshold (0.3), add it to the unmatched
    # Else: add the match
    for h in hungarian_matrix:
        if (iou_matrix[h[0], h[1]] > 0.3):
            matches.append(h.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches


def overlay_yolo5_boxes(image, detected_objects):
    img = image.copy()
    class_colors = {}  # Dictionary to store colors for each class

    for obj in detected_objects:
        xmin, ymin, xmax, ymax = obj.bbox
        category = obj.category
        conf = obj.confidence

        # Assign a random color to each class (once)
        if category not in class_colors:
            class_colors[category] = np.random.randint(0, 255, (3,)).tolist()

        color = class_colors[category]

        # Draw the bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, f"{category} {conf:.2f}", (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

class FusedObject(object):
    def __init__(self, bbox2d, bbox3d, category, t, confidence):
        self.bbox2d = bbox2d
        self.bbox3d = bbox3d
        self.category = category
        self.t = t
        # Read class names from file
        with open("Yolov4/classes.txt", 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        # If category is a string, find its index
        if isinstance(category, str):
            if category in classes:
                category = classes.index(category)  # Convert string to integer index
            else:
                category = -1  # Assign -1 for unknown class

        self.class_ = classes[category] if 0 <= category < len(classes) else "Unknown"

def build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, image):
    "Input: Image with 3D Boxes already drawn"
    final_image = image.copy()
    list_of_fused_objects = []
    for match in matches:
        fused_object = FusedObject(list_of_2d_objects[match[1]].bbox, list_of_3d_objects[match[0]].bbox3d,
                                   list_of_2d_objects[match[1]].category, list_of_3d_objects[match[0]].t,
                                   list_of_2d_objects[match[1]].confidence)
        cv2.putText(final_image, '{0:.2f} m'.format(fused_object.t[2]),
                    (int(fused_object.bbox2d[0]), int(fused_object.bbox2d[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (200, 200, 100), 3, cv2.LINE_AA)
        cv2.putText(final_image, fused_object.class_,
                    (int(fused_object.bbox2d[0] + 30), int(fused_object.bbox2d[1] + 30)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (200, 200, 100), 3, cv2.LINE_AA)
    return final_image, list_of_fused_objects


def pipeline(image, calib_file, label_file):
    lidar2cam = LiDAR2Camera(calib_file)    # only initialiazing P matrix
    # 1 - Run Obstacle Detection
    result, pred_bboxes = run_obstacle_detection(image)

    # 2 - Build a 2D Object
    list_of_2d_objects = fill_2D_obstacles(pred_bboxes)

    # 3 - Build a 3D Object (from labels)
    list_of_3d_objects = read_label(label_file)

    # 4 - Get the LiDAR Boxes in the Image in 2D and 3D
    lidar_2d, lidar_3d =lidar2cam.get_image_with_bboxes(image, list_of_3d_objects)

    # 5 - Associate the LiDAR boxes and the Camera Boxes
    lidar_boxes = [obs.bbox2d for obs in list_of_3d_objects] # Simply get the boxes
    camera_boxes = [obs.bbox for obs in list_of_2d_objects]
    matches = associate(lidar_boxes, camera_boxes)

    #6 - Build a Fused Object
    final_image, _ = build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, lidar_2d)

    return final_image

def main() -> None:
    image_files = sorted(glob.glob("data/img/*.png"))
    point_files = sorted(glob.glob("data/velodyne/*.pcd"))
    label_files = sorted(glob.glob("data/label/*.txt"))
    calib_files = sorted(glob.glob("data/calib/*.txt"))

    print("There are", len(image_files), "images")
    index = 4
    pcd_file = point_files[index]
    image = cv2.cvtColor(cv2.imread(image_files[index]), cv2.COLOR_BGR2RGB)
    cloud = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(cloud.points)

    # --------- Yolo --------- #
    '''
    The 2D Bounding Box coordinates (X1, Y1, X2, Y2)
    The class of the object (Car, Pedestrian, ...)
    The confidence
    '''
    result_img, raw_detections = run_obstacle_detection(image)
    detected_objects = fill_2D_obstacles(raw_detections)


    # --------- PCL Detections --------- #
    '''
    Detected Obstacles in 3D (X,Y,Z)
    Build a Bounding Box in 3D (W,H,L)
    Computed the orientation of that Bounding Box (Yaw Angle)
    '''

    list_of_3d_objects = read_label(label_files[index])
    for obj3d in list_of_3d_objects:
        print("Object Lateral Position (X), Height(Y), Distance (Z) :" + str(obj3d.t))

    # --------- Show image --------- #
    # f, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
    # ax1.imshow(result_img)
    # ax1.set_title('Image', fontsize=30)
    # plt.show()

    # --------- Show point cloud --------- #
    # o3d.visualization.draw_geometries([cloud])

    # --------- 3D Bounding Box --------- #
    # list_of_3d_objects = read_label(label_files[index])
    # lidar2camera = LiDAR2Camera(calib_files[index])
    # lidar_2d_boxes_img, lidar_3d_boxes_img = lidar2camera.get_image_with_bboxes(image, list_of_3d_objects)
    # # f, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20))
    # ax1.imshow(lidar_2d)
    # ax1.set_title('Image with 2D Bounding Boxes from LiDAR', fontsize=15)
    # ax2.imshow(lidar_3d)
    # ax2.set_title('Image with 3D Bounding Boxes from LiDAR', fontsize=15)
    # plt.show()

    # --------- 2D Yolo + 2D PCL --------- #
    # result = overlay_yolo5_boxes(lidar_2d_boxes_img, detected_objects)
    #
    # f, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
    # ax1.imshow(result)
    # ax1.set_title('Image', fontsize=30)
    # plt.show()

    # --------- Late Fusion --------- #
    # list_of_2d_objects = detected_objects
    # lidar_boxes = [obs.bbox2d for obs in list_of_3d_objects]  # Simply get the boxes
    # camera_boxes = [obs.bbox for obs in list_of_2d_objects]
    # matches = associate(lidar_boxes, camera_boxes)
    # final_image, _ = build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, lidar_2d_boxes_img)
    #
    # plt.figure(figsize=(14, 7))
    # plt.imshow(final_image)
    # plt.show()

    # --------- Pipeline --------- #
    index = 4
    final_image = pipeline(image, calib_files[index], label_files[index])
    plt.figure(figsize=(14, 7))
    plt.imshow(final_image)
    plt.show()


if __name__ == '__main__':
    main()

