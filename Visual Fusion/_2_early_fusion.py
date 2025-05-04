import statistics

from ultralytics import YOLO
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import open3d as o3d
import random

from _1_image_starter import LiDAR2Camera

def load_class_names(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def get_class_colors(class_names):
    random.seed(42)  # Make colors consistent across runs
    return {class_name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for class_name in class_names}


yolo = YOLO("yolov5s.pt")
class_names = load_class_names("Yolov4\\coco.names")
class_colors = get_class_colors(class_names)

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
            # cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)

    exec_time = time.time() - start_time
    print(f"Inference Time: {exec_time:.2f} seconds")

    return result, np.array(_candidates)


def rectContains_abs(x1, y1, w, h, pt, shrink_factor=0.2):
    x2, y2 = x1 + w, y1 + h

    # Compute shrink amount, ensuring it's not larger than half the box size
    shrink_w = min(int(w * shrink_factor), w // 2)
    shrink_h = min(int(h * shrink_factor), h // 2)

    x1 += shrink_w
    y1 += shrink_h
    x2 -= shrink_w
    y2 -= shrink_h

    # Prevent invalid boxes
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid bounding box after shrinking: {x1}, {y1}, {x2}, {y2}")
        return False

    inside = x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2
    return inside


def filter_outliers2(distances):
    inliers = []
    mu = statistics.mean(distances)
    std = statistics.stdev(distances)
    for x in distances:
        if abs(x-mu) < std:
            inliers.append(x)
    return inliers

def get_best_distance(distances, technique='closest'):
    if technique == 'closest':
        return min(distances)
    elif technique == 'average':
        return statistics.mean(distances)
    elif technique == 'random':
        return random.choice(distances)
    else:
        return statistics.median(distances)


def lidar_camera_fusion(imgfov_pts_2d, imgfov_pcl, bboxes, image):
    img_bis = image.copy()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    filtered_points = []
    filtered_depths = []

    for box in bboxes:
        distances = []

        x1, y1, w, h = map(int, box[:4])
        x2, y2 = x1 + w, y1 + h
        valid_points = 0

        for i in range (imgfov_pts_2d.shape[0]):
            depth = imgfov_pcl[i, 0]
            pt = (imgfov_pts_2d[i, 0], imgfov_pts_2d[i, 1])
            if (rectContains_abs(x1, y1, w, h, pt, shrink_factor=0.1) == True):
                valid_points += 1
                distances.append(depth)
                filtered_points.append(pt)
                filtered_depths.append(depth)

        if len(distances) > 2:
            distances = filter_outliers2(distances)
            best_distance = get_best_distance(distances, technique='closest')
            cv2.putText(img_bis, '{:.2f} m'.format(best_distance),
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3, cv2.LINE_AA)

    for (x, y), depth in zip(filtered_points, filtered_depths):
        color = cmap[int(510.0 / depth), :]
        cv2.circle(img_bis, (int(np.round(x)), int(np.round(y))), 2, color=tuple(color), thickness=-1)

    return img_bis



def get_lidar_on_image(calib_file, pcl, image, debug=False):
    lidar2camera = LiDAR2Camera(calib_file)
    image_pcl, pts_2d, fov_indx = lidar2camera.get_lidar_in_image_fov(
        pcl, 0, 0, image.shape[1], image.shape[0], True
    )

    if debug:
        print("image_pcl: ", image_pcl)
        print("pts_2d: ", pts_2d)
        print("fov_indx: ", fov_indx)

    image_pcl_2d = pts_2d[fov_indx, :]

    cmap = plt.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    image_pcl = image_pcl

    return image_pcl_2d, image_pcl, cmap

def filter_outliers(depth_values):
    if len(depth_values) == 0:
        return None  # No valid depth values

    depth_values = np.array(depth_values)
    q1, q3 = np.percentile(depth_values, [25, 75])
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    filtered_depths = depth_values[(depth_values >= lower_bound) & (depth_values <= upper_bound)]
    if len(filtered_depths) == 0:
        return None  # If filtering removes everything, return None

    return np.median(filtered_depths)  # Best representative depth

def filter_points_by_boxes(image_pcl_2d, image_pcl, pred_bboxes):
    filtered_points = []
    filtered_depth = []

    for bbox in pred_bboxes:
        x_min, y_min, w, h = map(int, bbox[:4])
        x_max, y_max = x_min + w, y_min + h

        distances = []
        points_inside_box = []

        for i, (x, y) in enumerate(image_pcl_2d):
            if x_min <= x <= x_max and y_min <= y <= y_max:
                distances.append(image_pcl[i, 0])
                points_inside_box.append((x, y))

        best_depth = filter_outliers(distances)
        if best_depth is None:
            continue  # Skip if no valid depth remains

        for (x, y) in points_inside_box:
            filtered_points.append((x, y))
            filtered_depth.append(best_depth)

    return np.array(filtered_points), np.array(filtered_depth)

def draw_lidar_on_image(image, filtered_points, filtered_depth, cmap):
    for i in range(len(filtered_points)):
        x, y = filtered_points[i]
        depth = filtered_depth[i]

        color_index = int(510.0 / depth)
        color_index = np.clip(color_index, 0, 255)  # Prevents invalid indexing

        color = cmap[color_index, :]

        cv2.circle(
            image, (int(np.round(x)), int(np.round(y))), 2,
            color=tuple(color), thickness=-1
        )

    return image

def object_detection_amr(calib_file, image, points):
    result, pred_bboxes = run_obstacle_detection(image)  # YOLO detection
    # Step 2: Project LiDAR points onto the image
    image_pcl_2d, image_pcl, cmap = get_lidar_on_image(calib_file, points, result)
    # Step 3: Filter LiDAR points inside YOLO bounding boxes
    filtered_points, filtered_depth = filter_points_by_boxes(image_pcl_2d, image_pcl, pred_bboxes)
    # Step 4: Draw the filtered LiDAR points with the correct colormap
    final_image = draw_lidar_on_image(result, filtered_points, filtered_depth, cmap)

    return final_image

def object_detection_jeremy(calib_file, image, points):
    # Step 1: Run YOLO object detection
    image, bboxes = run_obstacle_detection(image)

    # Step 2: Project LiDAR points onto the image
    image_pcl_2d, image_pcl, cmap = get_lidar_on_image(calib_file, points, image)

    # Step 3: Fuse LiDAR points with YOLO bounding boxes
    final_image = lidar_camera_fusion(image_pcl_2d, image_pcl, bboxes, image)

    return final_image


def compare_with_ground_truth(calib_file, label_file, image, points):
    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.split(" ")[0] == "DontCare":
            x1_value = int(float(line.split(" ")[4]))
            y1_value = int(float(line.split(" ")[5]))
            x2_value = int(float(line.split(" ")[6]))
            y2_value = int(float(line.split(" ")[7]))

            dist = float(line.split(" ")[13])
            cv2.rectangle(image, (x1_value, y1_value), (x2_value, y2_value), (0, 205, 0), 10)
            cv2.putText(image, str(dist), (int((x1_value+x2_value)/2), int((y1_value+y2_value)/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 205, 0), 2, cv2.LINE_AA)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20))
    ax1.imshow(image)
    ax1.set_title('Ground Truth', fontsize=30)
    ax2.imshow(object_detection_jeremy(calib_file, image, points))
    ax2.set_title('Object Detection', fontsize=30)
    plt.show()

def main():
    image_files = sorted(glob.glob("data\\img\\*.png"))
    point_files = sorted(glob.glob("data\\velodyne\\*.pcd"))
    calib_files = sorted(glob.glob("data\\calib\\*.txt"))
    label_files = sorted(glob.glob("data\\label\\*.txt"))

    index = 0
    image = cv2.cvtColor(cv2.imread(image_files[index]), cv2.COLOR_BGR2RGB)
    cloud = o3d.io.read_point_cloud(point_files[index])
    points = np.asarray(cloud.points)
    calib_file = calib_files[index]
    label_file = label_files[index]

    # ---------- MY CODE on Image ---------- #
    # # Step 1: Run YOLO object detection
    # result, pred_bboxes = run_obstacle_detection(image)  # YOLO detection
    #
    # # # Step 2: Project LiDAR points onto the image
    # image_pcl_2d, image_pcl, cmap = get_lidar_on_image(calib_file, points, result)
    #
    # # # Step 3: Filter LiDAR points inside YOLO bounding boxes
    # filtered_points, filtered_depth = filter_points_by_boxes(image_pcl_2d, image_pcl, pred_bboxes)
    #
    # # # Step 4: Draw the filtered LiDAR points with the correct colormap
    # final_image = draw_lidar_on_image(result, filtered_points, filtered_depth, cmap)

    # ---------- Jeremy CODE on Image ---------- #
    # image, bboxes = run_obstacle_detection(image)
    # image_pcl_2d, image_pcl, cmap = get_lidar_on_image(calib_file, points, image)
    # final_image = lidar_camera_fusion(image_pcl_2d, image_pcl, bboxes, image)
    #
    # plt.figure(figsize=(14, 7))
    # plt.imshow(final_image)
    # plt.axis("off")
    # plt.show()

    # ---------- Video ---------- #
    # index = 1
    # calib_files = sorted(glob.glob("data\\calib\\*.txt"))
    # calib_file = calib_files[index]
    #
    # result_video = []
    # video_images = sorted((glob.glob(f"data\\videos\\video{index}\\images\\*.png")))
    # video_points = sorted((glob.glob(f"data\\videos\\video{index}\\points\\*.pcd")))
    # for idx, image in enumerate(video_images):
    #     image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    #     pcl = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)
    #     # result_video.append(object_detection_amr(calib_file, image, pcl))
    #     result_video.append(object_detection_jeremy(calib_file, image, pcl))
    #
    # out = cv2.VideoWriter(f'output\\video{index}_jeremy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (image.shape[1], image.shape[0]))
    #
    # for i in range(len(result_video)):
    #     out.write(cv2.cvtColor(result_video[i], cv2.COLOR_RGB2BGR))
    #
    # out.release()

    # ---------- Compare to Ground Truth ---------- #
    compare_with_ground_truth(calib_file, label_file, image, points)



if __name__ == '__main__':
    main()


