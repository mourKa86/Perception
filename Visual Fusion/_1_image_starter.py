import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import open3d as o3d


class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = self.read_calib(calib_file)
        P = calibs['P2'] # calibration of colored camera
        self.P = np.reshape(P, (3, 4))

        V2C = calibs['Tr_velo_to_cam'] # transform from lidar to camera
        self.V2C = np.reshape(V2C, (3, 4))

        R0 = calibs['R0_rect'] # rotation from lidar to camera
        self.R0 = np.reshape(R0, (3, 3))

    def read_calib(self, calib_file):
        data = {}
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def project_lidar_to_image(self, points):
        # Y(2D) = P x R0 x R|t x X (3D)
        # P: [3x4]
        # R0: [3x3]
        # R|t = Lidar2Cam: [3x4]
        # X: [3x1]

        # doing the projection but starting from the back
        # homogeneous_points = np.column_stack((points, np.ones((points.shape[0], 1)))) # nx4
        # pts_RT = np.dot(homogeneous_points, np.transpose(self.V2C)) # nx3
        # pts_RT_R0 = np.transpose(np.dot(self.R0, np.transpose(pts_RT))) # nx3
        # homogeneous_points2 = np.column_stack((pts_RT_R0, np.ones((pts_RT_R0.shape[0], 1)))) # nx4
        # pts_2d = np.dot(homogeneous_points2, np.transpose(self.P)) # nx3

        R0_homog = np.vstack((self.R0, np.array([0, 0, 0])))
        R0_homo_2 = np.column_stack((R0_homog, np.array([0, 0, 0, 1])))
        P_r0 = np.dot(self.P, R0_homo_2)
        P_r0_rt = np.dot(P_r0, np.vstack((self.V2C, np.array([0, 0, 0, 1]))))
        pts_3d_homog = np.column_stack((points, np.ones((points.shape[0], 1))))
        P_r0_rt_x = np.dot(P_r0_rt, np.transpose(pts_3d_homog))
        pts_2d = np.transpose(P_r0_rt_x)

        pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2]
        pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2]
        pts_2d = np.delete(pts_2d, 2, 1)

        return pts_2d

    def get_lidar_in_image_fov(self, pcl, xmin, ymin, xmax, ymax, return_more=False, clip_dist=2.0):
        pts_2d = self.project_lidar_to_image(pcl)
        fov_indx = (
            (pts_2d[:, 0] < xmax) &
            (pts_2d[:, 0] >= xmin) &
            (pts_2d[:, 1] < ymax) &
            (pts_2d[:, 1] >= ymin)
        )

        fov_indx = fov_indx & (pcl[:, 0] > clip_dist)  # x-axis of lidar is forward direction - clip near points
        image_pcl = pcl[fov_indx, :]
        if return_more:
            return image_pcl, pts_2d, fov_indx
        else:
            return image_pcl

    def get_lidar_on_image(self, pcl, image, debug=False):
        image_pcl, pts_2d, fov_indx = self.get_lidar_in_image_fov(pcl, 0, 0,
                                                                  image.shape[1], image.shape[0],
                                                                  True)

        if debug:
            print("image_pcl: ", image_pcl)
            print("pts_2d: ", pts_2d)
            print("fov_indx: ", fov_indx)
        self.image_pcl_2d = pts_2d[fov_indx, :]

        # homogeneous = np.hstack((image_pcl, np.ones((image_pcl.shape[0], 1))))
        # homogenous = np.column_stack((image_pcl.shape[0], 1))
        # transposed_RT = np.dot(homogenous, np.transpose(self.V2C))
        # dotted_R0 = np.transpose(np.dot(self.R0, np.transpose(transposed_RT)))
        # self.image_pcl_rect = dotted_R0
        #
        # if debug:
        #     print("FOV PC RECT", self.img_fov_pc_rect)

        cmap = plt.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        # cmap = cmap[::-1, :] # reverse the order of the colors in the map
        # cmap = 255 - cmap
        self.image_pcl = image_pcl

        for i in range(self.image_pcl_2d.shape[0]):
            # depth = self.image_pcl_rect[i, 2] # Camera Frame (z-axis of camera is forward direction)
            depth = image_pcl[i, 0]  # Lidar Frame (x-axis of lidar is forward direction)
            color = cmap[int(510.0/depth), :]  # we clipped 2 and so all depth is more than 2 so it is less than 510/2 = 255
            cv2.circle(
                image, (int(np.round(self.image_pcl_2d[i, 0])), int(np.round(self.image_pcl_2d[i, 1]))), 2,
                color=tuple(color), thickness=-1
            )

        return image




def main() -> None:
    image_files = sorted(glob.glob("data\\img\\*.png"))
    point_files = sorted(glob.glob("data\\velodyne\\*.pcd"))
    label_files = sorted(glob.glob("data\\label\\*.txt"))
    calib_files = sorted(glob.glob("data\\calib\\*.txt"))

    index = 0
    pcd_file = point_files[index]
    image = cv2.cvtColor(cv2.imread(image_files[index]), cv2.COLOR_BGR2RGB)
    cloud = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(cloud.points)

    # --------- Show image --------- #
    # f, (ax1) = plt.subplots(1, 1, figsize=(20,10))
    # ax1.imshow(image)
    # ax1.set_title('Image', fontsize=30)
    # plt.show()

    # --------- Show point cloud --------- #
    # o3d.visualization.draw_geometries([cloud])

    # --------- Quiz --------- #
    # print(points[:1,:])

    # lidar_x = 21
    # lidar_y = 2.5
    # lidar_z = 0.9
    #
    # camera_x = -lidar_y
    # camera_y = -(lidar_z + (1.73 - 1.65))
    # camera_z = lidar_x  - 0.27
    #
    # print(camera_x, camera_y, camera_z)

    # --------- Fusion --------- #
    lidar2camera = LiDAR2Camera(calib_files[index])
    # print("P:\n", lidar2camera.P)
    # print("V2C:\n", lidar2camera.V2C)
    # print("R0:\n", lidar2camera.R0)

    # print("Points:\n", points[:1, :3])
    # print("Euclidean Pixels\n "+str(lidar2camera.project_lidar_to_image(points[:1, :3])))

    # --------- Clip PCL not in image scene --------- #
    # pts_2d = lidar2camera.project_lidar_to_image(points)
    # imgfov_pcl, pts_2d, fov_indx = lidar2camera.get_lidar_in_image_fov(points, 0, 0, image.shape[1], image.shape[0], True)


    # --------- Show point cloud on image --------- #
    # img_lidar = lidar2camera.get_lidar_on_image(points, image)
    # plt.figure(figsize=(14, 7))
    # plt.imshow(img_lidar)
    # plt.show()

    # --------- show a video --------- #
    index = 4
    result_video = []
    video_images = sorted((glob.glob(f"data\\videos\\video{index}\\images\\*.png")))
    video_points = sorted((glob.glob(f"data\\videos\\video{index}\\points\\*.pcd")))
    for idx, image in enumerate(video_images):
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        pcl = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)
        result_video.append(lidar2camera.show_lidar_on_image(pcl, image))

    out = cv2.VideoWriter(f'output\\video{index}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (image.shape[1], image.shape[0]))

    for i in range(len(result_video)):
        out.write(cv2.cvtColor(result_video[i], cv2.COLOR_RGB2BGR))

    out.release()

if __name__ == '__main__':
    main()




