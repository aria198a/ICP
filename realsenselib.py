import pyrealsense2 as rs
import numpy as np
import yaml
import cv2
# import pupil_apriltags
class Cam_worker():
    def __init__(self, width = 1280, height = 720, serial_num = None, frame = 30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial_num != None:
            self.config.enable_device(serial_num)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, frame)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, frame)
        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.matrix = {}
        # self.at_detector = pupil_apriltags.Detector(families='tag36h11',
        #                nthreads=1,
        #                quad_decimate=1.0,
        #                quad_sigma=0.0,
        #                refine_edges=1,
        #                decode_sharpening=0.5,
        #                debug=0)
        
    def find_device_that_supports_advanced_mode(self) :
        DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE",\
                           "0AFF", "0B00", "0B01", "0B03", "0B07","0B3A"]
        ctx = rs.context()
#        ds5_dev = rs.device()
        devices = ctx.query_devices();
        for dev in devices:
            if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
                if dev.supports(rs.camera_info.name):
                    print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
                return dev
        raise Exception("No device that supports advanced mode was found")
        
    def load_json(self, path_to_settings_file):
        
        dev = self.find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")
        file = open(path_to_settings_file, 'r')
        json_text = file.read().strip()
        file.close()

        advnc_mode.load_json(json_text)
        serialized_string = advnc_mode.serialize_json()
        print("Controls as JSON: \n", serialized_string)

    def load_matrix(self, path_to_yaml_file, matrix_name):
        skip_lines = 3
        with open(path_to_yaml_file) as infile:
            for i in range(skip_lines):
                _ = infile.readline()
            y = yaml.safe_load(infile)
        raw_matrix = np.asanyarray(y['data'])
        self.matrix[matrix_name] = raw_matrix.reshape(y['rows'], y['cols'])
        print("Total transform matrix is : \n", self.matrix[matrix_name])
        
    def take_pic(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        color_frame = aligned_frames.get_color_frame()
        self.depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
#        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
#        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)

        self.depth_image = np.asanyarray(aligned_depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())
        return self.color_image, self.depth_image
    
    def take_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        color_frame = aligned_frames.get_color_frame()

        return color_frame, aligned_depth_frame
    
    def ICS2CCS(self, pixel):
        # pixel[0] must be 1280 side & pixel[1] be 720 side
        depth_pixel = [pixel[0], pixel[1]]
        if int(pixel[1]) >= 720:
            pixel[1] = 719
        if int(pixel[0]) >= 1279:
            pixel[0] = 1279
        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, depth_pixel, self.depth_image[int(pixel[1]), int(pixel[0])])
#        if depth_point[2] == 0:
#            print('Warning depth is 0')
        return depth_point
    
    def ICS2CCS_GivenDepth(self, pixel, depth):
        # pixel[0] must be 1280 side & pixel[1] be 720 side
        depth_pixel = [pixel[0], pixel[1]]
        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, depth_pixel, depth)
#        if depth_point[2] == 0:
#            print('Warning depth is 0')
        return depth_point
    
    def CCS2RCS(self, CCS, matrix_name):
        op = np.ones([4,1])
        for j in range(3):
            op[j] = CCS[j]
#        print("op: ", op)
        if bool(self.matrix): #hasattr(self, 'matrix'):
            picking = np.dot(self.matrix[matrix_name], op)
            return picking
        else:
            print("No load matrix")
            return None
    def plotAxis(self,HTM, img, length, matrix, intrin):
        x_point = HTM[:3,3] + length * HTM[:3,0]
        x_point = np.append(x_point, 1)
        y_point = HTM[:3,3] + length * HTM[:3,1]
        y_point = np.append(y_point, 1)
        z_point = HTM[:3,3] + length * HTM[:3,2]
        z_point = np.append(z_point, 1)
#        print("HTM[:4,3]: ", HTM[:4,3])
        origin_pixel = self.RCS2ICS(matrix, intrin, HTM[:4,3])
        x_pixel = self.RCS2ICS(matrix, intrin, x_point)
        y_pixel = self.RCS2ICS(matrix, intrin, y_point)
        z_pixel = self.RCS2ICS(matrix, intrin, z_point)
#        print(tuple(np.around(origin_pixel)))
#        print(tuple(np.around(x_pixel)))
        cv2.line(img, tuple(origin_pixel), tuple(x_pixel), (0, 0, 255), 5)
        cv2.line(img, tuple(origin_pixel), tuple(y_pixel), (0, 255, 0), 5)
        cv2.line(img, tuple(origin_pixel), tuple(z_pixel), (255, 0, 0), 5)
        return img
    def find_center_surrounding(self,ICS, kernal):
        for i in range(-int(kernal/2),int(kernal/2)):
            for j in range(-int(kernal/2),int(kernal/2)):
                new_pixel = [int(ICS[0]+i), int(ICS[1]+j)]
                #print(new_pixel)
                new_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, new_pixel, self.depth_image[new_pixel[1], new_pixel[0]])
                #print("new_point: ", new_point)
                if new_point[2] != 0:
                    return new_point
        return None
    def RCS2ICS(self,matrix, intrin, point):
        temp1 = np.dot(np.linalg.inv(matrix), point)
#        print (temp1[:3])
        a = [temp1[0], temp1[1], temp1[2]]
        pixel = rs.rs2_project_point_to_pixel(intrin, a)
        out = [int(i) for i in pixel]
        return out
    def vector_rot(self, vector, matrix_name):
        op = np.ones([3,1])
        for j in range(3):
            op[j] = vector[j]
#        print("op: ", op)
        vector_rotated = np.dot(self.matrix[matrix_name][:3, :3], op)
        return vector_rotated
    # def find_apriltag(self, img):
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     tags = self.at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
    #     corner_info = []
    #     for i in tags:
    #         img = cv2.circle(img, tuple(i.center.astype(int)), 5, (0, 0, 255), 2)
    #         img = cv2.putText(img, str(i.tag_id), tuple(i.center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX,\
    #                                           1, (0, 255, 0), 3, cv2.LINE_AA)
    #         corner_CCS = []
    #         for j in i.corners:
    #             pt_corner = self.ICS2CCS(j)
    #             corner_CCS.append(pt_corner)
    #             if pt_corner[2] == 0:
    #                 img = cv2.circle(img, tuple(j.astype(int)), 6, (0, 0, 255), 2)
    #             else:
    #                 img = cv2.circle(img, tuple(j.astype(int)), 6, (0, 255, 0), 2)
    #         color_image = cv2.line(img, tuple(i.corners[0].astype(int)), \
    #                            tuple(i.corners[1].astype(int)), (0, 0, 255), 2)
    #         color_image = cv2.line(img, tuple(i.corners[0].astype(int)), \
    #                            tuple(i.corners[3].astype(int)), (0, 255, 0), 2)
    #         corner_info.append([i.tag_id, corner_CCS, i.corners])
    #     return corner_info, img
    
    def stop(self):
        self.pipeline.stop()
    def start(self):
        self.pipeline.start(self.config)
        

if __name__ == '__main__':
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        Cam_list = []
        for i in devices:
            print(i)
            Cam_list.append(Cam_worker(serial_num = i.get_info(rs.camera_info.serial_number), frame = 15))
        # Cam = Cam_worker(serial_num = "827112072119", frame = 15)
        # Cam2 = Cam_worker(serial_num = "822512061177", frame = 15)
        # Cam.load_json('wrench.json')
        while(True):
            first = True
            for i in Cam_list:
                color_image, depth_image = i.take_pic()
                if first:
                    compare = color_image
                    first = False
                else:
                    compare = np.hstack((compare, color_image))
            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            # compare = np.hstack((color_image, color_image2))
            cv2.imshow('RealSense', compare)
            k = cv2.waitKey(1)
            if k == -1:
                pass
            else:
                k = chr(k)
            if k == 'q':
                print('1')
                break
    finally:
        cv2.destroyAllWindows() 
        for i in Cam_list:
            i.stop()