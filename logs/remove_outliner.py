import sys
sys.path.append('/home/longcv/librealsense/build/wrappers/python')

import pyrealsense2 as rs
import cv2
import numpy as np
import os
import time
from collections import deque

# ==================================================
# 0. KHỞI TẠO MÔ HÌNH AI
# ==================================================
prototxt_path = 'MobileNetSSD_deploy.prototxt'
caffemodel_path = 'MobileNetSSD_deploy.caffemodel'

if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
    raise FileNotFoundError("Thiếu file mô hình MobileNet-SSD!")

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# ==================================================
# 1. BỘ LỌC TRUNG BÌNH (MOVING AVERAGE FILTER)
# ==================================================
class MovingAverageFilter:
    """
    Bộ lọc trung bình động (Moving Average)
    - Lưu N giá trị gần nhất
    - Trả về trung bình của N giá trị đó
    - Giúp làm mượt dữ liệu và loại bỏ nhiễu đột biến
    """
    def __init__(self, window_size=5):
        """
        Args:
            window_size (int): Số lượng điểm dùng để tính trung bình
                              - Nhỏ (3): Phản ứng nhanh, ít làm mượt
                              - Lớn (10): Phản ứng chậm, làm mượt nhiều
        """
        self.window_size = window_size
        self.distance_buffer = deque(maxlen=window_size)
        self.angle_buffer = deque(maxlen=window_size)
        self.last_filtered_distance = 0.0
        self.last_filtered_angle = 0.0
        self.is_initialized = False
        
    def filter(self, new_distance, new_angle):
        """
        Lọc dữ liệu mới
        
        Returns:
            tuple: (filtered_distance, filtered_angle)
        """
        # Thêm dữ liệu mới vào buffer
        self.distance_buffer.append(new_distance)
        self.angle_buffer.append(new_angle)
        
        # Nếu chưa đủ dữ liệu, trả về giá trị gốc
        if len(self.distance_buffer) < self.window_size:
            self.last_filtered_distance = new_distance
            self.last_filtered_angle = new_angle
            self.is_initialized = False
        else:
            # Tính trung bình của buffer
            self.last_filtered_distance = np.mean(self.distance_buffer)
            self.last_filtered_angle = np.mean(self.angle_buffer)
            self.is_initialized = True
            
        return self.last_filtered_distance, self.last_filtered_angle
    
    def get_raw_buffer(self):
        """Lấy dữ liệu thô trong buffer (để debug)"""
        return {
            'distance': list(self.distance_buffer),
            'angle': list(self.angle_buffer)
        }
    
    def get_info(self):
        """Thông tin trạng thái của filter"""
        return {
            'window_size': self.window_size,
            'buffer_size': len(self.distance_buffer),
            'is_initialized': self.is_initialized,
            'current_distance': self.last_filtered_distance,
            'current_angle': self.last_filtered_angle
        }

# ==================================================
# 2. BỘ LỌC KẾT HỢP (Median + Average)
# ==================================================
class CombinedFilter:
    """
    Bộ lọc kết hợp:
    1. Median Filter: Loại bỏ outlier đột biến
    2. Moving Average: Làm mượt dữ liệu
    """
    def __init__(self, median_window=3, avg_window=5):
        """
        Args:
            median_window (int): Cửa sổ cho Median Filter (loại outlier)
            avg_window (int): Cửa sổ cho Moving Average (làm mượt)
        """
        self.median_window = median_window
        self.avg_window = avg_window
        
        # Buffer cho Median Filter
        self.median_dist_buffer = deque(maxlen=median_window)
        self.median_angle_buffer = deque(maxlen=median_window)
        
        # Buffer cho Moving Average
        self.avg_dist_buffer = deque(maxlen=avg_window)
        self.avg_angle_buffer = deque(maxlen=avg_window)
        
        self.last_filtered_distance = 0.0
        self.last_filtered_angle = 0.0
        
    def filter(self, new_distance, new_angle):
        """
        Lọc dữ liệu qua 2 bước:
        Bước 1: Median Filter - Loại bỏ outlier
        Bước 2: Moving Average - Làm mượt
        """
        # --- BƯỚC 1: MEDIAN FILTER (loại outlier) ---
        self.median_dist_buffer.append(new_distance)
        self.median_angle_buffer.append(new_angle)
        
        if len(self.median_dist_buffer) < self.median_window:
            # Chưa đủ dữ liệu, dùng giá trị gốc
            median_dist = new_distance
            median_angle = new_angle
        else:
            # Lấy median của buffer
            median_dist = np.median(self.median_dist_buffer)
            median_angle = np.median(self.median_angle_buffer)
        
        # --- BƯỚC 2: MOVING AVERAGE (làm mượt) ---
        self.avg_dist_buffer.append(median_dist)
        self.avg_angle_buffer.append(median_angle)
        
        if len(self.avg_dist_buffer) < self.avg_window:
            # Chưa đủ dữ liệu, dùng median
            filtered_dist = median_dist
            filtered_angle = median_angle
        else:
            # Tính trung bình
            filtered_dist = np.mean(self.avg_dist_buffer)
            filtered_angle = np.mean(self.avg_angle_buffer)
        
        self.last_filtered_distance = filtered_dist
        self.last_filtered_angle = filtered_angle
        
        return filtered_dist, filtered_angle
    
    def get_info(self):
        return {
            'median_buffer_size': len(self.median_dist_buffer),
            'avg_buffer_size': len(self.avg_dist_buffer),
            'current_distance': self.last_filtered_distance,
            'current_angle': self.last_filtered_angle
        }

# ==================================================
# 3. KHỞI TẠO BỘ LỌC
# ==================================================
# Lựa chọn 1: Chỉ dùng Moving Average
filter_simple = MovingAverageFilter(window_size=5)

# Lựa chọn 2: Dùng kết hợp Median + Average (khuyến nghị)
filter_combined = CombinedFilter(median_window=3, avg_window=5)

# Chọn loại filter muốn dùng
FILTER_TYPE = 'combined'  # 'simple' hoặc 'combined'

if FILTER_TYPE == 'simple':
    data_filter = filter_simple
    print("[FILTER] Sử dụng Moving Average (window=5)")
else:
    data_filter = filter_combined
    print("[FILTER] Sử dụng Median (window=3) + Moving Average (window=5)")

# ==================================================
# 4. KHỞI ĐỘNG CAMERA T265
# ==================================================
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.fisheye, 1) 
cfg.enable_stream(rs.stream.fisheye, 2) 

print("[HỆ THỐNG] Đang khởi động luồng tính toán góc mắt cá chuẩn...")
profile = pipe.start(cfg)

orig_w, orig_h = 848, 800
SCALE_FACTOR = 0.5
new_w, new_h = int(orig_w * SCALE_FACTOR), int(orig_h * SCALE_FACTOR)
image_size = (new_w, new_h)

# --------------------------------------------------
# THÔNG SỐ NỘI THAM GỐC CỦA MẮT TRÁI (Dùng tính góc thô)
# --------------------------------------------------
cx_raw = 420.233612 * SCALE_FACTOR
cy_raw = 407.466888 * SCALE_FACTOR
fx_raw = 284.611786 * SCALE_FACTOR
fy_raw = 285.758209 * SCALE_FACTOR

# Ma trận dùng cho Stereo phẳng (Giữ nguyên để tính khoảng cách Z)
K1 = np.array([[fx_raw, 0.0, cx_raw], [0.0, fy_raw, cy_raw], [0.0, 0.0, 1.0]], dtype=np.float64)
K2 = np.array([[284.981995 * SCALE_FACTOR, 0.0, 428.147308 * SCALE_FACTOR],
               [0.0, 286.101807 * SCALE_FACTOR, 405.857910 * SCALE_FACTOR],
               [0.0, 0.0, 1.0]], dtype=np.float64)
D1 = np.array([-0.005314, 0.041641, -0.039234, 0.007532], dtype=np.float64)
D2 = np.array([-0.005123, 0.040124, -0.037531, 0.007122], dtype=np.float64)
R_stereo = np.eye(3, dtype=np.float64)
T_stereo = np.array([[-0.064], [0.0], [0.0]], dtype=np.float64) 

R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K1, D1, K2, D2, image_size, R_stereo, T_stereo, flags=cv2.fisheye.CALIB_ZERO_DISPARITY, balance=0.0)

map1x, map1y = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

NUM_DISPARITIES = 64
BLOCK_SIZE = 5
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE, P1=8*3*5**2, P2=32*3*5**2)

last_process_time = 0
PROCESS_INTERVAL = 1.0 / 15.0

print("\n[READY] Đang quét... Bộ lọc trung bình đã được kích hoạt.")
print("---------------------------------------------------------")

try:
    while True:
        frames = pipe.poll_for_frames()
        if not frames:
            time.sleep(0.01)
            continue
            
        f1 = frames.get_fisheye_frame(1)
        f2 = frames.get_fisheye_frame(2)
        if not f1 or not f2: continue
            
        current_time = time.time()
        if (current_time - last_process_time) >= PROCESS_INTERVAL:
            last_process_time = current_time
            
            img_left_raw = np.asanyarray(f1.get_data())
            img_right_raw = np.asanyarray(f2.get_data())
            
            img_left_resized = cv2.resize(img_left_raw, (300, 300))
            img_rgb_detect = cv2.cvtColor(img_left_resized, cv2.COLOR_GRAY2BGR)
            blob = cv2.dnn.blobFromImage(img_rgb_detect, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    class_id = int(detections[0, 0, i, 1])
                    if class_id == 15: 
                        
                        box = detections[0, 0, i, 3:7] * np.array([new_w, new_h, new_w, new_h])
                        mx1, my1, mx2, my2 = box.astype("int")
                        
                        # --- TÍNH GÓC LỆCH ---
                        center_mx = (mx1 + mx2) / 2.0
                        center_my = (my1 + my2) / 2.0
                        
                        x_ang = center_mx - cx_raw
                        y_ang = center_my - cy_raw
                        
                        r = np.sqrt(x_ang**2 + y_ang**2)
                        
                        if r == 0:
                            angle_deg = 0.0
                        else:
                            theta = r / fx_raw 
                            sin_phi = x_ang / r
                            angle_rad = theta * sin_phi
                            angle_deg = np.degrees(angle_rad)
                        
                        # --- TÍNH KHOẢNG CÁCH Z ---
                        distorted_pts = np.array([[[mx1, my1]], [[mx2, my2]]], dtype=np.float32)
                        undistorted_pts = cv2.fisheye.undistortPoints(distorted_pts, K1, D1, R=R1, P=P1)
                        
                        ux1 = int(undistorted_pts[0][0][0])
                        uy1 = int(undistorted_pts[0][0][1])
                        ux2 = int(undistorted_pts[1][0][0])
                        uy2 = int(undistorted_pts[1][0][1])
                        
                        ux1, uy1 = max(0, ux1), max(0, uy1)
                        ux2, uy2 = min(new_w - 1, ux2), min(new_h - 1, uy2)
                        
                        shared_ux1 = max(0, ux1 - NUM_DISPARITIES)
                        shared_ux2 = min(new_w - 1, ux2)
                        shared_uy1 = max(0, uy1)
                        shared_uy2 = min(new_h - 1, uy2)
                        
                        if (shared_ux2 - shared_ux1) <= NUM_DISPARITIES or (shared_uy2 - shared_uy1) <= BLOCK_SIZE:
                            continue

                        img_left_scaled = cv2.resize(img_left_raw, image_size)
                        img_right_scaled = cv2.resize(img_right_raw, image_size)
                        
                        rectified_left = cv2.remap(img_left_scaled, map1x, map1y, cv2.INTER_LINEAR)
                        rectified_right = cv2.remap(img_right_scaled, map2x, map2y, cv2.INTER_LINEAR)
                        
                        crop_left = rectified_left[shared_uy1:shared_uy2, shared_ux1:shared_ux2]
                        crop_right = rectified_right[shared_uy1:shared_uy2, shared_ux1:shared_ux2]
                        
                        crop_disparity = stereo.compute(crop_left, crop_right).astype(np.float32) / 16.0
                        valid_mask = (crop_disparity > 0.1) & (crop_disparity < NUM_DISPARITIES)
                        
                        if np.any(valid_mask):
                            mean_disp = np.mean(crop_disparity[valid_mask])
                            if mean_disp > 0:
                                f_q = Q[2, 3]
                                B = 1.0 / Q[3, 2] if Q[3, 2] != 0 else 0.064
                                if B < 0: B = -B
                                mean_Z = (f_q * B) / mean_disp
                                
                                # === ÁP DỤNG BỘ LỌC TRUNG BÌNH ===
                                filtered_distance, filtered_angle = data_filter.filter(mean_Z, angle_deg)
                                
                                # --- In kết quả ---
                                # Hiển thị cả raw và filtered để so sánh
                                print(f"[RAW] Dist: {mean_Z:.2f}m | Angle: {angle_deg:.1f}°")
                                print(f"[FILTERED] Dist: {filtered_distance:.2f}m | Angle: {filtered_angle:.1f}°")
                                print("-" * 50)

except KeyboardInterrupt:
    print("\n[HỆ THỐNG] Đang ngắt luồng...")
finally:
    pipe.stop()
    print("[DONE] Đã đóng hệ thống.")