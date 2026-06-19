import sys
sys.path.append('/home/longcv/librealsense/build/wrappers/python')

import pyrealsense2 as rs
import cv2
import numpy as np
import os
import time
from datetime import datetime

# ==================================================
# 0. KHỞI TẠO MÔ HÌNH AI
# ==================================================
prototxt_path = 'MobileNetSSD_deploy.prototxt'
caffemodel_path = 'MobileNetSSD_deploy.caffemodel'

if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
    raise FileNotFoundError("Thiếu file mô hình MobileNet-SSD!")

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# ==================================================
# 1. KHỞI ĐỘNG CAMERA T265
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

# ==================================================
# 2. KHỞI TẠO FILE LOG (SAU KHI ĐÃ CÓ new_w, new_h)
# ==================================================
# Tạo thư mục logs nếu chưa tồn tại
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Tạo tên file với timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/t265_data_{timestamp}.txt"
csv_filename = f"{log_dir}/t265_data_{timestamp}.csv"

# Mở file để ghi
log_file = open(log_filename, 'w', encoding='utf-8')
csv_file = open(csv_filename, 'w', encoding='utf-8')

# Ghi header cho CSV
csv_file.write("Timestamp,KhoangCach(m),GocLech(do),CenterX,CenterY,BoxX1,BoxY1,BoxX2,BoxY2,Confidence\n")

# Ghi thông tin khởi tạo vào log
log_file.write(f"=== T265 DATA LOG ===\n")
log_file.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write(f"Image Size: {new_w}x{new_h}\n")  # Bây giờ đã có new_w, new_h
log_file.write(f"Processing Interval: 3.0 Hz\n")  # Sửa thành hằng số
log_file.write("=" * 50 + "\n\n")
log_file.flush()

# Biến đếm số frame đã ghi
frame_count = 0

# --------------------------------------------------
# THÔNG SỐ NỘI THAM GỐC CỦA MẮT TRÁI (Dùng tính góc thô)
# --------------------------------------------------
# Tâm quang học thực tế của thấu kính mắt cá thô (chưa nắn)
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
PROCESS_INTERVAL = 1.0 / 3.0

print("\n[READY] Đang quét... Thuật toán góc Fisheye gốc đã được kích hoạt.")
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
                    if class_id == 15:  # Person class
                        
                        box = detections[0, 0, i, 3:7] * np.array([new_w, new_h, new_w, new_h])
                        mx1, my1, mx2, my2 = box.astype("int")
                        
                        # --- GIẢI THUẬT TÍNH GÓC CHUẨN TỪ ẢNH MẮT CÁ GỐC ---
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
                                
                                # Tăng biến đếm
                                frame_count += 1
                                
                                # Lấy timestamp hiện tại
                                current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                
                                # --- GHI VÀO LOG FILE (dạng văn bản dễ đọc) ---
                                log_file.write(f"Frame #{frame_count:06d}\n")
                                log_file.write(f"  Time: {current_timestamp}\n")
                                log_file.write(f"  KhoangCach: {mean_Z:.3f} m\n")
                                log_file.write(f"  GocLechThuc: {angle_deg:.2f}°\n")
                                log_file.write(f"  Box: ({mx1}, {my1}) -> ({mx2}, {my2})\n")
                                log_file.write(f"  Center: ({center_mx:.1f}, {center_my:.1f})\n")
                                log_file.write(f"  Confidence: {confidence:.3f}\n")
                                log_file.write("-" * 40 + "\n")
                                log_file.flush()
                                
                                # --- GHI VÀO CSV FILE (dạng bảng cho Excel/Pandas) ---
                                csv_line = f"{current_timestamp},{mean_Z:.3f},{angle_deg:.2f},{center_mx:.1f},{center_my:.1f},{mx1},{my1},{mx2},{my2},{confidence:.3f}\n"
                                csv_file.write(csv_line)
                                csv_file.flush()
                                
                                # --- IN RA MÀN HÌNH ---
                                print(f"[DATA CHUẨN] KhoangCach: {mean_Z:.2f}m | GocLechThuc: {angle_deg:.1f}° | Frame: {frame_count}")

except KeyboardInterrupt:
    print("\n[HỆ THỐNG] Đang ngắt luồng...")
    
finally:
    # Đóng file log
    log_file.write("\n" + "=" * 50 + "\n")
    log_file.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Total Frames Recorded: {frame_count}\n")
    log_file.write("=" * 50 + "\n")
    
    log_file.close()
    csv_file.close()
    
    print(f"\n[DONE] Đã ghi {frame_count} frame dữ liệu vào:")
    print(f"  - Log file: {log_filename}")
    print(f"  - CSV file: {csv_filename}")
    
    pipe.stop()
    print("[DONE] Đã đóng hệ thống.")