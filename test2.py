import sys
sys.path.append('/home/longcv/librealsense/build/wrappers/python')
import pyrealsense2 as rs
import cv2
import numpy as np
import os
import time

# ==================================================
# 0. KHỞI TẠO MÔ HÌNH AI
# ==================================================
prototxt_path = 'MobileNetSSD_deploy.prototxt'
caffemodel_path = 'MobileNetSSD_deploy.caffemodel'

if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
    raise FileNotFoundError("Thiếu file mô hình MobileNet-SSD!")

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# ==================================================
# 1. KHỞI ĐỘNG CAMERA T265 (LUỒNG THÔ)
# ==================================================
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.fisheye, 1) 
cfg.enable_stream(rs.stream.fisheye, 2) 

print("Kích hoạt luồng thô T265 và khởi tạo cửa sổ hiển thị trực quan...")
profile = pipe.start(cfg)

orig_w, orig_h = 848, 800
SCALE_FACTOR = 0.5
new_w, new_h = int(orig_w * SCALE_FACTOR), int(orig_h * SCALE_FACTOR)
image_size = (new_w, new_h)

# Khởi tạo ma trận nắn thẳng mắt cá
K1 = np.array([[284.611786 * SCALE_FACTOR, 0.000000, 420.233612 * SCALE_FACTOR],
               [0.000000, 285.758209 * SCALE_FACTOR, 407.466888 * SCALE_FACTOR],
               [0.000000, 0.000000, 1.000000]], dtype=np.float64)
K2 = np.array([[284.981995 * SCALE_FACTOR, 0.000000, 428.147308 * SCALE_FACTOR],
               [0.000000, 286.101807 * SCALE_FACTOR, 405.857910 * SCALE_FACTOR],
               [0.000000, 0.000000, 1.000000]], dtype=np.float64)
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

try:
    while True:
        frames = pipe.poll_for_frames()
        if not frames:
            # Hãy giữ lệnh waitKey này hoạt động liên tục để tránh đơ cửa sổ OpenCV
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        f1 = frames.get_fisheye_frame(1)
        f2 = frames.get_fisheye_frame(2)
        if not f1 or not f2:
            continue
            
        img_left_raw = np.asanyarray(f1.get_data())
        img_right_raw = np.asanyarray(f2.get_data())
        
        current_time = time.time()
        if (current_time - last_process_time) >= PROCESS_INTERVAL:
            last_process_time = current_time
            
            # --- TẠO ẢNH HIỂN THỊ GỐC (ẢNH MÉO) ---
            img_left_scaled = cv2.resize(img_left_raw, image_size)
            img_display = cv2.cvtColor(img_left_scaled, cv2.COLOR_GRAY2BGR)
            
            # 1. Nhận diện thẳng trên luồng ảnh mắt cá méo
            img_left_resized = cv2.resize(img_left_raw, (300, 300))
            img_rgb_detect = cv2.cvtColor(img_left_resized, cv2.COLOR_GRAY2BGR)
            blob = cv2.dnn.blobFromImage(img_rgb_detect, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            detected_person = False
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    class_id = int(detections[0, 0, i, 1])
                    if class_id == 15:  # Tìm thấy Người
                        detected_person = True
                        
                        box = detections[0, 0, i, 3:7] * np.array([new_w, new_h, new_w, new_h])
                        (x1, y1, x2, y2) = box.astype("int")
                        
                        # Vẽ Bounding Box màu Đỏ trực tiếp lên ẢNH MÉO để kiểm tra AI nhận diện
                        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # 2. Khử méo để tính toán độ sâu chính xác
                        img_right_scaled = cv2.resize(img_right_raw, image_size)
                        rectified_left = cv2.remap(img_left_scaled, map1x, map1y, cv2.INTER_LINEAR)
                        rectified_right = cv2.remap(img_right_scaled, map2x, map2y, cv2.INTER_LINEAR)
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(new_w - 1, x2), min(new_h - 1, y2)
                        
                        shared_x1 = max(0, x1 - NUM_DISPARITIES)
                        shared_x2 = min(new_w - 1, x2)
                        shared_y1 = max(0, y1)
                        shared_y2 = min(new_h - 1, y2)
                        
                        crop_left = rectified_left[shared_y1:shared_y2, shared_x1:shared_x2]
                        crop_right = rectified_right[shared_y1:shared_y2, shared_x1:shared_x2]
                        
                        if crop_left.shape[1] <= NUM_DISPARITIES or crop_left.shape[0] <= BLOCK_SIZE:
                            continue
                            
                        crop_disparity = stereo.compute(crop_left, crop_right).astype(np.float32) / 16.0
                        valid_mask = (crop_disparity > 0.1) & (crop_disparity < NUM_DISPARITIES)
                        
                        if np.any(valid_mask):
                            mean_disp = np.mean(crop_disparity[valid_mask])
                            if mean_disp > 0:
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                
                                f = Q[2, 3]
                                B = 1.0 / Q[3, 2] if Q[3, 2] != 0 else 0.064
                                if B < 0: B = -B
                                
                                mean_Z = (f * B) / mean_disp
                                mean_X = (center_x - Q[0, 3]) * mean_Z / f
                                
                                angle_rad = np.arctan2(mean_X, mean_Z)
                                angle_deg = np.degrees(angle_rad)
                                
                                print(f"[OK] Z: {mean_Z:.2f}m | Góc: {angle_deg:.1f}°")
                                
                                # Ghi thông tin text lên màn hình live stream
                                text_info = f"{mean_Z:.2f}m | {angle_deg:.1f}deg"
                                cv2.putText(img_display, text_info, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # --- HIỂN THỊ CỬA SỔ LIVE VIEW ---
            # Ảnh hiển thị là ảnh mắt cá thực tế từ T265 giúp bạn kiểm soát được góc nhìn rộng
            cv2.imshow("Robot View - Nhận Diện Trên Ảnh Méo T265", img_display)
            
        # Nút bấm 'q' tại cửa sổ để thoát chương trình an toàn
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nĐang dừng hệ thống...")
finally:
    pipe.stop()
    cv2.destroyAllWindows()
    print("Đã giải phóng camera và đóng cửa sổ giao diện.")