import sys
import time
import cv2
import numpy as np

# 1. TRỎ ĐƯỜNG DẪN ĐẾN THƯ VIỆN PYREALSENSE2 ĐÃ BUILD
sys.path.append('/home/longcv/librealsense/build/wrappers/python')
import pyrealsense2 as rs

# Khởi tạo pipeline cấu hình cho T265
pipe = rs.pipeline()
cfg = rs.config()

# Giữ nguyên cấu hình phần cứng 30 FPS của T265 để tránh lỗi thiết bị
cfg.enable_stream(rs.stream.fisheye, 1, 848, 800, rs.format.y8, 30)
cfg.enable_stream(rs.stream.fisheye, 2, 848, 800, rs.format.y8, 30)

print("[INFO] Đang kết nối phần cứng với Intel RealSense T265...")
try:
    pipe.start(cfg)
    print("[SUCCESS] Kết nối thành công! Đang hiển thị với tốc độ 5 FPS...")
except Exception as e:
    print(f"[ERROR] Không thể mở kết nối tới thiết bị: {e}")
    sys.exit()

print("\nNhấn phím [q] tại cửa sổ hiển thị để THOÁT.")

try:
    while True:
        start_time = time.time()  # Ghi lại thời điểm bắt đầu frame

        # Chờ nhận khung hình mới từ phần cứng
        frames = pipe.wait_for_frames()
        f1 = frames.get_fisheye_frame(1)
        f2 = frames.get_fisheye_frame(2)
        
        if not f1 or not f2:
            continue

        # Chuyển mảng dữ liệu thô sang NumPy để hiển thị
        img_l = np.asanyarray(f1.get_data())
        img_r = np.asanyarray(f2.get_data())

        # Tạo bản sao để ghi chữ thông tin FPS lên màn hình
        vis_l = img_l.copy()
        cv2.putText(vis_l, "FPS: 5 (HET THONG DA DUOC HAM TOC DO)", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Ghép song song 2 mắt camera nằm ngang
        combined = np.hstack((vis_l, img_r))

        # Hiển thị video gốc lên màn hình
        cv2.imshow("T265 Test 5 FPS (Left | Right Raw)", cv2.resize(combined, (848, 400)))

        # Nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Đang ngắt kết nối chủ động.")
            break

        # --- THUẬT TOÁN HÃM FRAME RATE XUỐNG ĐÚNG 5 FPS ---
        time_elapsed = time.time() - start_time
        time_to_sleep = 0.2 - time_elapsed  # 0.2 giây ứng với 5 FPS
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

finally:
    pipe.stop()
    cv2.destroyAllWindows()
    print("[INFO] Đã giải phóng camera an toàn.")