import sys
import time
import os
import cv2
import numpy as np

# 1. TRỎ ĐƯỜNG DẪN ĐẾN THƯ VIỆN PYREALSENSE2 ĐÃ BUILD
sys.path.append('/home/longcv/librealsense/build/wrappers/python')
import pyrealsense2 as rs

# 2. Cấu hình luồng camera T265
pipe = rs.pipeline()
cfg = rs.config()

# Cấu hình camera chạy ở mức 30 FPS mặc định để mượt mà nhất
TARGET_FPS = 30
cfg.enable_stream(rs.stream.fisheye, 1, 848, 800, rs.format.y8, TARGET_FPS)
cfg.enable_stream(rs.stream.fisheye, 2, 848, 800, rs.format.y8, TARGET_FPS)

# Tạo thư mục lưu ảnh mới (Tự động đổi sang thư mục snapshot để tránh lẫn lộn)
dir_cam1 = "captured_fisheye1"
dir_cam2 = "captured_fisheye2"
os.makedirs(dir_cam1, exist_ok=True)
os.makedirs(dir_cam2, exist_ok=True)

print("[INFO] Đang khởi động T265...")
pipe.start(cfg)

img_counter = 0

print("\n=======================================================")
print(f" HƯỚNG DẪN CHỤP ẢNH:")
print(f" 1. Xem màn hình live-view từ 2 mắt camera.")
print(f" 2. Nhấn phím [SPACE] (Dấu cách) để chụp ảnh bất kỳ lúc nào.")
print(f" 3. Nhấn phím [Q] để thoát chương trình.")
print("=================================================\n")

try:
    while True:
        # Chờ nhận khung hình từ camera
        frames = pipe.wait_for_frames()
        f1 = frames.get_fisheye_frame(1)
        f2 = frames.get_fisheye_frame(2)
        
        if not f1 or not f2:
            continue

        # Lấy ảnh gốc (Ảnh đen trắng Y8)
        img1 = np.asanyarray(f1.get_data())
        img2 = np.asanyarray(f2.get_data())

        # Tạo bản sao để hiển thị chữ hoặc căn chỉnh nếu cần (không ảnh hưởng ảnh gốc lưu file)
        preview1 = img1.copy()
        
        # Hiển thị số ảnh đã chụp được lên màn hình để tiện theo dõi
        cv2.putText(preview1, f"Captured: {img_counter}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(preview1, "Press SPACE to Capture", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Ghép 2 ảnh mắt trái và mắt phải lại theo chiều ngang để xem preview
        # (Sử dụng trực tiếp ảnh grayscale giúp preview siêu mượt)
        combined = np.hstack((preview1, img2))
        cv2.imshow("T265 Dual-Eye Capture Mode", combined)

        key = cv2.waitKey(1) & 0xFF
        
        # Nhấn SPACEBAR để chụp ảnh ngay lập tức
        if key == 32:
            file_path1 = os.path.join(dir_cam1, f"shot1_{img_counter:04d}.png")
            file_path2 = os.path.join(dir_cam2, f"shot2_{img_counter:04d}.png")
            
            # Lưu ảnh thô (raw) nguyên bản không bị dính chữ chữ hiển thị
            cv2.imwrite(file_path1, img1)
            cv2.imwrite(file_path2, img2)
            
            print(f"[ĐÃ LƯU] Cặp ảnh số {img_counter} thành công!")
            img_counter += 1

        # Nhấn Q để thoát
        elif key == ord('q'):
            print("[INFO] Đã dừng chương trình.")
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()