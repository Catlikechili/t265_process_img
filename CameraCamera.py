import sys
import time

# 1. TRỎ ĐƯỜNG DẪN ĐẾN THƯ VIỆN PYREALSENSE2 ĐÃ BUILD
# Thêm đường dẫn này lên đầu để tránh lỗi "AttributeError: module pyrealsense2 has no attribute pipeline"
sys.path.append('/home/longcv/librealsense/build/wrappers/python')

import pyrealsense2 as rs
import numpy as np
import cv2

# ==========================================
# CẤU HÌNH TẦN SỐ LẤY MẪU (FPS) THEO Ý MUỐN
# ==========================================
DESIRED_FPS = 5  # Bạn có thể đổi thành 2, 5, 10... tùy mức độ chịu tải của máy ảo
FRAME_INTERVAL = 1.0 / DESIRED_FPS  # Khoảng thời gian tối thiểu giữa các lần lấy hình (giây)

# Khởi tạo luồng điều khiển camera (Pipeline)
pipeline = rs.pipeline()
config = rs.config()

# Cấu hình bật luồng hình ảnh mắt cá Fisheye (Mắt 1 và Mắt 2)
# Khóa cứng độ phân giải chuẩn của phần cứng T265: 848x800, định dạng Y8 (Grayscale)
config.enable_stream(rs.stream.fisheye, 1, 848, 800, rs.format.y8, 30)
config.enable_stream(rs.stream.fisheye, 2, 848, 800, rs.format.y8, 30)

# TUYỆT ĐỐI KHÔNG bật config.enable_stream(rs.stream.pose) để giảm tải băng thông USB

try:
    # Khởi động camera với cấu hình trên
    pipeline.start(config)
    print(f"[INFO] Khởi động T265 thành công! Đang lấy mẫu ở tốc độ: {DESIRED_FPS} FPS")
    print("[INFO] Bấm phím 'q' tại cửa sổ hình ảnh để THOÁT.")
except Exception as e:
    print(f"[ERROR] Không thể kết nối với camera RealSense T265. Lỗi: {e}")
    print("[HƯỚNG DẪN] Hãy chắc chắn bạn đã cắm camera vào cổng USB 3.0 và VMware đã 'Connect' thiết bị.")
    exit()

# Biến lưu mốc thời gian của khung hình gần nhất được xử lý
last_update_time = time.time()

try:
    while True:
        # Chờ nhận gói dữ liệu thô từ camera (phần cứng vẫn gửi ngầm ở 30 FPS)
        frames = pipeline.wait_for_frames()
        
        # Kiểm tra thời gian thực tế để hạ FPS xuống theo cấu hình bằng phần mềm
        current_time = time.time()
        if current_time - last_update_time < FRAME_INTERVAL:
            # Chưa đủ thời gian giãn cách -> Bỏ qua (drop) khung hình này ngay lập tức để nhẹ máy
            continue
            
        # Đã đủ thời gian giãn cách -> Cập nhật lại mốc thời gian và tiến hành xử lý ảnh
        last_update_time = current_time

        # Trích xuất dữ liệu hình ảnh từ hai mắt kính
        f1 = frames.get_fisheye_frame(1)  # Mắt trái
        f2 = frames.get_fisheye_frame(2)  # Mắt phải
        
        # Nếu một trong hai mắt bị mất khung hình thì bỏ qua vòng lặp này
        if not f1 or not f2:
            continue

        # Chuyển đổi dữ liệu byte thô sang ma trận mảng Numpy để OpenCV có thể hiểu được
        img_left = np.asanyarray(f1.get_data())
        img_right = np.asanyarray(f2.get_data())
        
        # Ghép 2 ảnh mắt trái và mắt phải lại theo chiều ngang để hiển thị chung 1 cửa sổ
        combined_view = np.hstack((img_left, img_right))
        
        # Hiển thị luồng hình ảnh lên màn hình máy ảo Ubuntu
        cv2.imshow(f'RealSense T265 Fisheye - Throttled to {DESIRED_FPS} FPS', combined_view)
        
        # Lắng nghe sự kiện bàn phím, nếu bấm 'q' thì thoát vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Nhận tín hiệu ngắt từ bàn phím (Ctrl+C).")

finally:
    # Giải phóng camera an toàn và đóng các cửa sổ hiển thị đồ họa
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng kết nối camera và dọn dẹp hệ thống.")