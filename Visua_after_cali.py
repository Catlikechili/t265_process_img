import os
import glob
import cv2
import numpy as np

# =======================================================
# 1. ĐIỀN TRỰC TIẾP MA TRẬN K VÀ D CỦA BẠN (CAM PHẢI - FISHEYE 2)
# =======================================================
K = np.array([
    [284.611786,   0.000000, 420.233612],
    [  0.000000, 285.758209, 407.466888],
    [  0.000000,   0.000000,   1.000000]
], dtype=np.float32)

# 4 hệ số biến dạng Kannala-Brandt [k1, k2, k3, k4] lấy từ kết quả calib trước của bạn
D = np.array([-0.001190, 0.037810, -0.035387, 0.005361], dtype=np.float32)

dir_in = "captured_fisheye1"          # Thư mục chứa 50 ảnh mắt cá gốc
dir_out = "fisheye1_undistorted"  # Thư mục lưu ảnh phẳng sau khi xử lý
os.makedirs(dir_out, exist_ok=True)
# =======================================================

# Quét danh sách ảnh gốc
images = sorted(glob.glob(os.path.join(dir_in, "*.png")))
if len(images) == 0:
    print(f"[ERROR] Không tìm thấy ảnh nào trong thư mục '{dir_in}'!")
    exit()

# Lấy kích thước ảnh (Mặc định T265 là 848x800)
sample_img = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
h, w = sample_img.shape[:2]

print(f"[INFO] Kích thước ảnh phát hiện: Rộng {w} x Cao {h}")
print("[INFO] Đang dựng lưới tọa độ remap từ điểm (0,0)...")

# 2. THUẬT TOÁN GIẢ MA TRẬN MAPPING
# Hàm này tự động tính toán vị trí bị méo tương ứng trên cảm biến cho TỪNG pixel (u, v) phẳng
map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), K, (w, h), cv2.CV_32FC1
)

print(f"[INFO] Bắt đầu lật ảnh phẳng và nội suy cho {len(images)} ảnh...")

# 3. DUYỆT QUA TỪNG FILE VÀ NỘI SUY SẮC ĐỘ
for img_path in images:
    base_name = os.path.basename(img_path)
    img_distorted = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Hàm remap duyệt toán học từ gốc (0,0) đến cuối ảnh.
    # Các điểm thập phân được tính toán nội suy song tuyến tính (INTER_LINEAR) để bù màu mịn.
    # Các vùng biên không có dữ liệu gốc chiếu vào sẽ được lấp đầy bằng màu đen (BORDER_CONSTANT).
    img_undistorted = cv2.remap(
        img_distorted, map_x, map_y, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT
    )
    
    # Lưu ảnh phẳng sạch méo xuống ổ cứng
    cv2.imwrite(os.path.join(dir_out, base_name), img_undistorted)
    
    # Hiển thị trực quan quá trình lật ảnh phẳng
    comparison = np.hstack((img_distorted, img_undistorted))
    cv2.imshow("Algorithm Undistort (Origin | Flattened)", cv2.resize(comparison, (848, 400)))
    
    # Chạy cuốn chiếu tự động giữa các ảnh, nhấn 'q' để thoát sớm
    if cv2.waitKey(30) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()
print(f"\n[HOÀN THÀNH] Thuật toán xử lý xong! Ảnh phẳng lưu tại: '{dir_out}/'")