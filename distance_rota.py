import cv2
import numpy as np
import os
import glob
import re

# ==========================================
# 1. THÔNG SỐ CAMERA
# ==========================================

K_left = np.array([
    [311.824889, 0.000000, 409.163080],
    [0.000000, 311.267217, 415.720185],
    [0.000000, 0.000000, 1.000000]
], dtype=np.float32)

D_left = np.array([-0.094808, 0.271351, -0.464782, 0.290414], dtype=np.float32)

K_right = np.array([
    [301.412414, 0.000000, 416.551563],
    [0.000000, 302.293132, 409.849962],
    [0.000000, 0.000000, 1.000000]
], dtype=np.float32)

D_right = np.array([-0.032813, -0.023566, 0.064681, -0.024126], dtype=np.float32)

# BASELINE = 64mm
BASELINE = 0.064  # meters
BASELINE_MM = 64  # mm

WIDTH, HEIGHT = 848, 800

# ==========================================
# 2. HÀM TÍNH TOÁN
# ==========================================

def calculate_angle_and_distance(foot_x, foot_y, img_width, img_height, K):
    """Tính góc và khoảng cách từ vị trí chân trong ảnh"""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    x_norm = (foot_x - cx) / fx
    y_norm = (foot_y - cy) / fy
    
    # Góc
    angle_rad = np.arctan2(x_norm, 1.0)
    angle_deg = np.degrees(angle_rad)
    
    # Khoảng cách (giả định camera cao 1m)
    CAMERA_HEIGHT = 1000  # mm
    if y_norm > 0:
        distance_mm = (fy * CAMERA_HEIGHT) / abs(y_norm * fy)
    else:
        distance_mm = 5000
    
    distance_mm = np.clip(distance_mm, 300, 8000)
    
    return angle_deg, distance_mm, distance_mm / 1000.0

def detect_object_simple(img):
    """Phát hiện đối tượng đơn giản"""
    h, w = img.shape[:2]
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest)
            foot_x = x + w_box // 2
            foot_y = y + h_box
            return foot_x, foot_y, largest
    
    return w//2, h//2, None

def process_image_pair(left_img_path, right_img_path):
    """Xử lý cặp ảnh trái/phải"""
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    
    if left_img is None or right_img is None:
        return None, None, None
    
    left_x, left_y, _ = detect_object_simple(left_img)
    right_x, right_y, _ = detect_object_simple(right_img)
    
    angle_left, dist_left_mm, dist_left_m = calculate_angle_and_distance(
        left_x, left_y, WIDTH, HEIGHT, K_left
    )
    
    # Stereo distance (nếu có disparity)
    f = (K_left[0, 0] + K_right[0, 0]) / 2
    disparity = abs(left_x - right_x)
    if disparity > 0:
        dist_stereo_mm = (f * BASELINE_MM) / disparity
        dist_stereo_mm = np.clip(dist_stereo_mm, 300, 8000)
    else:
        dist_stereo_mm = dist_left_mm
    
    result = {
        'left_point': (left_x, left_y),
        'right_point': (right_x, right_y),
        'angle_deg': angle_left,
        'distance_left_mm': dist_left_mm,
        'distance_left_m': dist_left_m,
        'distance_stereo_mm': dist_stereo_mm,
        'distance_stereo_m': dist_stereo_mm / 1000.0,
        'baseline_mm': BASELINE_MM
    }
    
    return result, left_img, right_img

def display_results(result, left_img, right_img, save_path=None):
    """Hiển thị kết quả"""
    if result is None:
        return
    
    left_display = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    right_display = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    
    lx, ly = result['left_point']
    rx, ry = result['right_point']
    
    cv2.circle(left_display, (lx, ly), 10, (0, 0, 255), -1)
    cv2.circle(right_display, (rx, ry), 10, (0, 255, 0), -1)
    
    info = [
        f"Goc: {result['angle_deg']:.1f}°",
        f"KC Left: {result['distance_left_mm']:.0f}mm",
        f"KC Stereo: {result['distance_stereo_mm']:.0f}mm",
        f"Baseline: {result['baseline_mm']}mm"
    ]
    
    y_pos = 30
    for text in info:
        cv2.putText(left_display, text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
    
    combined = np.hstack((left_display, right_display))
    
    cv2.putText(combined, "LEFT", (WIDTH//2 - 50, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(combined, "RIGHT", (WIDTH + WIDTH//2 - 50, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    combined_resized = cv2.resize(combined, (WIDTH, HEIGHT//2))
    cv2.imshow("Distance & Angle Measurement", combined_resized)
    
    if save_path:
        cv2.imwrite(save_path, combined)
        print(f"✅ Đã lưu kết quả: {save_path}")
    
    return combined

# ==========================================
# 3. MAIN - XỬ LÝ CẶP ẢNH ĐÚNG TÊN
# ==========================================

def main():
    # Đường dẫn
    fisheye1_dir = "./fisheye1_undistorted"
    fisheye2_dir = "./fisheye2_undistorted"
    
    # Lấy tất cả ảnh từ fisheye1
    left_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        left_images.extend(glob.glob(os.path.join(fisheye1_dir, ext)))
    left_images = sorted(left_images)
    
    if not left_images:
        print(f"❌ Không tìm thấy ảnh trong {fisheye1_dir}")
        print(f"Đang tìm trong: {os.path.abspath(fisheye1_dir)}")
        return
    
    print(f"📸 Tìm thấy {len(left_images)} ảnh bên trái")
    print(f"📏 Baseline: {BASELINE_MM}mm")
    print("=" * 60)
    
    results_data = []
    
    for i, left_path in enumerate(left_images):
        # Lấy tên file (không có extension)
        basename = os.path.basename(left_path)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Tạo tên file bên phải tương ứng
        # Thay "fisheye1" thành "fisheye2"
        right_basename = name_without_ext.replace("fisheye1", "fisheye2")
        # Nếu không có "fisheye1" trong tên, thử tìm ảnh có số thứ tự tương ứng
        if right_basename == name_without_ext:
            # Tìm số trong tên
            numbers = re.findall(r'\d+', name_without_ext)
            if numbers:
                # Giả sử ảnh phải có tên fisheye2_{số}.png
                right_basename = f"fisheye2_{numbers[0]}"
            else:
                # Fallback: thử với fisheye2
                right_basename = f"fisheye2_{i}"
        
        # Thử các extension khác nhau
        right_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = os.path.join(fisheye2_dir, right_basename + ext)
            if os.path.exists(test_path):
                right_path = test_path
                break
        
        # Nếu vẫn không tìm thấy, thử với số thứ tự i
        if right_path is None:
            test_path = os.path.join(fisheye2_dir, f"fisheye2_{i}.png")
            if os.path.exists(test_path):
                right_path = test_path
            else:
                test_path = os.path.join(fisheye2_dir, f"fisheye2_{i}.jpg")
                if os.path.exists(test_path):
                    right_path = test_path
        
        if right_path is None:
            print(f"⚠️ Không tìm thấy ảnh phải cho: {basename}")
            print(f"   Đã tìm: {right_basename}.* trong {fisheye2_dir}")
            continue
        
        print(f"\n📷 Xử lý ảnh {i+1}/{len(left_images)}:")
        print(f"   Left: {os.path.basename(left_path)}")
        print(f"   Right: {os.path.basename(right_path)}")
        
        # Xử lý
        result, left_img, right_img = process_image_pair(left_path, right_path)
        
        if result:
            print(f"   👣 Vị trí Left: ({result['left_point'][0]}, {result['left_point'][1]})")
            print(f"   👣 Vị trí Right: ({result['right_point'][0]}, {result['right_point'][1]})")
            print(f"   📐 Góc: {result['angle_deg']:.1f}°")
            print(f"   📏 Khoảng cách Left: {result['distance_left_mm']:.0f}mm ({result['distance_left_m']:.2f}m)")
            print(f"   📏 Khoảng cách Stereo: {result['distance_stereo_mm']:.0f}mm ({result['distance_stereo_m']:.2f}m)")
            
            # Lưu kết quả
            results_data.append({
                'image': basename,
                'angle': result['angle_deg'],
                'dist_left_mm': result['distance_left_mm'],
                'dist_stereo_mm': result['distance_stereo_mm']
            })
            
            # Hiển thị
            save_path = f"result_{i+1:03d}.jpg"
            display_results(result, left_img, right_img, save_path)
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Lưu vào file CSV
                with open("measurement_results.csv", "a") as f:
                    f.write(f"{basename},{result['angle_deg']:.1f},{result['distance_left_mm']:.0f},{result['distance_stereo_mm']:.0f}\n")
                print("   💾 Đã lưu kết quả vào measurement_results.csv")
    
    cv2.destroyAllWindows()
    
    # In summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    for i, data in enumerate(results_data):
        print(f"{i+1:3d}. {data['image']:20s} | Góc: {data['angle']:6.1f}° | Left: {data['dist_left_mm']:5.0f}mm | Stereo: {data['dist_stereo_mm']:5.0f}mm")
    
    print(f"\n✅ Hoàn thành! Đã xử lý {len(results_data)}/{len(left_images)} ảnh.")
    print("📁 Kết quả lưu trong: measurement_results.csv")

if __name__ == "__main__":
    main()