import sys
sys.path.append('/home/longcv/librealsense/build/wrappers/python')
import pyrealsense2 as rs
import numpy as np

def export_t265_intrinsics():
    # 1. Cấu hình pipeline của T265 (Bắt buộc phải bật cả 2 mắt)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.fisheye, 1) # Mắt trái (Left)
    cfg.enable_stream(rs.stream.fisheye, 2) # Mắt phải (Right)
    
    try:
        profile = pipe.start(cfg)
        
        # Tạo chuỗi nội dung để ghi vào file
        output_content = "==================================================\n"
        output_content += "     THÔNG SỐ NỘI THAM CAMERA INTEL REALSENSE T265\n"
        output_content += "==================================================\n\n"
        
        # Duyệt qua từng camera (1: Trái, 2: Phải)
        for i in [1, 2]:
            cam_name = "LEFT CAMERA (FISHEYE 1)" if i == 1 else "RIGHT CAMERA (FISHEYE 2)"
            
            # Lấy stream và intrinsics từ profile
            fisheye_stream = profile.get_stream(rs.stream.fisheye, i).as_video_stream_profile()
            intrinsics = fisheye_stream.get_intrinsics()
            
            # Khởi tạo Ma trận nội tham số K
            K = np.array([
                [intrinsics.fx, 0,             intrinsics.ppx],
                [0,             intrinsics.fy, intrinsics.ppy],
                [0,             0,             1]
            ])
            
            # Lấy mảng hệ số biến dạng D
            D = np.array(intrinsics.coeffs)
            
            # Định dạng chuỗi cho từng camera
            output_content += f"--- {cam_name} ---\n"
            output_content += f"Resolution: {intrinsics.width}x{intrinsics.height}\n"
            output_content += f"Distortion Model: {intrinsics.model}\n\n"
            
            output_content += "Intrinsic Matrix (K):\n"
            output_content += f"  [{K[0,0]:12.6f} {K[0,1]:12.6f} {K[0,2]:12.6f}]\n"
            output_content += f"  [{K[1,0]:12.6f} {K[1,1]:12.6f} {K[1,2]:12.6f}]\n"
            output_content += f"  [{K[2,0]:12.6f} {K[2,1]:12.6f} {K[2,2]:12.6f}]\n\n"
            
            output_content += "Distortion Coefficients (D) [k1, k2, k3, k4]:\n"
            output_content += f"  [{D[0]:10.6f}, {D[1]:10.6f}, {D[2]:10.6f}, {D[3]:10.6f}]\n"
            output_content += "\n" + "-"*50 + "\n\n"
            
        # 2. Ghi nội dung ra file định dạng txt
        file_path = "t265_intrinsics.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(output_content)
            
        print(f" Xuất file thành công! Thông số đã được lưu tại: '{file_path}'")
        
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        
    finally:
        # Luôn dừng pipeline để giải phóng phần cứng
        pipe.stop()

if __name__ == "__main__":
    export_t265_intrinsics()