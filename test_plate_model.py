"""
test_plate_model.py - Script kiểm tra độc lập xem model best.pt
có detect được biển số xe hay không, KHÔNG đụng vào project chính.

Cách dùng (chạy từ thư mục gốc project, chứa main.py):
    .\.venv\Scripts\python.exe test_plate_model.py "đường_dẫn_ảnh_có_xe.jpg"

Hoặc lấy 1 frame từ video:
    .\.venv\Scripts\python.exe test_plate_model.py "đường_dẫn_video.mp4" --frame 50
"""

import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python test_plate_model.py <đường_dẫn_ảnh_hoặc_video>")
        print("Ví dụ: python test_plate_model.py resources/test.jpg")
        print("Hoặc:   python test_plate_model.py video.mp4 --frame 50")
        sys.exit(1)

    input_path = sys.argv[1]
    frame_number = 0
    if "--frame" in sys.argv:
        idx = sys.argv.index("--frame")
        frame_number = int(sys.argv[idx + 1])

    print("=" * 60)
    print("KIỂM TRA MODEL best.pt - PHÁT HIỆN BIỂN SỐ")
    print("=" * 60)

    # ---- Bước 1: Load model ----
    model_path = "resources/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ KHÔNG TÌM THẤY model: {model_path}")
        print("   Hãy chạy script này từ thư mục gốc project (chứa main.py)")
        sys.exit(1)

    print(f"📂 Đang load model: {model_path}")
    from ultralytics import YOLO
    model = YOLO(model_path)
    print(f"✅ Load model thành công.")
    print(f"📋 Các class trong model: {model.names}")
    print()

    # ---- Bước 2: Lấy ảnh đầu vào ----
    import cv2
    ext = os.path.splitext(input_path)[1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        print(f"🖼️  Đọc ảnh: {input_path}")
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ Không đọc được ảnh: {input_path}")
            sys.exit(1)
    elif ext in [".mp4", ".avi", ".mkv", ".mov"]:
        print(f"🎬 Đọc video: {input_path}, lấy frame số {frame_number}")
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, image = cap.read()
        cap.release()
        if not ret:
            print(f"❌ Không đọc được frame {frame_number} từ video.")
            sys.exit(1)
    else:
        print(f"❌ Định dạng file không hỗ trợ: {ext}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"📐 Kích thước ảnh: {w}x{h}")
    print()

    # ---- Bước 3: Chạy detect với nhiều mức confidence ----
    print("🔍 Đang chạy model.predict() với các mức confidence khác nhau...")
    print("-" * 60)

    for conf_threshold in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        results = model.predict(image, conf=conf_threshold, verbose=False)
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print(f"conf={conf_threshold:.2f}  →  0 box phát hiện được")
        else:
            confs = boxes.conf.tolist()
            print(f"conf={conf_threshold:.2f}  →  {len(boxes)} box(es), "
                  f"conf cao nhất = {max(confs):.3f}, "
                  f"thấp nhất = {min(confs):.3f}")

    print("-" * 60)
    print()

    # ---- Bước 4: Detect ở conf rất thấp để xem box thực tế ----
    print("🔬 Chi tiết box ở conf=0.05 (ngưỡng rất thấp để xem mọi khả năng):")
    results = model.predict(image, conf=0.05, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("❌ KHÔNG có box nào, kể cả ở conf=0.05.")
        print()
        print("==> KẾT LUẬN: Model best.pt KHÔNG nhận ra được biển số trong ảnh này.")
        print("    Nguyên nhân có thể là:")
        print("    1. Ảnh/video test khác xa với ảnh trong tập train (góc quay, độ phân giải, ánh sáng)")
        print("    2. Biển số trong ảnh quá nhỏ / quá xa / bị che khuất")
        print("    3. Model train chưa đủ tốt cho domain ảnh này")
    else:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            box_w, box_h = x2 - x1, y2 - y1
            print(f"  Box #{i}: class='{cls_name}' conf={conf:.3f} "
                  f"bbox=({x1},{y1},{x2},{y2}) kích thước={box_w}x{box_h}px")

            # Lưu crop ra để xem bằng mắt
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                out_name = f"debug_plate_crop_{i}.jpg"
                cv2.imwrite(out_name, crop)
                print(f"           → Đã lưu ảnh crop: {out_name}")

        print()
        print("==> KẾT LUẬN: Model CÓ phát hiện được vùng nghi là biển số.")
        print("    Mở các file debug_plate_crop_*.jpg để xem có đọc được chữ bằng mắt không.")
        print("    Nếu ảnh crop quá nhỏ/mờ → đó là lý do EasyOCR đọc ra UNKNOWN.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
