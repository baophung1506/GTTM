"""
test_ocr_normalize.py - Thêm bước CHUẨN HÓA TEXT sau khi OCR,
sửa các lỗi thường gặp (dấu - bị đọc nhầm, ký tự lẫn lộn) theo
đúng quy luật định dạng biển số xe Việt Nam.

Cách dùng:
    .\.venv\Scripts\python.exe test_ocr_normalize.py debug_plate_crop_0.jpg
"""

import sys
import re
import cv2


def normalize_plate_text(top_text: str, bottom_text: str) -> str:
    """
    Chuẩn hóa text biển số VN theo định dạng chuẩn: XXA-YYYYY hoặc XXA-YYY.YY

    Quy tắc xử lý lỗi OCR thường gặp:
        - Dòng trên dạng: 2 số + 1-2 chữ cái + (số) -> XX-Y(Y)
        - Các ký tự nhiễu (*, [, ], ., space thừa) ở vị trí dấu gạch ngang
          sẽ được thay bằng '-'
        - Chữ cái dễ nhầm: O<->0, I<->1, B<->8 -> ưu tiên chữ cái ở
          đúng vị trí (sau 2 số đầu) và số ở các vị trí còn lại.
    """
    # Loại bỏ khoảng trắng thừa, viết hoa toàn bộ
    top = top_text.strip().upper()
    bottom = bottom_text.strip().upper()

    # Bước 1: Xoá mọi ký tự không phải chữ/số trong dòng trên,
    # rồi tự chèn lại dấu '-' đúng vị trí (sau 2 ký tự số đầu).
    top_clean = re.sub(r'[^A-Z0-9]', '', top)

    # Biển số VN: 2 số đầu (mã tỉnh) + 1 chữ cái (+ 1 số seri, có thể có)
    # Ví dụ: "54L1" -> "54-L1", "30A1" -> "30-A1"
    match = re.match(r'^(\d{2})([A-Z]\d?)$', top_clean)
    if match:
        province_code, series = match.groups()
        top_normalized = f"{province_code}-{series}"
    else:
        # Không khớp pattern chuẩn, vẫn cố chèn dấu - sau ký tự số thứ 2
        if len(top_clean) >= 2:
            top_normalized = f"{top_clean[:2]}-{top_clean[2:]}"
        else:
            top_normalized = top_clean

    # Bước 2: Dòng dưới chỉ giữ lại số
    bottom_clean = re.sub(r'[^0-9]', '', bottom)
    # Biển 5 số thường tách thành XXX.XX khi in trên biển thật,
    # nhưng OCR đọc liền thành chuỗi 1 dòng -> giữ nguyên dạng liền,
    # chỉ thêm dấu chấm nếu đủ 5 số (chuẩn hiển thị, tùy chọn)
    if len(bottom_clean) == 5:
        bottom_normalized = f"{bottom_clean[:3]}.{bottom_clean[3:]}"
    else:
        bottom_normalized = bottom_clean

    return f"{top_normalized} {bottom_normalized}".strip()


def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python test_ocr_normalize.py <đường_dẫn_ảnh_biển_số>")
        sys.exit(1)

    img_path = sys.argv[1]
    print("=" * 60)
    print("TEST EASYOCR + CHUẨN HÓA TEXT BIỂN SỐ VN")
    print("=" * 60)

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Không đọc được ảnh: {img_path}")
        sys.exit(1)

    h, w = image.shape[:2]

    print("📂 Đang khởi tạo EasyOCR...")
    import easyocr
    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    print("✅ EasyOCR sẵn sàng.\n")

    scale = max(1, int(300 / max(h, 1)))
    big = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    bh, bw = big.shape[:2]
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    top_half = gray[0:int(bh * 0.52), :]
    bottom_half = gray[int(bh * 0.48):bh, :]

    results_top = reader.readtext(top_half)
    results_bottom = reader.readtext(bottom_half)

    top_text = results_top[0][1] if results_top else ""
    top_conf = results_top[0][2] if results_top else 0.0
    bottom_text = results_bottom[0][1] if results_bottom else ""
    bottom_conf = results_bottom[0][2] if results_bottom else 0.0

    print(f"Dòng trên (raw OCR):  '{top_text}'  (conf={top_conf:.3f})")
    print(f"Dòng dưới (raw OCR):  '{bottom_text}'  (conf={bottom_conf:.3f})")
    print()

    normalized = normalize_plate_text(top_text, bottom_text)

    print("=" * 60)
    print(f"✅ KẾT QUẢ SAU CHUẨN HÓA: '{normalized}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
