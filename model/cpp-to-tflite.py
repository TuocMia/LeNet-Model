import re

# Đọc toàn bộ nội dung file .cpp
with open("model-LeNet.cc", "r", encoding="utf-8") as f:
    data = f.read()

# Tìm tất cả các byte ở dạng 0x??
hex_values = re.findall(r"0x([0-9A-Fa-f]{2})", data)

# Ghi ra file .tflite
with open("model-LeNet.tflite", "wb") as f:
    f.write(bytes(int(h, 16) for h in hex_values))

print(f"Đã tạo model.tflite ({len(hex_values)} bytes) thành công!")
