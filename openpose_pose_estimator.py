import os
import cv2
import subprocess
from pathlib import Path

# 定義資料夾路徑
input_dir = Path('/Users/joyfulsucessful/Desktop/AIGC/AIGC_test/woman-posing_512_8fps_8s')
output_dir = input_dir / 'output'

# 檢查輸入資料夾是否存在
if not input_dir.exists():
    raise FileNotFoundError(f"輸入資料夾不存在: {input_dir}")

# 確保輸出資料夾存在
output_dir.mkdir(parents=True, exist_ok=True)

# 嘗試多個可能的 OpenPose 路徑
possible_paths = [
    './build/examples/openpose/openpose.bin',  # 預設路徑
    '/usr/local/bin/openpose.bin',            # 系統安裝路徑
    'C:/Program Files/OpenPose/bin/OpenPoseDemo.exe',  # Windows 常見路徑
    # 添加其他可能的路徑
]

# 尋找有效的 OpenPose 路徑
openpose_binary = None
for path in possible_paths:
    if Path(path).exists():
        openpose_binary = Path(path)
        break

if openpose_binary is None:
    print("錯誤：找不到 OpenPose 執行檔！")
    print("請確認 OpenPose 已正確安裝，並提供正確的執行檔路徑。")
    print("\n可能的解決方案：")
    print("1. 確認 OpenPose 已正確安裝")
    print("2. 手動指定 OpenPose 執行檔的完整路徑")
    print("3. 將 OpenPose 加入系統環境變數")
    exit(1)

try:
    # 執行 OpenPose
    command = [
        str(openpose_binary),
        '--image_dir', str(input_dir),
        '--write_images', str(output_dir),
        '--display', '0',
        '--render_pose', '1'
    ]
    
    print(f"使用的 OpenPose 路徑: {openpose_binary}")
    print("執行命令:", ' '.join(command))
    
    # 執行命令並捕獲輸出
    result = subprocess.run(command, 
                          capture_output=True, 
                          text=True, 
                          check=True)
    
    print("\n處理完成！")
    print(f"輸出已保存到: {output_dir}")
    
except subprocess.CalledProcessError as e:
    print(f"\n執行 OpenPose 時發生錯誤: {e}")
    print(f"錯誤輸出: {e.stderr}")
    print("\n請檢查：")
    print("1. OpenPose 是否正確安裝")
    print("2. 執行檔路徑是否正確")
    print("3. 是否有足夠的權限執行")

except Exception as e:
    print(f"\n發生未預期的錯誤: {e}")