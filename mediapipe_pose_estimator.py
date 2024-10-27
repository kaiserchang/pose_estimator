import cv2
import mediapipe as mp
from pathlib import Path
import numpy as np
from tqdm import tqdm  # 用於顯示進度條

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化姿態檢測器，使用更高精度的設置
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # 使用最高精度模型
            enable_segmentation=True,  # 啟用分割
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_directory(self, input_dir, output_dir):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 支援多種圖片格式
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(input_path.glob(ext)))
        
        # 排序文件以確保處理順序一致
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"警告: 在 {input_dir} 中未找到圖片文件")
            return

        print(f"\n開始處理 {len(image_files)} 張圖片...")
        
        # 使用 tqdm 顯示進度條
        for image_file in tqdm(image_files, desc="處理進度"):
            try:
                # 讀取圖片
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"\n無法讀取圖片: {image_file}")
                    continue

                # 處理圖片
                result_image, landmarks_dict = self.process_image(image)
                
                # 保存結果圖片
                output_file = output_path / f'pose_{image_file.name}'
                cv2.imwrite(str(output_file), result_image)
                
                # 保存關鍵點數據（可選）
                if landmarks_dict:
                    landmark_file = output_path / f'landmarks_{image_file.stem}.txt'
                    self.save_landmarks(landmarks_dict, landmark_file)
                
            except Exception as e:
                print(f'\n處理 {image_file.name} 時發生錯誤: {e}')

    def process_image(self, image):
        # 轉換顏色空間
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 獲取圖片尺寸
        image_height, image_width, _ = image.shape
        
        # 進行姿態估計
        results = self.pose.process(image_rgb)
        
        # 創建輸出圖片
        output_image = image.copy()
        landmarks_dict = None
        
        if results.pose_landmarks:
            # 繪製骨架
            self.mp_drawing.draw_landmarks(
                output_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0),  # 綠色關鍵點
                    thickness=2,
                    circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0),  # 藍色連接線
                    thickness=2
                )
            )
            
            # 提取並保存關鍵點數據
            landmarks_dict = {
                'landmarks': [],
                'image_size': {'width': image_width, 'height': image_height}
            }
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_dict['landmarks'].append({
                    'id': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            # 如果有分割結果，可以疊加顯示（可選）
            if results.segmentation_mask is not None:
                mask = results.segmentation_mask * 255
                mask = mask.astype(np.uint8)
                output_image = cv2.addWeighted(output_image, 0.9, 
                                             cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.1, 0)
        
        return output_image, landmarks_dict

    def save_landmarks(self, landmarks_dict, output_file):
        """保存關鍵點數據到文本文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Image size: {landmarks_dict['image_size']}\n\n")
            f.write("Landmarks:\n")
            for landmark in landmarks_dict['landmarks']:
                f.write(f"ID: {landmark['id']}\n")
                f.write(f"Position: x={landmark['x']:.4f}, y={landmark['y']:.4f}, z={landmark['z']:.4f}\n")
                f.write(f"Visibility: {landmark['visibility']:.4f}\n\n")

    def __del__(self):
        self.pose.close()

def main():
    try:
        # 檢查並安裝必要的包
        import subprocess
        import sys
        
        def install_package(package):
            print(f"安裝 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        # 檢查並安裝所需套件
        required_packages = ['mediapipe', 'tqdm']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                install_package(package)
        
        # 設置路徑
        input_dir = '/Users/joyfulsucessful/Desktop/AIGC/AIGC_test/woman-posing_512_8fps_8s'
        output_dir = '/Users/joyfulsucessful/Desktop/AIGC/AIGC_test/woman-posing_512_8fps_8s/output2'
        # output_dir = f'{input_dir}/output2'
        
        # 創建和使用 PoseEstimator
        print("初始化姿態估計器...")
        estimator = PoseEstimator()
        estimator.process_directory(input_dir, output_dir)
        print(f"\n處理完成！結果已保存到: {output_dir}")
        
    except Exception as e:
        print(f'\n發生錯誤: {str(e)}')

if __name__ == "__main__":
    main()