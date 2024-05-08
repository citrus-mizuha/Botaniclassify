import os
import cv2
import numpy as np

def load_and_preprocess_images(directory, target_size=(224, 224)):
    images = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            image = cv2.resize(image, target_size)
            image = image / 255.0  # 正規化
            images.append(image)
    
    return np.array(images)

def main():
    base_directory = "/Users/s12810162/Documents/investigation/AI/plant/dataset"  # データセットのルートディレクトリを指定してください
    num_classes = 10
    images_per_class = 70
    target_size = (224, 224)
    
    for class_index in range(num_classes):
        class_directory = os.path.join(base_directory, f"class_{class_index:02d}")
        class_images = load_and_preprocess_images(class_directory, target_size)
        
        # 処理した画像の形状を確認
        print(f"Class {class_index}: {class_images.shape}")
        
        # ここでclass_imagesをモデルに入力するなどの処理を行う
        
if __name__ == "__main__":
    main()
