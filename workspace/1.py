import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 画像フォルダのパスとクラス数
data_dir = "/Users/s12810162/Documents/investigation/AI/plant/BotaniClassify/train"
num_classes = 13
img_size = 100

# 画像データとラベルをロードする関数
def load_data():
    images = []
    labels = []
    for i in range(num_classes):
        class_dir = os.path.join(data_dir, f"class_{i:02d}")
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (img_size, img_size))
            images.append(image)
            labels.append(i)
    return np.array(images), np.array(labels)

# データのロード
images, labels = load_data()

# データを正規化
images = images.astype('float32') / 255.0

# ラベルをone-hotエンコーディング
labels = to_categorical(labels, num_classes)

# データをトレーニングセットとテストセットに分割
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# モデルの構築
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# モデルのトレーニング
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# モデルの評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
