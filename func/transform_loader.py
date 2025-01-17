from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch

class ChestXRayDataset(Dataset):
    def __init__(self, image_dir, sequence_length = 1, transform = None):
        self.image_dir = image_dir
        self.sequence_length = sequence_length    # 일반적으로 학습에 1장의 이미지만 사용됨
        self.transform = transform

        # 이미지 파일 정렬 및 레이블 생성
        self.image_files = []
        self.labels = []                          # 0 : Normal | 1 : Pneumonia

        # 폴더 구조에 따라 이미지 파일과 라벨을 로드
        for label, category in enumerate(["NORMAL", "PNEUMONIA"]):
            category_path = os.path.join(image_dir, category)
            image_list = sorted([f for f in os.listdir(category_path) if f.endswith(".jpeg")])
            self.image_files += [os.path.join(category_path, img) for img in image_list]
            self.labels += [label] * len(image_list)

        if len(self.image_files) == 0:
            raise ValueError("No images found in the provided directory")

    def __len__(self):
        return len(self.image_files) - self.sequence_length + 1

    def __getitem__(self, idx):
        image_sequence = []
        for i in range(self.sequence_length):
            img_path = self.image_files[idx + i]
            img = Image.open(img_path).convert("L")           # 흑백 이미지 유지
            if self.transform:
                img = self.transform(img)
            image_sequence.append(img)

        # 단일 이미지라면 시퀀스를 풀어서 반환
        images = torch.stack(image_sequence) if self.sequence_length > 1 else image_sequence[0]
        label = torch.tensor(self.labels[idx + self.sequence_length - 1], dtype = torch.long)   # 0 or 1

        return images, label


### Test Code
### 각 폴더에 적합한 이미지 개수가 저장되어 있는지 확인

###            <출력되어야 하는 형식>                 ###
###             --- TRAIN DATA ---                    ###
###             NORMAL: 1341 images                   ###
###             PNEUMONIA: 3875 images                ###
###             Total in train: 5216 images           ###
###
###             --- VAL DATA ---                      ###
###             NORMAL: 8 images                      ###
###             PNEUMONIA: 8 images                   ###
###             Total in val: 16 images               ###
###
###             --- TEST DATA ---                     ###
###             NORMAL: 234 images                    ###
###             PNEUMONIA: 390 images                 ###
###             Total in test: 624 images             ###


import os

def count_images_in_folders(base_dir):
    categories = ["NORMAL", "PNEUMONIA"]
    folders = ["train", "val", "test"]

    for folder in folders:
        print(f"\n--- {folder.upper()} DATA ---")
        total_images = 0
        for category in categories:
            category_path = os.path.join(base_dir, folder, category)
            image_files = [f for f in os.listdir(category_path) if f.endswith('.jpeg')]
            num_images = len(image_files)
            total_images += num_images
            print(f"{category}: {num_images} images")
        print(f"Total in {folder}: {total_images} images")

# Base directory (Coalb 기준)
base_dir = '/content'       # train, val, test 폴더가 이 경로 기반 존재함
count_images_in_folders(base_dir)



# Train을 위한 transform 설정
transform_for_train = transforms.Compose([
    transforms.Resize((400, 121)),
    transforms.RandomHorizontalFlip(p = 0.5),                 # 좌우 반전 (50% 확률)
    transforms.RandomRotation(15),                            # -15 ~ 15도 범위 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2),     # 밝기 및 대비 변화
    transforms.ToTensor(),                                    # Tensor 변환
    transforms.Normalize(mean=[0.5], std=[0.5])               # 정규화
])

# Validation을 위한 transform 설정
transform_for_val = transforms.Compose([
    transforms.Resize((400, 121)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5], std=[0.5])
])

# Test를 위한 transform 설정
transform_for_test = transforms.Compose([
    transforms.Resize((400, 121)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5], std=[0.5])
])



# Data Directory path
train_dir = '/content/train/train'
val_dir = '/content/val/val'
test_dir = '/content/test/test'

# ChestXRayDataset 인스턴스 생성
train_dataset = ChestXRayDataset(train_dir, transform = transform_for_train)
val_dataset = ChestXRayDataset(val_dir, transform = transform_for_val)
test_dataset = ChestXRayDataset(test_dir, transform = transform_for_test)

# Batch Size 설정
batch_size_train = 16
batch_size_val = 16
batch_size_test = 16

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size = batch_size_train, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size_val, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size_test, shuffle = False)


### Test Data Loader

print(f"Train 데이터 수: {len(train_loader.dataset)}")
print(f"Validation 데이터 수: {len(val_loader.dataset)}")
print(f"Test 데이터 수: {len(test_loader.dataset)}")

images, labels = nex(iter(val_loader))
print(f"Validation Images Shape; {images.shape}")             # (batch_size, 1, 400, 121)
print(f"Validation Labels: {labels}")                         # Tensor 형태의 레이블 출력
