# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?select=chest_xray
# 데이터 출처

from google.colab import drive
from google.colab import files
import os
import zipfile

# Google Drive 마운트
drive.mount('/content/drive')

# Google Drive 저장 경로 설정
drive_data_path = 'content/drive/MyDrive/Colab_Data'
os.makedirs(drive_data_path, exist_ok=True)

# 필요한 파일 업로드
print("train 파일을 업로드하세요")
uploaded_train_zip = files.upload()

print("val 파일을 엄로드하세요")
uploaded_val_zip = files.upload()

print("test 파일을 업로드하세요")
uploaded_test_zip = files.upload()

# 업로드된 파일을 Google Drive에 저장
for filename in uploaded_train_zip.keys():
    save_path = os.path.join(drive_data_path, filename)
    with open(save_path, 'wb') as f:
        f.write(uploaded_train_zip[filename])
    print(f"train.zip이 Google Drive에 저장되었습니다: {save_path}")

for filename in uploaded_val_zip.keys():
    save_path = os.path.join(drive_data_path, filename)
    with open(save_path, 'wb') as f:
        f.write(uploaded_val_zip[filename])
    print(f"val.zip이 Google Drive에 저장되었습니다: {save_path}")

for filename in uploaded_test_zip.keys():
    save_path = os.path.join(drive_data_path, filename)
    with open(save_path, 'wb') as f:
        f.write(uploaded_test_zip[filename])
    print(f"test.zip이 Google Drive에 저장되었습니다: {save_path}")

# 업로드한 zip 파일의 압축 해제 ( 위에서 Google Drive에 한 번 업로드한 뒤로는 이 코드부터 시행할 것 )
### from google.colab import drive
### from google.colab import files
### import os
### import zipfile

# Google Drive 마운트
drive.mount('/content/drive')

# Google Drive에 저장된 파일 경로
train_zip_path = '/content/drive/MyDrive/Colab_Data/train.zip'
val_zip_path = '/content/drive/MyDrive/Colab_Data/val.zip'
test_zip_path = '/content/drive/MyDrive/Coalb_Data/test.zip'

# 압축 해제 경로 설정
train_dir = '/content/train'
val_dir = '/content/val'
test_dir = '/content/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ZIP 파일 압축 해제
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(train_dir)
with zipfile.ZipFile(val_zip_path, 'r') as zip_ref:
    zip_ref.extractall(val_dir)
with zipfile_ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(test_dir)

print("Train, Validation, Test 데이터가 성공적으로 압축 해제되었습니다.")
print(f"Train 데이터 경로: {train_dir}")
print(f"Val 데이터 경로: {val_dir}")
print(f"Test 데이터 경로: {test_dir}")
