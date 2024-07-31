import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim  # 옵티마이저 모듈을 가져옵니다.

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # CUDA가 가능하면 GPU를, 아니면 CPU를 사용하도록 설정합니다.

def get_train_valid_loader(data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 데이터셋의 평균값
        std=[0.2023, 0.1994, 0.2010],  # CIFAR-10 데이터셋의 표준편차
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),  # 이미지 크기를 227x227로 조정
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        normalize,  # 정규화
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 32x32 크기로 랜덤 크롭, 패딩 4
            transforms.RandomHorizontalFlip(),  # 랜덤으로 수평 뒤집기
            transforms.Resize((227, 227)),  # 이미지 크기를 227x227로 조정
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            normalize,  # 정규화
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),  # 이미지 크기를 227x227로 조정
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            normalize,  # 정규화
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,  # 학습용 CIFAR-10 데이터셋 로드
        download=True, transform=train_transform,  # 위에서 정의한 변환을 적용
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,  # 검증용 CIFAR-10 데이터셋 로드
        download=True, transform=valid_transform,  # 위에서 정의한 변환을 적용
    )

    num_train = len(train_dataset)  # 학습 데이터셋의 크기
    indices = list(range(num_train))  # 인덱스 리스트 생성
    split = int(np.floor(valid_size * num_train))  # 검증 데이터셋 크기 계산

    if shuffle:
        np.random.seed(random_seed)  # 난수 시드 설정
        np.random.shuffle(indices)  # 인덱스 섞기

    train_idx, valid_idx = indices[split:], indices[:split]  # 학습 인덱스와 검증 인덱스로 나누기
    train_sampler = SubsetRandomSampler(train_idx)  # 학습 샘플러
    valid_sampler = SubsetRandomSampler(valid_idx)  # 검증 샘플러

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)  # 학습 데이터 로더 생성
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)  # 검증 데이터 로더 생성

    return train_loader, valid_loader  # 학습 및 검증 데이터 로더 반환

def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # 일반적으로 사용하는 이미지넷 데이터셋의 평균값
        std=[0.229, 0.224, 0.225],  # 일반적으로 사용하는 이미지넷 데이터셋의 표준편차
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # 이미지 크기를 227x227로 조정
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        normalize,  # 정규화
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,  # 테스트용 CIFAR-10 데이터셋 로드
        download=True, transform=transform,  # 위에서 정의한 변환을 적용
    )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # 테스트 데이터 로더 생성

    return data_loader  # 테스트 데이터 로더 반환

# CIFAR10 dataset
train_loader, valid_loader = get_train_valid_loader(data_dir='./data', batch_size=64, augment=False, random_seed=1)  # 학습 및 검증 데이터 로더 생성
test_loader = get_test_loader(data_dir='./data', batch_size=64)  # 테스트 데이터 로더 생성

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),  # 11x11 커널, 스트라이드 4, 패딩 없음
            nn.BatchNorm2d(96),  # 배치 정규화
            nn.ReLU(),  # ReLU 활성화 함수
            nn.MaxPool2d(kernel_size=3, stride=2))  # 3x3 최대 풀링, 스트라이드 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # 5x5 커널, 스트라이드 1, 패딩 2
            nn.BatchNorm2d(256),  # 배치 정규화
            nn.ReLU(),  # ReLU 활성화 함수
            nn.MaxPool2d(kernel_size=3, stride=2))  # 3x3 최대 풀링, 스트라이드 2
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # 3x3 커널, 스트라이드 1, 패딩 1
            nn.BatchNorm2d(384),  # 배치 정규화
            nn.ReLU())  # ReLU 활성화 함수
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # 3x3 커널, 스트라이드 1, 패딩 1
            nn.BatchNorm2d(384),  # 배치 정규화
            nn.ReLU())  # ReLU 활성화 함수
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 3x3 커널, 스트라이드 1, 패딩 1
            nn.BatchNorm2d(256),  # 배치 정규화
            nn.ReLU(),  # ReLU 활성화 함수
            nn.MaxPool2d(kernel_size=3, stride=2))  # 3x3 최대 풀링, 스트라이드 2
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # 드롭아웃 확률 0.5
            nn.Linear(256 * 6 * 6, 4096),  # 완전 연결 층
            nn.ReLU())  # ReLU 활성화 함수
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),  # 드롭아웃 확률 0.5
            nn.Linear(4096, 4096),  # 완전 연결 층
            nn.ReLU())  # ReLU 활성화 함수
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))  # 출력층, 클래스 개수만큼의 뉴런

    def forward(self, x):
        out = self.layer1(x)  # 첫 번째 합성곱 층
        out = self.layer2(out)  # 두 번째 합성곱 층
        out = self.layer3(out)  # 세 번째 합성곱 층
        out = self.layer4(out)  # 네 번째 합성곱 층
        out = self.layer5(out)  # 다섯 번째 합성곱 층
        out = out.reshape(out.size(0), -1)  # 평탄화
        out = self.fc(out)  # 첫 번째 완전 연결 층
        out = self.fc1(out)  # 두 번째 완전 연결 층
        out = self.fc2(out)  # 출력 층
        return out

# 모델, 손실 함수, 옵티마이저 정의
model = AlexNet(num_classes=10).to(device)  # 모델을 생성하고, GPU/CPU 설정
criterion = nn.CrossEntropyLoss()  # 교차 엔트로피 손실 함수 정의
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD 옵티마이저 정의

# 학습 파라미터 설정
num_epochs = 10  # 총 학습 에포크 수
total_step = len(train_loader)  # 학습 단계 수

# 모델 학습
for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)  # 입력 이미지 GPU/CPU로 이동
        labels = labels.to(device)  # 레이블 GPU/CPU로 이동
        
        # Forward pass
        outputs = model(images)  # 모델 예측
        loss = criterion(outputs, labels)  # 손실 계산
        
        # Backward and optimize
        optimizer.zero_grad()  # 옵티마이저의 기울기 초기화
        loss.backward()  # 역전파를 통해 기울기 계산
        optimizer.step()  # 옵티마이저를 통해 가중치 업데이트

        running_loss += loss.item()  # 손실 누적
        
        if (i + 1) % 100 == 0:  # 매 100 스텝마다 손실 출력
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

# 모델 평가
model.eval()  # 모델을 평가 모드로 설정
correct = 0
total = 0
with torch.no_grad():  # 기울기 계산 중지
    for images, labels in test_loader:
        images = images.to(device)  # 입력 이미지 GPU/CPU로 이동
        labels = labels.to(device)  # 레이블 GPU/CPU로 이동
        outputs = model(images)  # 모델 예측
        _, predicted = torch.max(outputs.data, 1)  # 예측된 클래스
        total += labels.size(0)  # 총 샘플 수
        correct += (predicted == labels).sum().item()  # 정확히 예측된 샘플 수

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')  # 테스트 데이터셋의 정확도 출력
