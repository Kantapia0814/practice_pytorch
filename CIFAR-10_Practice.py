import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 이미지를 화면에 보여주기 위한 함수
def imshow(img):
    img = img / 2 + 0.5  # 정규화를 해제 (unnormalize)
    npimg = img.numpy()  # 이미지를 numpy 배열로 변환
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 채널 축을 변환
    plt.show()

# 데이터 로딩 및 전처리 함수
def load_data(batch_size=4, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화: 평균 0.5, 표준편차 0.5로 정규화
    ])

    # 학습 데이터셋 로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 테스트 데이터셋 로드
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 검증 데이터셋 로드 (여기서는 테스트 데이터셋을 검증 용도로 재사용)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # CIFAR-10 데이터셋의 클래스들
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, valloader, classes

# 신경망 모델 정의
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, 5)  # 입력 채널 3, 출력 채널 6, 커널 크기 5인 첫 번째 합성곱 층
        self.pool = nn.MaxPool2d(3, 2, 1)  # 커널 크기 2, 스트라이드 2인 최대 풀링 층
        self.conv2 = nn.Conv2d(48, 128, 3, padding='same')  # 입력 채널 6, 출력 채널 16, 커널 크기 5인 두 번째 합성곱 층
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 192, 3, padding='same')
        self.conv4 = nn.Conv2d(192, 192, 3, padding='same')
        self.pool3 = nn.MaxPool2d(3, 2)
        self.conv5 = nn.Conv2d(192, 128, 1, padding='same')
        self.fc1 = nn.Sequential(
            nn.Linear(1152, 512),  # 첫 번째 완전 연결 층
            nn.ReLU(),
            nn.Dropout(0.5))  # 드롭아웃 레이어 추가
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),  # 두 번째 완전 연결 층
            nn.ReLU(),
            nn.Dropout(0.5))  # 드롭아웃 레이어 추가
        self.fc3 = nn.Linear(256, 10)  # 출력 층 (클래스 개수 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 첫 번째 합성곱 후 ReLU 활성화 함수, 풀링 적용
        x = self.pool2(F.relu(self.conv2(x)))  # 두 번째 합성곱 후 ReLU 활성화 함수, 풀링 적용
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = F.relu(self.conv5(x))

        x = torch.flatten(x, 1)  # 배치 차원 제외 모든 차원을 평탄화
        x = self.fc1(x)  # 첫 번째 완전 연결 층
        x = self.fc2(x)  # 두 번째 완전 연결 층
        x = self.fc3(x)  # 출력 층
        return x

# 모델 학습 함수
def train_model(net, trainloader, valloader, criterion, optimizer, scheduler, num_epochs=10, checkpoint_interval=1):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        net.train()  # 모델을 학습 모드로 설정
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()  # 기울기 초기화
            outputs = net(inputs)  # 모델 예측
            loss = criterion(outputs, labels)  # 손실 계산
            loss.backward()  # 역전파를 통해 기울기 계산
            optimizer.step()  # 가중치 업데이트
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)  # 예측된 클래스
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 0:  # 매 10 미니배치마다 손실 출력
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.6f}')
                running_loss = 0.0
        
        train_accuracy = 100 * correct / total
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(train_accuracy)
        print(f'Accuracy of the network on the train images after epoch {epoch + 1}: {train_accuracy:.2f}%')

        # 검증 데이터에 대한 평가
        net.eval()  # 모델을 평가 모드로 설정
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # 기울기 계산 중지
            for val_data in valloader:
                val_inputs, val_labels = val_data
                val_outputs = net(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss / len(valloader))
        val_accuracies.append(val_accuracy)
        print(f'Validation loss after epoch {epoch + 1}: {val_loss / len(valloader):.3f}')
        print(f'Accuracy of the network on the validation images after epoch {epoch + 1}: {val_accuracy:.2f}%')

        # 학습률 조정
        scheduler.step(val_loss / len(valloader))

        # 체크포인트 저장
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f'./checkpoint_epoch_{epoch + 1}.pth'
            torch.save(net.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch + 1}')

    print('Finished Training')
    return train_losses, val_losses, train_accuracies, val_accuracies

# 정확도 계산 함수
def calculate_accuracy(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():  # 기울기 계산 중지
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 메인 함수
def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로딩
    trainloader, testloader, valloader, classes = load_data(batch_size=32)
    
    # 모델 생성
    net = Net().to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()  # 교차 엔트로피 손실 함수
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGD 옵티마이저
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)  # 학습률 조정 스케줄러

    # 모델 학습
    num_epochs = 10
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(net, trainloader, valloader, criterion, optimizer, scheduler, num_epochs)

    # 학습 및 검증 손실 시각화
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend() # hi
    plt.title('Train and Validation Accuracy')
    
    plt.show()

    # 모델 저장
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # 학습 데이터에서 샘플 가져오기
    train_dataiter = iter(trainloader)
    train_images, train_labels = next(train_dataiter)

    # 테스트 데이터에서 샘플 가져오기
    test_dataiter = iter(testloader)
    test_images, test_labels = next(test_dataiter)

    # 학습 데이터 샘플 이미지 출력 및 라벨 출력
    imshow(torchvision.utils.make_grid(train_images))
    print('Train GroundTruth: ', ' '.join(f'{classes[train_labels[j]]:5s}' for j in range(4)))

    # 테스트 데이터 샘플 이미지 출력 및 라벨 출력
    imshow(torchvision.utils.make_grid(test_images))
    print('Test GroundTruth: ', ' '.join(f'{classes[test_labels[j]]:5s}' for j in range(4)))

    # 모델 로드
    net.load_state_dict(torch.load(PATH))

    # 학습 데이터 예측 수행
    train_outputs = net(train_images)
    _, train_predicted = torch.max(train_outputs, 1)
    print('Train Predicted: ', ' '.join(f'{classes[train_predicted[j]]:5s}' for j in range(4)))

    # 테스트 데이터 예측 수행
    test_outputs = net(test_images)
    _, test_predicted = torch.max(test_outputs, 1)
    print('Test Predicted: ', ' '.join(f'{classes[test_predicted[j]]:5s}' for j in range(4)))

    # 모델의 전체 정확도 계산
    train_accuracy = calculate_accuracy(net, trainloader)
    test_accuracy = calculate_accuracy(net, testloader)
    print(f'Accuracy of the network on the train images: {train_accuracy:.2f}%')
    print(f'Accuracy of the network on the test images: {test_accuracy:.2f}%')

if __name__ == '__main__':
    main()
