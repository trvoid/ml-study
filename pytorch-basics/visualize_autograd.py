import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchviz import make_dot

# 2. 간단한 신경망 모델 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 입력(28x28=784) -> 은닉층(128) -> 출력(10: 0~9까지의 숫자)
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 입력 데이터 펼치기 (Flatten)
        x = x.view(-1, 784)
        
        # [Autograd 매커니즘]
        # 이 연산들이 수행될 때 PyTorch는 내부적으로 '계산 그래프'를 동적으로 생성합니다.
        # 각 텐서는 자신이 어떤 연산(Function)을 통해 만들어졌는지 기억합니다 (grad_fn).
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 훈련 함수
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train() # 모델을 훈련 모드로 설정
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # --- [Step A] 기울기 초기화 ---
        # 이전 배치의 그래디언트가 누적되지 않도록 0으로 초기화
        optimizer.zero_grad()

        # --- [Step B] 순전파 (Forward Pass) ---
        # 입력 데이터를 모델에 통과시켜 예측값(output) 계산
        # 이 과정에서 연산 그래프(Graph)가 구축됨
        output = model(data)

        # --- [Step C] 손실 계산 (Loss Calculation) ---
        # 예측값과 실제값(target) 사이의 오차 계산
        # loss는 그래프의 최종 노드(Leaf Node)가 됨
        loss = criterion(output, target)

        # --- [Step D] 역전파 (Backward Pass - Autograd의 핵심) ---
        # loss.backward() 호출 시 그래프를 역순으로 탐색하며 연쇄 법칙(Chain Rule) 적용
        # 각 파라미터(W, b)들의 .grad 속성에 미분값(기울기)이 저장됨
        loss.backward()

        # --- [Step E] 파라미터 갱신 (Optimizer Step) ---
        # 계산된 기울기(gradient)를 사용하여 모델의 가중치를 수정
        # W_new = W_old - learning_rate * gradient
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return loss

def main():
    # 1. 데이터셋 준비 (MNIST)
    # 텐서 변환 및 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 학습용 데이터 로드
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    

    # 모델 및 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)

    # 3. 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss() # 분류 문제를 위한 손실 함수
    optimizer = optim.SGD(model.parameters(), lr=0.01) # 확률적 경사 하강법

    print(f"Training on {device}")
    for epoch in range(1, 2):
        loss = train(model, device, train_loader, criterion, optimizer, epoch)
        
        # --- 시각화 생성 (make_dot) ---
        # loss 객체를 루트로 하여 그래프를 그림
        # params 딕셔너리를 넘겨주면 그래프의 끝(Leaf)에 파라미터 이름 표시
        dot = make_dot(loss, params=dict(model.named_parameters()))

        # 이미지로 저장 및 출력
        dot.format = 'png'
        dot.render("computation_graph")

# 실행
if __name__ == '__main__':
    main()