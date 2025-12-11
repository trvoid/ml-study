import torch
import torch.nn as nn

# 1. 간단한 모델과 데이터
model = nn.Linear(2, 1)  # 파라미터 W와 b가 포함됨
input_data = torch.randn(1, 2)
target = torch.randn(1, 1)

# 2. 순전파 (Forward)
pred = model(input_data)
loss = (pred - target).mean()

# 3. 연결 고리 확인 (재귀적 탐색)
print(f"Loss의 생성자: {loss.grad_fn}") 

def print_grad_fn_chain(grad_fn, indent=0):
    print(" " * indent + str(grad_fn))
    if hasattr(grad_fn, 'variable'):
        print(" " * (indent + 2) + f"Accumulates gradient for tensor with shape: {grad_fn.variable.shape}")
    
    for next_fn, _ in grad_fn.next_functions:
        if next_fn is not None:
            print_grad_fn_chain(next_fn, indent + 2)

print("\n--- grad_fn Chain ---")
print_grad_fn_chain(loss.grad_fn)