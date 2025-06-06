import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 定义Teacher模型（较大）
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# 定义Student模型（较小）
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # 通道数减半
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(4608, 64)       # 参数量更小
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# 训练Teacher模型
def train_teacher(model, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Teacher Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 测试函数
def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc

# 蒸馏训练Student模型
def train_student_with_distillation(student, teacher, optimizer, criterion_ce, T=5, alpha=0.7, epochs=5):
    student.train()
    teacher.eval()  # Teacher不参与参数更新
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Teacher输出
            with torch.no_grad():
                teacher_logits = teacher(data)

            # Student输出
            student_logits = student(data)

            # 计算蒸馏损失
            loss_ce = criterion_ce(student_logits, target)  # 交叉熵损失（硬标签）
            # 软标签（温度缩放softmax）
            p_teacher = F.log_softmax(teacher_logits / T, dim=1)
            p_student = F.softmax(student_logits / T, dim=1)
            loss_kd = F.kl_div(p_teacher, p_student, reduction='batchmean') * (T * T)

            loss = alpha * loss_ce + (1 - alpha) * loss_kd
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Student Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 实例化模型和优化器
teacher = TeacherNet().to(device)
student = StudentNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=1e-3)
optimizer_student = torch.optim.Adam(student.parameters(), lr=1e-3)

# 1. 先训练Teacher模型
print("Training Teacher Model...")
train_teacher(teacher, optimizer_teacher, criterion, epochs=5)
print("Teacher Model Evaluation:")
test(teacher)

# 2. 使用蒸馏方法训练Student模型
print("Training Student Model with Distillation...")
train_student_with_distillation(student, teacher, optimizer_student, criterion, T=5, alpha=0.7, epochs=5)
print("Student Model Evaluation:")
test(student)

# 3. 可选：训练Student模型（无蒸馏）作对比
def train_student_plain(student, optimizer, criterion, epochs=5):
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = student(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Student Plain Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

print("Training Student Model without Distillation...")
student_plain = StudentNet().to(device)
optimizer_plain = torch.optim.Adam(student_plain.parameters(), lr=1e-3)
train_student_plain(student_plain, optimizer_plain, criterion, epochs=5)
print("Student Plain Model Evaluation:")
test(student_plain)
