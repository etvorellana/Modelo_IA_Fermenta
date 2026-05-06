import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Geração de dataset artificial multivariado
N = 301  # número de amostras
window_size = 20
t = np.linspace(0, 300, N)
x = np.sin(2 * np.pi * t / 10) + 0.1 * np.random.randn(N)
y = np.sin(2 * np.pi * t / 20) + 0.1 * np.random.randn(N)
z = np.cos(2 * np.pi * t / 15) + 0.1 * np.random.randn(N)
data = np.column_stack([x, y, z])

# Normalização
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Cria janelas deslizantes
def create_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

X, y = create_windows(data_scaled, window_size)

# Divide em treino e teste e depois o treino em treino e validação (60,20,20)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

split_idx_val = int(0.8 * len(X_train))
X_train, X_val = X_train[:split_idx_val], X_train[split_idx_val:]
y_train, y_val = y_train[:split_idx_val], y_train[split_idx_val:]

# Classe Dataset personalizada
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Modelo MultivariateLSTM
class MultivariateLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=1, output_size=3):
        super(MultivariateLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out

model = MultivariateLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Treinamento
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)
    # Validação
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}  ')

# Exemplo de predição com o grupo de teste
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    predictions = model(X_test_tensor).cpu().numpy()
    # Inverter a normalização para comparar com os valores originais
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test)
    print("Predições (rescaladas):", predictions_rescaled[:5])
    print("Valores reais (rescalados):", y_test_rescaled[:5])      

