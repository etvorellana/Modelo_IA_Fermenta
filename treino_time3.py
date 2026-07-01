import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


#CSV_PATH = Path("report-file-1.csv")
CSV_PATH = Path("interpolated_N20_data.csv")
OUTPUT_DIR = Path("artifacts_time")
WINDOW_SIZE = 20
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

#Ok
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=r"\s+",
        skiprows=3,
        header=None,
        engine="python",
    )

    df.columns = [
        "TimeStep",
        "flow_time",
        "delta_time",
        "iters_per_timestep",
        "mon_x",
        "mon_sacarose",
        "mon_glicose",
        "mon_etanol",
    ]

    df = df.replace({"\"": "", r"\(": "", r"\)": ""}, regex=True)
    df = df[["flow_time", "mon_sacarose", "mon_glicose", "mon_etanol"]]
    df = df.astype(float)
    df = df.sort_values("flow_time").reset_index(drop=True)
    return df

#Ok
def create_samples(state_scaled, flow_scaled, window_size):
    X_hist, future_t, y = [], [], []

    for i in range(len(state_scaled) - window_size):
        target_idx = i + window_size
        history_states = state_scaled[i:target_idx]
        history_flow = flow_scaled[i:target_idx]

        history_features = np.concatenate(
            [history_states, history_flow],
            axis=1,
        )

        X_hist.append(history_features)
        future_t.append(flow_scaled[target_idx])
        y.append(state_scaled[target_idx])

    return np.array(X_hist), np.array(future_t), np.array(y)

#Ok
class TemporalForecastDataset(Dataset):
    def __init__(self, X_hist, future_t, y):
        self.X_hist = torch.FloatTensor(X_hist)
        self.future_t = torch.FloatTensor(future_t)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X_hist)

    def __getitem__(self, idx):
        return self.X_hist[idx], self.future_t[idx], self.y[idx]

#Ok
class TimeConditionedLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.1, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 16, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x_hist, future_t):
        _, (hn, _) = self.lstm(x_hist)
        context = hn[-1]
        time_context = self.time_encoder(future_t)
        combined = torch.cat([context, time_context], dim=1)
        return self.head(combined)


def inverse_state(state_scaler, values):
    return state_scaler.inverse_transform(values)


def forecast_future_steps(
    model,
    last_history_features,
    future_flow_scaled,
    state_scaler,
    flow_scaler,
    device,
):
    model.eval()
    history = last_history_features.copy()
    forecasts = []

    # Aceita entrada como array 1D (n,) ou 2D (n, 1).
    future_flow_scaled = np.asarray(future_flow_scaled).reshape(-1)

    with torch.no_grad():
        for next_flow_scaled in future_flow_scaled:
            hist_tensor = torch.FloatTensor(history[None, :, :]).to(device)
            flow_tensor = torch.FloatTensor([[next_flow_scaled]]).to(device)

            predicted_state_scaled = model(hist_tensor, flow_tensor).cpu().numpy()
            predicted_state = inverse_state(state_scaler, predicted_state_scaled)
            forecasts.append(predicted_state[0])

            # Próxima linha do histórico: estado previsto + flow_time futuro informado.
            next_row = np.array(
                [
                    predicted_state_scaled[0, 0],
                    predicted_state_scaled[0, 1],
                    predicted_state_scaled[0, 2],
                    next_flow_scaled,
                ]
            )
            history = np.vstack([history[1:], next_row])

    return np.array(forecasts)

 
def main():
    df = load_data(CSV_PATH)

    state_values = df[["mon_sacarose", "mon_glicose", "mon_etanol"]].values
    flow_values = df[["flow_time"]].values

    state_scaler = MinMaxScaler()
    flow_scaler = MinMaxScaler()

    state_scaled = state_scaler.fit_transform(state_values)
    flow_scaled = flow_scaler.fit_transform(flow_values)

    X_hist, future_t, y = create_samples(state_scaled, flow_scaled, WINDOW_SIZE)

    split_idx = int(0.8 * len(X_hist))
    print(f"Total samples: {len(X_hist)}, Train samples: {split_idx}, Test samples: {len(X_hist) - split_idx}")
    X_train, X_test = X_hist[:split_idx], X_hist[split_idx:]
    t_train, t_test = future_t[:split_idx], future_t[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    split_idx_val = int(0.8 * len(X_train))
    print(f"Train samples: {len(X_train)}, Train split: {split_idx_val}, Val samples: {len(X_train) - split_idx_val}")
    X_train, X_val = X_train[:split_idx_val], X_train[split_idx_val:]
    t_train, t_val = t_train[:split_idx_val], t_train[split_idx_val:]
    y_train, y_val = y_train[:split_idx_val], y_train[split_idx_val:]

    train_dataset = TemporalForecastDataset(X_train, t_train, y_train)
    val_dataset = TemporalForecastDataset(X_val, t_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TimeConditionedLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for X_batch, t_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            t_batch = t_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch, t_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, t_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                t_batch = t_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(X_batch, t_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        t_test_tensor = torch.FloatTensor(t_test).to(device)
        predictions = model(X_test_tensor, t_test_tensor).cpu().numpy()

    predictions_rescaled = inverse_state(state_scaler, predictions)
    y_test_rescaled = inverse_state(state_scaler, y_test)

    inference_mse = np.mean((predictions_rescaled - y_test_rescaled) ** 2, axis=1)
    results_df = pd.DataFrame(
        {
            "pred_mon_sacarose": predictions_rescaled[:, 0],
            "pred_mon_glicose": predictions_rescaled[:, 1],
            "pred_mon_etanol": predictions_rescaled[:, 2],
            "esperado_mon_sacarose": y_test_rescaled[:, 0],
            "esperado_mon_glicose": y_test_rescaled[:, 1],
            "esperado_mon_etanol": y_test_rescaled[:, 2],
            "mse": inference_mse,
        }
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / "modelo_temporal_3.pth"
    results_path = OUTPUT_DIR / "inferencias_temporais_3.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_scaler": state_scaler,
            "flow_scaler": flow_scaler,
            "window_size": WINDOW_SIZE,
            "input_size": 4,
            "hidden_size": 64,
            "num_layers": 2,
        },
        model_path,
    )
    results_df.to_csv(results_path, index=False)

    print(f"Modelo temporal salvo em: {model_path}")
    print(f"Inferências de teste salvas em: {results_path}")


if __name__ == "__main__":
    main()