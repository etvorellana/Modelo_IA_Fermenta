import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


CSV_PATH = Path("report-file-1.csv")
OUTPUT_DIR = Path("artifacts_time")
WINDOW_SIZE = 20
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001


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
    df = df[["flow_time", "delta_time", "mon_sacarose", "mon_glicose", "mon_etanol"]]
    df = df.astype(float)
    df = df.sort_values("flow_time").reset_index(drop=True)
    return df


def create_samples(state_scaled, flow_scaled, delta_scaled, window_size):
    X_hist, future_dt, y = [], [], []

    for i in range(len(state_scaled) - window_size):
        target_idx = i + window_size
        history_states = state_scaled[i:target_idx]
        history_flow = flow_scaled[i:target_idx]
        history_delta = delta_scaled[i:target_idx]

        history_features = np.concatenate(
            [history_states, history_flow, history_delta],
            axis=1,
        )

        X_hist.append(history_features)
        future_dt.append(delta_scaled[target_idx])
        y.append(state_scaled[target_idx])

    return np.array(X_hist), np.array(future_dt), np.array(y)


class TemporalForecastDataset(Dataset):
    def __init__(self, X_hist, future_dt, y):
        self.X_hist = torch.FloatTensor(X_hist)
        self.future_dt = torch.FloatTensor(future_dt)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X_hist)

    def __getitem__(self, idx):
        return self.X_hist[idx], self.future_dt[idx], self.y[idx]


class TimeConditionedLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.1, output_size=3):
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

    def forward(self, x_hist, future_dt):
        _, (hn, _) = self.lstm(x_hist)
        context = hn[-1]
        time_context = self.time_encoder(future_dt)
        combined = torch.cat([context, time_context], dim=1)
        return self.head(combined)


def inverse_state(state_scaler, values):
    return state_scaler.inverse_transform(values)


def forecast_future_steps(
    model,
    last_history_features,
    future_deltas_scaled,
    state_scaler,
    flow_scaler,
    delta_scaler,
    device,
):
    model.eval()
    history = last_history_features.copy()
    forecasts = []

    with torch.no_grad():
        for future_dt_scaled in future_deltas_scaled:
            hist_tensor = torch.FloatTensor(history[None, :, :]).to(device)
            dt_tensor = torch.FloatTensor([[future_dt_scaled]]).to(device)
            predicted_state_scaled = model(hist_tensor, dt_tensor).cpu().numpy()
            predicted_state = inverse_state(state_scaler, predicted_state_scaled)
            forecasts.append(predicted_state[0])

            last_row = history[-1].copy()
            last_flow_scaled = last_row[3]
            last_delta_scaled = last_row[4]

            last_flow = flow_scaler.inverse_transform([[last_flow_scaled]])[0, 0]
            future_dt = delta_scaler.inverse_transform([[future_dt_scaled]])[0, 0]
            next_flow = last_flow + future_dt

            next_flow_scaled = flow_scaler.transform([[next_flow]])[0, 0]
            next_row = np.array(
                [
                    predicted_state_scaled[0, 0],
                    predicted_state_scaled[0, 1],
                    predicted_state_scaled[0, 2],
                    next_flow_scaled,
                    future_dt_scaled,
                ]
            )

            history = np.vstack([history[1:], next_row])

    return np.array(forecasts)


def main():
    df = load_data(CSV_PATH)

    state_values = df[["mon_sacarose", "mon_glicose", "mon_etanol"]].values
    flow_values = df[["flow_time"]].values
    delta_values = df[["delta_time"]].values

    state_scaler = MinMaxScaler()
    flow_scaler = MinMaxScaler()
    delta_scaler = MinMaxScaler()

    state_scaled = state_scaler.fit_transform(state_values)
    flow_scaled = flow_scaler.fit_transform(flow_values)
    delta_scaled = delta_scaler.fit_transform(delta_values)

    X_hist, future_dt, y = create_samples(state_scaled, flow_scaled, delta_scaled, WINDOW_SIZE)

    split_idx = int(0.8 * len(X_hist))
    X_train, X_test = X_hist[:split_idx], X_hist[split_idx:]
    dt_train, dt_test = future_dt[:split_idx], future_dt[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    split_idx_val = int(0.8 * len(X_train))
    X_train, X_val = X_train[:split_idx_val], X_train[split_idx_val:]
    dt_train, dt_val = dt_train[:split_idx_val], dt_train[split_idx_val:]
    y_train, y_val = y_train[:split_idx_val], y_train[split_idx_val:]

    train_dataset = TemporalForecastDataset(X_train, dt_train, y_train)
    val_dataset = TemporalForecastDataset(X_val, dt_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TimeConditionedLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for X_batch, dt_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            dt_batch = dt_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch, dt_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, dt_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                dt_batch = dt_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(X_batch, dt_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        dt_test_tensor = torch.FloatTensor(dt_test).to(device)
        predictions = model(X_test_tensor, dt_test_tensor).cpu().numpy()

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
    model_path = OUTPUT_DIR / "modelo_temporal.pth"
    results_path = OUTPUT_DIR / "inferencias_temporais.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_scaler": state_scaler,
            "flow_scaler": flow_scaler,
            "delta_scaler": delta_scaler,
            "window_size": WINDOW_SIZE,
            "input_size": 5,
            "hidden_size": 64,
            "num_layers": 2,
        },
        model_path,
    )
    results_df.to_csv(results_path, index=False)

    print(f"Modelo temporal salvo em: {model_path}")
    print(f"Inferências de teste salvas em: {results_path}")

    # Exemplo de uso para passos futuros: use os deltas reais ou escolhidos pelo usuário.
    future_deltas_raw = df["delta_time"].tail(5).values.reshape(-1, 1)
    future_deltas_scaled = delta_scaler.transform(future_deltas_raw).reshape(-1)
    last_history = X_hist[-1]
    future_forecasts = forecast_future_steps(
        model,
        last_history,
        future_deltas_scaled,
        state_scaler,
        flow_scaler,
        delta_scaler,
        device,
    )
    forecast_df = pd.DataFrame(
        future_forecasts,
        columns=["mon_sacarose", "mon_glicose", "mon_etanol"],
    )
    forecast_df.to_csv(OUTPUT_DIR / "forecast_passos_futuros.csv", index=False)
    print(f"Forecast futuro salvo em: {OUTPUT_DIR / 'forecast_passos_futuros.csv'}")


if __name__ == "__main__":
    main()