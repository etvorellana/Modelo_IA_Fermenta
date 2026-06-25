import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch

from treino_time import (
    CSV_PATH,
    OUTPUT_DIR,
    WINDOW_SIZE,
    TimeConditionedLSTM,
    create_samples,
    forecast_future_steps,
    load_data,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Carrega o modelo temporal treinado e gera previsoes ate um tempo final com passo temporal fixo."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUT_DIR / "modelo_temporal.pth",
        help="Caminho do checkpoint .pth gerado pelo treino temporal.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=CSV_PATH,
        help="CSV original usado no treino.",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        required=True,
        help="Tempo final absoluto (flow_time) para encerrar as previsoes.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        required=True,
        help="Passo temporal fixo (delta_time) usado entre inferencias consecutivas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "previsoes_futuras_usuario.csv",
        help="Arquivo CSV de saida com as previsoes futuras.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def build_last_history(df, state_scaler, flow_scaler, delta_scaler, window_size):
    state_values = df[["mon_sacarose", "mon_glicose", "mon_etanol"]].values
    flow_values = df[["flow_time"]].values
    delta_values = df[["delta_time"]].values

    state_scaled = state_scaler.transform(state_values)
    flow_scaled = flow_scaler.transform(flow_values)
    delta_scaled = delta_scaler.transform(delta_values)

    X_hist, _, _ = create_samples(state_scaled, flow_scaled, delta_scaled, window_size)
    return X_hist[-1]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, device)
    state_scaler = checkpoint["state_scaler"]
    flow_scaler = checkpoint["flow_scaler"]
    delta_scaler = checkpoint["delta_scaler"]
    window_size = checkpoint.get("window_size", WINDOW_SIZE)

    df = load_data(args.dataset)

    last_observed_time = float(df["flow_time"].iloc[-1])
    if args.time_step <= 0:
        raise ValueError("O passo temporal deve ser positivo.")
    if args.final_time <= last_observed_time:
        raise ValueError("O tempo final deve ser maior que o ultimo flow_time observado.")

    print(f"Ultimo flow_time observado: {last_observed_time:.4f} segundos")
    delta_t_total = args.final_time - last_observed_time
    print(f"Delta_t total para previsao: {delta_t_total:.4f} segundos")
    num_inferences_float = delta_t_total / args.time_step
    print(f"Quantidade de inferencias calculada (float): {num_inferences_float:.4f}")   
    
    # Fixed: Use floor instead of round to avoid exceeding final_time
    num_inferences = int(math.floor(num_inferences_float))
    print(f"Quantidade de inferencias calculada (int): {num_inferences}")
    
    # Adjust final time to be multiple of time_step
    args.final_time = last_observed_time + num_inferences * args.time_step
    print(f"Tempo final ajustado para ser multiplo do passo: {args.final_time:.4f} segundos")

    if num_inferences <= 0:
        raise ValueError("A configuracao informada resulta em zero inferencias.")
    
    # Check if adjusted final time is close to requested
    #if abs(args.final_time - float(args._get_kwargs()[0][1])) > 1e-6:
    #    print(f"Nota: O tempo final foi ajustado de {args._get_kwargs()[0][1]:.4f} para {args.final_time:.4f}")

    future_deltas_raw = np.full((num_inferences, 1), args.time_step, dtype=float)
    future_flow_times = last_observed_time + np.cumsum(future_deltas_raw.flatten())

    future_deltas_scaled = delta_scaler.transform(future_deltas_raw).reshape(-1)
    last_history = build_last_history(df, state_scaler, flow_scaler, delta_scaler, window_size)

    # Fixed: Use correct num_layers from checkpoint (2, not 12)
    model = TimeConditionedLSTM(
        input_size=checkpoint.get("input_size", 5),
        hidden_size=checkpoint.get("hidden_size", 64),
        num_layers=checkpoint.get("num_layers", 2),  # Fixed: Use 2 as default, not 12
        output_size=3,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    predictions = forecast_future_steps(
        model,
        last_history,
        future_deltas_scaled,
        state_scaler,
        flow_scaler,
        delta_scaler,
        device,
    )

    results_df = pd.DataFrame(
        {
            "flow_time": future_flow_times,
            "delta_time": future_deltas_raw.flatten(),
            "pred_mon_sacarose": predictions[:, 0],
            "pred_mon_glicose": predictions[:, 1],
            "pred_mon_etanol": predictions[:, 2],
        }
    )

    args.output.parent.mkdir(exist_ok=True)
    results_df.to_csv(args.output, index=False)

    print(f"Checkpoint carregado de: {args.checkpoint}")
    print(f"Previsoes salvas em: {args.output}")
    print(f"Quantidade de inferencias: {num_inferences}")
    print(results_df.head(10))


if __name__ == "__main__":
    main()
