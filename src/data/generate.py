"""
Traffic data generation and preprocessing utilities.
Generates synthetic but realistic traffic data for demonstration.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_traffic_data(
    n_days: int = 365,
    junctions: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic hourly traffic volume data with realistic patterns.

    Parameters
    ----------
    n_days : int
        Number of days to simulate.
    junctions : int
        Number of traffic junctions.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Hourly traffic data with weather and event features.
    """
    rng = np.random.default_rng(seed)
    records = []

    start = datetime(2023, 1, 1)

    for junction in range(1, junctions + 1):
        for day in range(n_days):
            dt = start + timedelta(days=day)
            is_weekend = dt.weekday() >= 5
            is_event = rng.random() < 0.05  # 5% chance of event day

            # Weather
            base_temp = 22 + 8 * np.sin(2 * np.pi * day / 365)
            temperature = base_temp + rng.normal(0, 3)
            humidity = np.clip(60 + rng.normal(0, 15), 20, 100)
            wind = np.abs(rng.normal(10, 5))
            precipitation = max(0, rng.normal(-2, 5))  # mostly 0

            for hour in range(24):
                # Traffic pattern: bimodal peaks at 8am and 5pm
                morning_peak = np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
                evening_peak = np.exp(-0.5 * ((hour - 17) / 1.5) ** 2)
                base_volume = 20 * (morning_peak + 0.9 * evening_peak)

                # Junction multipliers
                junction_factor = 1.0 + 0.3 * (junction - 1)

                # Weekend reduction
                weekend_factor = 0.65 if is_weekend else 1.0

                # Event boost
                event_factor = 1.4 if is_event else 1.0

                # Weather impact
                weather_factor = 1.0 - 0.3 * (precipitation / 30)

                # Night minimum traffic
                night_base = 3 if 0 <= hour < 5 else 0

                volume = (
                    base_volume * junction_factor * weekend_factor
                    * event_factor * weather_factor
                    + night_base
                    + rng.normal(0, 1)
                )
                volume = max(0, round(volume, 2))

                records.append({
                    "datetime": dt + timedelta(hours=hour),
                    "junction": junction,
                    "vehicles": volume,
                    "temperature": round(temperature, 1),
                    "humidity": round(humidity, 1),
                    "wind": round(wind, 1),
                    "precipitation": round(precipitation, 2),
                    "is_event": int(is_event),
                })

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["junction", "datetime"]).reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based columns used throughout the app."""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["date"] = df["datetime"].dt.date
    return df
