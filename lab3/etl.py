import numpy as np
import pandas as pd

# df = pd.read_csv("cleaned_weather.csv")
#
# df["date"] = pd.to_datetime(df["date"])
#
# df["time_slot"] = df["date"].dt.strftime("%Y-%m-%d") + "_" + (df["date"].dt.hour // 6 * 6).astype(str).str.zfill(2) + ":00"
#
# valid_times = ["00:00", "06:00", "12:00", "18:00"]
# df_filtered = df[df["time_slot"].str[-5:].isin(valid_times)]
#
# df_grouped = df_filtered.groupby("time_slot")["p"].mean().reset_index()
#
# df_grouped.columns = ["date", "p"]
#
# df_grouped.to_csv("output.csv", index=False)


df = pd.read_csv('output.csv')

df['datetime'] = pd.to_datetime(df['date'], format='%Y-%m-%d_%H:%M')

df['hour'] = df['datetime'].dt.hour
df['daily_pressure_wave'] = np.cos(2 * np.pi * df['hour'] / 24)

df['day_of_year'] = df['datetime'].dt.dayofyear
df['seasonal_pressure_factor'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

df = df[['date', 'p', 'daily_pressure_wave', 'seasonal_pressure_factor']]

df.to_csv('output_with_features.csv', index=False)