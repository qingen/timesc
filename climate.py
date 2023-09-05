# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import paddlets
from paddlets import TSDataset
from paddlets import TimeSeries
from paddlets.models.forecasting import MLPRegressor, LSTNetRegressor
from paddlets.transform import Fill, StandardScaler
from paddlets.metrics import MSE, MAE


#dfAlt = pd.read_csv("./data/0614_150_s.csv",header=0, sep=',',encoding='gbk')
csv_path = "./data/jena_climate_2009_2016.csv"
df = pd.read_csv(csv_path)
print(df.head())

titles = [
    "Pressure",
    "Temperature",
    "Temperature in Kelvin",
    "Temperature (dew point)",
    "Relative Humidity",
    "Saturation vapor pressure",
    "Vapor pressure",
    "Vapor pressure deficit",
    "Specific humidity",
    "Water vapor concentration",
    "Airtight",
    "Wind speed",
    "Maximum wind speed",
    "Wind direction in degrees",
]

feature_keys = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

date_time_key = "Date Time"


def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

df.drop_duplicates(subset=["Date Time"],keep='first',inplace=True)
target_cov_dataset = TSDataset.load_from_dataframe(
    df,
    time_col='Date Time',
    target_cols='T (degC)',
    observed_cov_cols=['p (mbar)', 'VPmax (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'rho (g/m**3)', 'wv (m/s)'],
    freq='1h',
    fill_missing_dates=True,
    fillna_method='pre'
)
#target_cov_dataset.plot(['T (degC)', 'p (mbar)', 'VPmax (mbar)', 'VPdef (mbar)', 'sh (g/kg)','rho (g/m**3)', 'wv (m/s)'])

train_dataset, val_test_dataset = target_cov_dataset.split(0.7)
val_dataset, test_dataset = val_test_dataset.split(0.5)
#train_dataset.plot(add_data=[val_dataset,test_dataset], labels=['Val', 'Test'])

lstm = LSTNetRegressor(
    in_chunk_len = 10 * 72,
    out_chunk_len = 72,
    max_epochs=20
)
lstm.fit(train_dataset, val_dataset)

subset_test_pred_dataset = lstm.predict(val_dataset)
print(subset_test_pred_dataset)

subset_test_pred_dataset = lstm.predict(test_dataset)
print(subset_test_pred_dataset)
subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))

mae = MAE()
mae(subset_test_dataset, subset_test_pred_dataset)

subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])

plt.show()