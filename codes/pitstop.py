from numpy.core.fromnumeric import transpose
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import pandas as pd
import numpy as np
import sys

## 使用 GPU 运行而不是 CPU ##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
##########################################

from tensorflow.python.keras.layers.core import Dropout
pd.options.mode.chained_assignment = None  # default='warn'

# 函数名称：safety_car_checker
# 函数功能：检查指定赛车数据文件中是否发生过安全车事件，如果发生过，则返回 True，否则返回 False。
# 参数：
# - file_location：赛车数据的 Excel 文件路径
# 返回：
# - True 或 False，表示是否发生安全车事件
def safety_car_checker(file_location):
    race = pd.read_excel(
        file_location, sheet_name="compoundchange",
        index_col = 0
    )

    # 移除不要的行
    for i in race.index:
        if (
            pd.isna(race["tyre_2"][i])
            or pd.isna(race["final_pos_gained"][i])
            or race["tyre_grid"][i] == "W"
            or race["tyre_grid"][i] == "I"
            or race["tyre_1"][i] == "W"
            or race["tyre_1"][i] == "I"
            or race["tyre_2"][i] == "W"
            or race["tyre_2"][i] == "I"
            ):
            race = race.drop([i])

    for i in race.sc:
        if (i>0):
            print(file_location)
            return True
    return False

# 函数名称：excel_unboxer
# 函数功能：解析指定的 Excel 赛车数据文件，并将其数据进行整合和返回。
# 参数：
# - file_location：Excel 文件路径
# 返回：
# - combined_table：整合了安全车信息、气温、湿度、高度和路线信息的数据表
def excel_unboxer(file_location):
    safetycar = pd.read_excel(
        file_location, sheet_name="sc", index_col=None
    )

    # 4. Track temperature
    weather = pd.read_excel(
        file_location, sheet_name="weather",
        index_col=0
    ).T

    temperature = weather[["Temperature"]]
    temperature["Temperature"] = temperature["Temperature"].str[:-2]
    temperature["Temperature"] = temperature["Temperature"].astype(float)

    humidity = weather[["Humidity"]]
    humidity["Humidity"] = humidity["Humidity"].astype(float)

    # 5. Track altitude
    altitude = pd.read_excel(
        file_location, sheet_name="altitude",
        dtype={"delta": float},
        index_col=0
    )

    fastestlap = pd.read_excel(
        file_location, sheet_name="fastestlap",
        dtype={"No.": str, "Team": str, "Time": str},
        index_col=0
    )

    combined_table = safetycar[
        ["No.", "initial_pos", "sc_lap", "sc_decision", "sc_laps_travelled", "sc_laps_remaining", "sc_tyre_compound",
         "before_pit_pos"]]

    for i in combined_table.index:
        if pd.isna(combined_table["sc_decision"][i]):
            print("error at: ", file_location)

    combined_table["Temperature"] = temperature.values[0][0]
    combined_table["Humidity"] = humidity.values[0][0]
    combined_table["Altitude"] = altitude.columns.values[0]

    # Adding static track information based on the file location
    track_info = {
        "australian": (16, 58, 5.303),
        "bahrain": (15, 57, 5.412),
        "chinese": (16, 56, 5.451),
        "azerbaijan": (20, 51, 6.003),
        "spanish": (16, 66, 4.655),
        "monaco": (19, 78, 3.337),
        "canada": (17, 70, 4.430),
        "french": (15, 53, 5.842),
        "austrian": (8, 71, 4.318),
        "british": (18, 52, 5.891),
        "german": (15, 64, 5.148),
        "hungarian": (13, 70, 3.975),
        "belgian": (19, 44, 7.004),
        "italian": (11, 59, 5.793),
        "singapore": (23, 61, 5.065),
        "russian": (18, 53, 5.848),
        "japanese": (18, 53, 5.807),
        "mexican": (17, 71, 4.304),
        "unitedstates": (20, 56, 5.513),
        "brazilian": (15, 71, 4.309),
        "abudhabi": (21, 55, 5.554),
        "eifel2020": (15, 60, 5.148),
        "imola2020": (15, 63, 4.909),
        "portuguese2020": (15, 66, 4.653),
        "sakhir2020": (11, 87, 3.543),
        "turkish2020": (15, 59, 5.245),
        "tuscan2020": (14, 58, 5.338),
        "malaysian2017": (15, 56, 5.543)
    }

    for track, info in track_info.items():
        if track in file_location:
            combined_table["Turns"] = info[0]
            combined_table["RaceDistance"] = info[1]
            combined_table["TrackLength"] = info[2]
            break
    else:
        combined_table["Turns"] = 15
        combined_table["RaceDistance"] = 50
        combined_table["TrackLength"] = 5
        print("Can't find the corresponding track...")

    # Team and driver ability assignment based on fastestlap data
    ability_mapping = {
        "TeamAbility": {
            "Mercedes-AMG Petronas Formula One Team": 573,
            "Scuderia Ferrari": 131,
            "Aston Martin Red Bull Racing": 319,
            "McLaren F1 Team": 202,
            "Renault DP World F1 Team": 181,
            "Scuderia AlphaTauri Honda": 107,
            "BWT Racing Point F1 Team": 195,
            "Alfa Romeo Racing ORLEN": 8,
            "Haas F1 Team": 3,
            "Williams Racing": 0
        },
        "DriverAbility": {
            "Lewis Hamilton": 347,
            "Valtteri Bottas": 223,
            "Max Verstappen": 214,
            "Charles Leclerc": 98,
            "Sebastian Vettel": 33,
            "Carlos Sainz": 105,
            "Pierre Gasly": 75,
            "Alexander Albon": 105,
            "Daniel Ricciardo": 119,
            "Sergio Pérez": 125,
            "Lando Norris": 97,
            "Kimi Räikkönen": 4,
            "Daniil Kvyat": 32,
            "Nico Hülkenberg": 10,
            "Lance Stroll": 75,
            "Kevin Magnussen": 1,
            "Antonio Giovinazzi": 4,
            "Romain Grosjean": 2,
            "Nicholas Latifi": 0,
            "George Russell": 3
        }
    }

    # 将 `fastestlap` 索引转换为字符串类型，确保索引类型一致
    fastestlap.index = fastestlap.index.astype(str)

    # 通过 `apply` 赋值时，先检查索引是否存在于 `fastestlap` 数据集
    combined_table["TeamAbility"] = combined_table["No."].apply(
        lambda x: ability_mapping["TeamAbility"].get(fastestlap.loc[str(x), "Team"], 0) if str(
            x) in fastestlap.index else 0)

    combined_table["DriverAbility"] = combined_table["No."].apply(
        lambda x: ability_mapping["DriverAbility"].get(fastestlap.loc[str(x), "Driver"], 0) if str(
            x) in fastestlap.index else 0)


    combined_table["final_pos_gained"] = safetycar["final_pos_gained"]  # 从安全车数据中添加 final_pos_gained 列

    # Reorder the columns
    combined_table = combined_table[
        ["initial_pos", "sc_lap", "sc_decision", "sc_laps_travelled", "sc_laps_remaining", "sc_tyre_compound",
         "before_pit_pos", "Temperature", "Humidity", "Altitude", "Turns", "RaceDistance", "TrackLength",
         "TeamAbility", "DriverAbility", "final_pos_gained"]  # 确保包含 final_pos_gained
    ]

    combined_table = combined_table.astype(float)
    return combined_table


# -----------------------------------------------------------------------------------------------------------------
# 1. 定义数据集的年份和路径信息
data_years = {
    "2020": ["70thanniversary2020", "abudhabi2020", "austrian2020", "bahrain2020", "belgian2020", "british2020",
             "eifel2020", "hungarian2020", "imola2020", "italian2020", "portuguese2020", "russian2020",
             "sakhir2020", "spanish2020", "styrian2020", "turkish2020", "tuscan2020"],
    "2019": ["abudhabi2019", "australian2019", "austrian2019", "azerbaijan2019", "bahrain2019", "belgian2019",
             "brazilian2019", "british2019", "canada2019", "chinese2019", "french2019", "german2019",
             "hungarian2019", "italian2019", "japanese2019", "mexican2019", "monaco2019", "russian2019",
             "singapore2019", "spanish2019", "unitedstates2019"],
    "2018": ["abudhabi2018", "australian2018", "austrian2018", "azerbaijan2018", "bahrain2018", "belgian2018",
             "brazilian2018", "british2018", "canada2018", "chinese2018", "french2018", "german2018",
             "hungarian2018", "italian2018", "japanese2018", "mexican2018", "monaco2018", "russian2018",
             "singapore2018", "spanish2018", "unitedstates2018"],
    "2017": ["abudhabi2017", "australian2017", "austrian2017", "azerbaijan2017", "bahrain2017", "belgian2017",
             "brazilian2017", "british2017", "canada2017", "chinese2017", "malaysian2017", "hungarian2017",
             "italian2017", "japanese2017", "mexican2017", "monaco2017", "russian2017", "singapore2017",
             "spanish2017", "unitedstates2017"],
    "2016": ["abudhabi2016", "australian2016", "austrian2016", "belgian2016", "british2016", "canada2016",
             "european2016", "hungarian2016", "italian2016", "japanese2016", "mexican2016", "monaco2016",
             "russian2016"]
}

dataset = pd.DataFrame()

# 2. 迭代读取数据集
for year, files in data_years.items():
    for file in files:
        file_path = f"./data{year}/{file}.xlsx"
        if safety_car_checker(file_path):
            temp_dataset = excel_unboxer(file_path)
            dataset = pd.concat([dataset, temp_dataset], axis=0)

# 重置索引
dataset = dataset.reset_index(drop=True)


# 打印列名以确认包含 `final_pos_gained`
print(dataset.columns)

# 3. 删除特定的行
# 使用字典来归类需要删除的索引
to_be_deleted_indices = {
    0: [],
    -1: [],
    1: [],
    2: [],
    3: [],
    4: []
}

for index, row in dataset.iterrows():
    if row['final_pos_gained'] in to_be_deleted_indices:
        to_be_deleted_indices[row['final_pos_gained']].append(index)

# 将最终的 DataFrame 保存为 CSV 文件
dataset.to_csv('dataset_safetycar.csv')



# ------------------------------------------------模型和评估模块------------------------------------------------
# apply tensorflow logics
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# split the dataset into train data and test data
from sklearn.model_selection import train_test_split

dataset = dataset.reset_index(drop=True)
print(dataset)

# 将标签 sc_decision 转换为二分类标签
# 1 表示需要进站，0 表示不需要进站
dataset["sc_decision"] = dataset["sc_decision"].apply(lambda x: 1 if x == "pit" else 0)

# 分割训练集和测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("sc_decision")
test_labels = test_features.pop("sc_decision")

# 数据标准化
from tensorflow.keras.layers.experimental import preprocessing
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

# 构建二分类模型
from tensorflow.keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    normalizer,
    layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 输出层，使用sigmoid激活函数进行二分类
])

# 编译模型，使用二分类的损失函数和评估指标
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 模型训练
noOfEpochs = 100
history = model.fit(train_features, train_labels, epochs=noOfEpochs, verbose=1, validation_split=0.2, shuffle=True)

# 绘制训练和验证的损失曲线
import matplotlib.pyplot as plt

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(range(noOfEpochs), history.history['loss'], label='Training Loss')
plt.plot(range(noOfEpochs), history.history['val_loss'], label='Validation Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# 绘制准确率曲线
plt.figure()
plt.xlabel('Epoch Number')
plt.ylabel("Accuracy")
plt.plot(range(noOfEpochs), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(noOfEpochs), history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.title('Training and Validation Accuracy')
plt.show()

# 评估模型性能
print("now evaluating:")
results = model.evaluate(test_features, test_labels, batch_size=16, verbose=1)
print("Test Loss, Test Accuracy:", results)

# 保存模型
model.save('safetycar/saved_model/')
print("Finished training the model")

