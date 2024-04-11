import pandas as pd
import glob

# 设置合并CSV的路径
path = 'D:/python_code/pythonProject/data_list/training_data/c_ci_na_1'  # 用你存放CSV文件的文件夹路径替换这里
all_files = glob.glob(path + "/*.csv")

# 读取并合并文件
df = pd.concat((pd.read_csv(f) for f in all_files))

# 保存合并后的CSV文件
df.to_csv("c_ci_na_1_training_data.csv", index=False)
