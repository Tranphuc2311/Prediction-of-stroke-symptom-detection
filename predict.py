import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
data = pd.read_csv('stroke.csv')

# Chia dữ liệu thành các nhóm dựa trên giới tính
plt.figure(figsize=(12, 8))
sns.countplot(data=data, x='gender', hue='gender', palette='pastel', legend=False)
plt.title('Phân bố dữ liệu theo giới tính')
plt.xlabel('Giới tính')
plt.ylabel('Số lượng')
plt.show()

# Chia dữ liệu thành các nhóm dựa trên tình trạng hút thuốc
plt.figure(figsize=(12, 8))
sns.countplot(data=data, x='smoking_status', hue='smoking_status', palette='pastel', legend=False)
plt.title('Phân bố dữ liệu theo tình trạng hút thuốc')
plt.xlabel('Tình trạng hút thuốc')
plt.ylabel('Số lượng')
plt.show()

# Chia dữ liệu thành các nhóm dựa trên tiền sử bệnh tim
plt.figure(figsize=(12, 8))
sns.countplot(data=data, x='heart_disease', hue='heart_disease', palette='pastel', legend=False)
plt.title('Phân bố dữ liệu theo tiền sử bệnh tim')
plt.xlabel('Tiền sử bệnh tim')
plt.ylabel('Số lượng')
plt.show()
