import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Tắt thông báo cảnh báo
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Đọc dữ liệu từ file CSV
data = pd.read_csv('stroke.csv')

# Xử lý dữ liệu thiếu
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

# Chuyển đổi dữ liệu phân loại thành số
label_encoders = {}
for column in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Loại bỏ cột 'id' và 'stroke' để làm dữ liệu đầu vào
X = data.drop(columns=['id', 'stroke'])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dữ liệu target
y = data['stroke']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Đánh giá độ chính xác trên tập kiểm tra
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Đếm số lượng bản ghi cho mỗi giá trị của các tiêu chí
gender_counts = data['gender'].value_counts()
hypertension_counts = data['hypertension'].value_counts()
heart_disease_counts = data['heart_disease'].value_counts()
ever_married_counts = data['ever_married'].value_counts()
work_type_counts = data['work_type'].value_counts()
residence_type_counts = data['Residence_type'].value_counts()
smoking_status_counts = data['smoking_status'].value_counts()

# Tính tỷ lệ phần trăm
total_records = len(data)
gender_percentage = (gender_counts / total_records) * 100
hypertension_percentage = (hypertension_counts / total_records) * 100
heart_disease_percentage = (heart_disease_counts / total_records) * 100
ever_married_percentage = (ever_married_counts / total_records) * 100
work_type_percentage = (work_type_counts / total_records) * 100
residence_type_percentage = (residence_type_counts / total_records) * 100
smoking_status_percentage = (smoking_status_counts / total_records) * 100

# In ra các tỷ lệ phần trăm
print("Gender percentage:")
print(gender_percentage)
print("\nHypertension percentage:")
print(hypertension_percentage)
print("\nHeart Disease percentage:")
print(heart_disease_percentage)
print("\nEver Married percentage:")
print(ever_married_percentage)
print("\nWork Type percentage:")
print(work_type_percentage)
print("\nResidence Type percentage:")
print(residence_type_percentage)
print("\nSmoking Status percentage:")
print(smoking_status_percentage)
