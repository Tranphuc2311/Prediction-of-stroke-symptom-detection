import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Tắt thông báo cảnh báo
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

# Hàm dự đoán nguy cơ đột quỵ và trả về tỉ lệ phần trăm
def predict_stroke_risk(model, input_data):
    input_data = input_data.reshape(1, -1)
    proba = model.predict_proba(input_data)[0][1] * 100  # Lấy phần trăm xác suất của lớp 1
    return proba

# Hàm nhập dữ liệu từ người dùng và dự đoán nguy cơ đột quỵ
def main():
    print("Nhập thông tin của bạn:")
    gender = input("Giới tính (Male/Female): ")
    age = int(input("Tuổi: "))
    hypertension = int(input("Tiền sử cao huyết áp (1: Có, 0: Không): "))
    heart_disease = int(input("Tiền sử bệnh tim (1: Có, 0: Không): "))
    ever_married = input("Đã kết hôn chưa (Yes/No): ")
    work_type = input("Loại công việc (Private/Self-employed/Govt_job): ")
    Residence_type = input("Loại nơi cư trú (Urban/Rural): ")
    avg_glucose_level = float(input("Mức độ glucose trung bình: "))
    bmi = float(input("Chỉ số BMI: "))
    smoking_status = input("Tình trạng hút thuốc (formerly smoked/never smoked/smokes/Unknown): ")

    # Chuyển đổi dữ liệu đầu vào thành dạng số
    gender = label_encoders['gender'].transform([gender])[0]
    ever_married = label_encoders['ever_married'].transform([ever_married])[0]
    work_type = label_encoders['work_type'].transform([work_type])[0]
    Residence_type = label_encoders['Residence_type'].transform([Residence_type])[0]
    smoking_status = label_encoders ['smoking_status'].transform([smoking_status])[0]

    input_data = [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]
    input_data = scaler.transform([input_data])

    # Dự đoán và in ra tỉ lệ phần trăm
    proba = predict_stroke_risk(model, input_data)
    print(f"Nguy cơ bị đột quỵ: {proba*10:.2f}%")

if __name__ == "__main__":
    main()

# Bật lại thông báo cảnh báo
warnings.filterwarnings("default", category=UserWarning)
