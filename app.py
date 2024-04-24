from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from bson import ObjectId

app = Flask(__name__)

# Kết nối đến MongoDB
client = MongoClient('mongodb+srv://kkimkhoi3010:7WdJZZ3rsGHhDoRK@cluster0.hxhjxgo.mongodb.net/')
db = client['Stroke']
collection = db['stroke']

# Kết nối đến collection chứa ảnh
image_collection = db['images']

feedback_collection = db['feedbacks']

# Kết nối đến collection chứa thông tin người dùng
user_collection = db['users']

# Chọn dữ liệu từ MongoDB và chuyển đổi thành DataFrame
data = pd.DataFrame(list(collection.find()))

# Loại bỏ cột ObjectId từ DataFrame
data = data.drop('_id', axis=1)

# Xử lý dữ liệu thiếu
data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
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
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_accuracy = accuracy_score(y_test, logistic_model.predict(X_test))

# Huấn luyện mô hình Random Forest
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
random_forest_accuracy = accuracy_score(y_test, random_forest_model.predict(X_test))

# Huấn luyện mô hình Neural Network
nn_model = MLPClassifier()
nn_model.fit(X_train, y_train)
nn_accuracy = accuracy_score(y_test, nn_model.predict(X_test))

# Huấn luyện mô hình Decision Tree
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_model.predict(X_test))

# Đánh giá độ chính xác trên tập kiểm tra
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        residence_type = request.form['residence_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']

        # Chuyển đổi giá trị hypertension, heart_disease, ever_married thành dạng "Yes" hoặc "No"
        hypertension_str = "Yes" if hypertension == 1 else "No"
        heart_disease_str = "Yes" if heart_disease == 1 else "No"
        ever_married_str = ever_married  # Giữ nguyên giá trị vì đã ở dạng "Yes" hoặc "No"

        # Chuyển đổi dữ liệu thành dạng số để chạy hàm predict_stroke_risk
        gender_encoded = label_encoders['gender'].transform([gender])[0].item()
        work_type_encoded = label_encoders['work_type'].transform([work_type])[0].item()
        residence_type_encoded = label_encoders['Residence_type'].transform([residence_type])[0].item()
        smoking_status_encoded = label_encoders['smoking_status'].transform([smoking_status])[0].item()
        ever_married_encoded = label_encoders['ever_married'].transform([ever_married])[0].item()


        input_data = [gender_encoded, age, hypertension, heart_disease, ever_married_encoded, work_type_encoded,
                      residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded]
        input_data = scaler.transform([input_data])

        proba = predict_stroke_risk(logistic_model, input_data)

        # Lưu dữ liệu người dùng vào collection "users" với khóa "stroke_probability"
        user_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension_str,  # Sử dụng giá trị đã chuyển đổi
            'heart_disease': heart_disease_str,  # Sử dụng giá trị đã chuyển đổi
            'ever_married': ever_married_str,  # Sử dụng giá trị đã chuyển đổi
            'work_type': work_type,
            'residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status,
            'stroke_probability': proba,  # Thêm khóa "stroke_probability" vào dữ liệu người dùng
            'logistic_accuracy': logistic_accuracy,
            'random_forest_accuracy': random_forest_accuracy,
            'neural_network_accuracy': nn_accuracy,
            'decision_tree_accuracy': decision_tree_accuracy
        }
        user_id = user_collection.insert_one(user_data).inserted_id

        draw_and_save_charts(user_id,user_data)
        # Dự đoán xác suất đột quỵ với các mô hình
        logistic_proba = logistic_model.predict_proba(input_data)[0][1]
        rf_proba = random_forest_model.predict_proba(input_data)[0][1]
        nn_proba = nn_model.predict_proba(input_data)[0][1]
        dt_proba = decision_tree_model.predict_proba(input_data)[0][1]

        # Lưu xác suất đột quỵ vào MongoDB
        user_collection.update_one({'_id': user_id}, {'$set': {'logistic_probability': logistic_proba,
                                                               'random_forest_probability': rf_proba,
                                                               'neural_network_probability': nn_proba,
                                                               'decision_tree_probability': dt_proba}})
        draw_and_save_charts(str(user_id),user_data)
        return redirect(url_for('prediction_result', user_id=user_id))


def draw_and_save_charts(user_id, user_data):
    # Tính tỉ lệ phần trăm nguy cơ bị đột quỵ theo nhóm tuổi từ collection "stroke"
    age_groups = data.groupby(pd.cut(data['age'], bins=[0, 18, 35, 50, 65, 100], labels=['<18', '18-35', '36-50', '51-65', '65+']))
    stroke_counts = age_groups['stroke'].sum()
    total_counts = age_groups['stroke'].count()
    stroke_percentage_age = (stroke_counts / total_counts) * 100

    # Tạo một danh sách màu sắc cho từng nhóm tuổi
    colors = ['grey'] * len(stroke_percentage_age.index)
    # Xác định chỉ số của nhóm tuổi của người dùng trong danh sách
    if user_data['age'] < 18:
        user_age_index = 0
    elif 18 <= user_data['age'] < 35:
        user_age_index = 1
    elif 35 <= user_data['age'] < 50:
        user_age_index = 2
    elif 50 <= user_data['age'] < 65:
        user_age_index = 3
    else:
        user_age_index = 4
    # Gán màu xanh lá cây cho nhóm tuổi của người dùng
    colors[user_age_index] = 'green'

    plt.figure(figsize=(8, 6))
    sns.barplot(x=stroke_percentage_age.index, y=stroke_percentage_age.values, palette=colors)
    plt.xlabel('Age Group')
    plt.ylabel('Stroke Risk Percentage')
    plt.title('Stroke Risk Percentage by Age Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Lưu biểu đồ vào một buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode biểu đồ dưới dạng base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Lưu biểu đồ vào collection images của MongoDB
    image_doc = {
        'name': f'stroke_percentage_by_age_{user_id}.png',
        'data': encoded_image
    }
    image_collection.insert_one(image_doc)

    plt.close()




    # Tính tỉ lệ phần trăm nguy cơ đột quỵ theo nhóm giới tính từ collection "stroke"
    gender_groups = data.groupby('gender')
    stroke_counts_gender = gender_groups['stroke'].sum()
    total_counts_gender = gender_groups['stroke'].count()
    stroke_percentage_gender = (stroke_counts_gender / total_counts_gender) * 100

    # Tạo danh sách màu sắc cho từng nhóm giới tính
    colors = ['gray'] * len(stroke_percentage_gender.index)
    # Xác định chỉ số của giới tính của người dùng trong danh sách
    if user_data['gender'] == 'Male':
        user_gender_index = 0
    elif user_data['gender'] == 'Female':
        user_gender_index = 1
    # Gán màu xanh lá cây cho giới tính của người dùng
    colors[user_gender_index] = 'green'

    plt.figure(figsize=(8, 6))
    sns.barplot(x=stroke_percentage_gender.index, y=stroke_percentage_gender.values, palette=colors)
    plt.xlabel('Gender')
    plt.ylabel('Stroke Risk Percentage')
    plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
    plt.title('Stroke Risk Percentage by Gender')
    plt.tight_layout()
    # Lưu biểu đồ vào một buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode biểu đồ dưới dạng base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Lưu biểu đồ vào collection images của MongoDB
    image_doc = {
        'name': f'stroke_percentage_by_gender_{user_id}.png',
        'data': encoded_image
    }
    image_collection.insert_one(image_doc)
    plt.close()




    # Tính tỉ lệ phần trăm nguy cơ bị đột quỵ theo nhóm tiền sử bệnh tim từ collection "stroke"
    heart_disease_groups = data.groupby('heart_disease')
    stroke_counts_heart_disease = heart_disease_groups['stroke'].sum()
    total_counts_heart_disease = heart_disease_groups['stroke'].count()
    stroke_percentage_heart_disease = (stroke_counts_heart_disease / total_counts_heart_disease) * 100

    # Tạo một danh sách màu sắc cho từng nhóm tiền sử bệnh tim
    colors = ['gray'] * len(stroke_percentage_heart_disease.index)
    # Xác định chỉ số của nhóm tiền sử bệnh tim của người dùng trong danh sách
    if user_data['heart_disease'] == 0:
        user_heart_disease_index = 0
    else:
        user_heart_disease_index = 1
    # Gán màu xanh lá cây cho nhóm tiền sử bệnh tim của người dùng
    colors[user_heart_disease_index] = 'green'

    plt.figure(figsize=(8, 6))
    sns.barplot(x=stroke_percentage_heart_disease.index, y=stroke_percentage_heart_disease.values, palette=colors)
    plt.xlabel('Heart Disease History')
    plt.ylabel('Stroke Risk Percentage')
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
    plt.title('Stroke Risk Percentage by Heart Disease History')
    plt.tight_layout()
    # Lưu biểu đồ vào một buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode biểu đồ dưới dạng base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Lưu biểu đồ vào collection images của MongoDB
    image_doc = {
        'name': f'stroke_percentage_by_heart_{user_id}.png',
        'data': encoded_image
    }
    image_collection.insert_one(image_doc)
    plt.close()





    # Tính tỉ lệ phần trăm nguy cơ đột quỵ theo nhóm sinh sống từ collection "stroke"
    residence_groups = data.groupby('Residence_type')
    stroke_counts_residence = residence_groups['stroke'].sum()
    total_counts_residence = residence_groups['stroke'].count()
    stroke_percentage_residence = (stroke_counts_residence / total_counts_residence) * 100

    # Tạo danh sách màu sắc cho từng nhóm sinh sống
    colors = ['gray'] * len(stroke_percentage_residence.index)
    # Xác định chỉ số của nhóm sinh sống của người dùng trong danh sách
    if user_data['residence_type'] == 'Urban':
        user_residence_index = 0
    else:
        user_residence_index = 1
    # Gán màu xanh lá cây cho nhóm sinh sống của người dùng
    colors[user_residence_index] = 'green'

    plt.figure(figsize=(8, 6))
    sns.barplot(x=stroke_percentage_residence.index, y=stroke_percentage_residence.values, palette=colors)
    plt.xlabel('Residence Type')
    plt.ylabel('Stroke Risk Percentage')
    plt.title('Stroke Risk Percentage by Residence Type')
    plt.xticks(ticks=[0, 1], labels=['Rural', 'Urban'])
    plt.tight_layout()
    # Lưu biểu đồ vào một buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode biểu đồ dưới dạng base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Lưu biểu đồ vào collection images của MongoDB
    image_doc = {
        'name': f'stroke_percentage_by_residence_{user_id}.png',
        'data': encoded_image
    }
    image_collection.insert_one(image_doc)
    plt.close()





    # Tính tỉ lệ phần trăm nguy cơ đột quỵ theo nhóm tình trạng hút thuốc từ collection "stroke"
    smoking_groups = data.groupby('smoking_status')
    stroke_counts_smoking = smoking_groups['stroke'].sum()
    total_counts_smoking = smoking_groups['stroke'].count()
    stroke_percentage_smoking = (stroke_counts_smoking / total_counts_smoking) * 100

    # Tạo danh sách màu sắc cho từng nhóm tình trạng hút thuốc
    colors = ['gray'] * len(stroke_percentage_smoking.index)
    user_smoking_index = None
    # Xác định chỉ số của nhóm tình trạng hút thuốc của người dùng trong danh sách
    if user_data['smoking_status'] == 'Unknown':
        user_smoking_index = 0
    elif user_data['smoking_status'] == 'Never smoked':
        user_smoking_index = 1
    elif user_data['smoking_status'] == 'Formerly smoked':
        user_smoking_index = 2
    else:
        user_smoking_index = 3
    # Gán màu xanh lá cây cho nhóm tình trạng hút thuốc của người dùng
    colors[user_smoking_index] = 'green'

    plt.figure(figsize=(8, 6))
    sns.barplot(x=stroke_percentage_smoking.index, y=stroke_percentage_smoking.values, palette=colors)
    plt.xlabel('Smoking Status')
    plt.ylabel('Stroke Risk Percentage')
    plt.title('Stroke Risk Percentage by Smoking Status')
    plt.tight_layout()
    # Lưu biểu đồ vào một buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode biểu đồ dưới dạng base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Lưu biểu đồ vào collection images của MongoDB
    image_doc = {
        'name': f'stroke_percentage_by_smoking_{user_id}.png',
        'data': encoded_image
    }
    image_collection.insert_one(image_doc)
    plt.close()




    # Tính tỉ lệ phần trăm nguy cơ đột quỵ theo nhóm tình trạng huyết áp từ collection "stroke"
    hypertension_groups = data.groupby('hypertension')
    stroke_counts_hypertension = hypertension_groups['stroke'].sum()
    total_counts_hypertension = hypertension_groups['stroke'].count()
    stroke_percentage_hypertension = (stroke_counts_hypertension / total_counts_hypertension) * 100

    # Tạo danh sách màu sắc cho từng nhóm tình trạng huyết áp
    colors = ['gray'] * len(stroke_percentage_hypertension.index)
    # Xác định chỉ số của nhóm tình trạng huyết áp của người dùng trong danh sách
    if user_data['hypertension'] == 0:
        user_hypertension_index = 0
    else:
        user_hypertension_index = 1
    # Gán màu xanh lá cây cho nhóm tình trạng huyết áp của người dùng
    colors[user_hypertension_index] = 'green'

    plt.figure(figsize=(8, 6))
    sns.barplot(x=stroke_percentage_hypertension.index, y=stroke_percentage_hypertension.values, palette=colors)
    plt.xlabel('Hypertension Status')
    plt.ylabel('Stroke Risk Percentage')
    plt.title('Stroke Risk Percentage by Hypertension Status')
    plt.xticks(rotation=45)  # Điều chỉnh góc quay của nhãn trục x
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
    plt.tight_layout()
    # Lưu biểu đồ vào một buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode biểu đồ dưới dạng base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Lưu biểu đồ vào collection images của MongoDB
    image_doc = {
        'name': f'stroke_percentage_by_hypertension_{user_id}.png',
        'data': encoded_image
    }
    image_collection.insert_one(image_doc)

    plt.close()

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        rating = int(request.form['rating'])
        comment = request.form['comment']

        # Lưu dữ liệu feedback vào MongoDB
        feedback_data = {
            'rating': rating,
            'comment': comment
        }
        feedback_collection.insert_one(feedback_data)
        return render_template('some_template.html', message='Feedback submitted successfully!')

@app.route('/predict_result')
def prediction_result():
    user_id = request.args.get('user_id')
    user_data = user_collection.find_one({'_id': ObjectId(user_id)})
    return render_template('predict.html', user_data=user_data)

@app.route('/display_images')
def display_images():
    image_name = request.args.get('name')
    image = image_collection.find_one({'name': image_name})
    image_data = image['data']
    return send_file(BytesIO(base64.b64decode(image_data)), mimetype='image/png')


def predict_stroke_risk(model, input_data):
    proba = model.predict_proba(input_data)[0][1]
    return proba



if __name__ == '__main__':
    app.run(debug=True)
