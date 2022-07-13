# PCA
Principal Components Analysis

## Hướng dẫn
- Repo về Machine Learning: 
https://github.com/HT0710/ML-From-Scratch.git
- Repo chi tiết về PCA: 
https://github.com/HT0710/dsc-performing-principal-component-analysis.git

### main.py
Setup data từ file data.py

'''python
X_train, X_test, X_pca_train, X_pca_test, y_train, y_test, feature = setup_data()
LR = LinearRegression()
'''

Các lựa chọn để thực thi chương trình

'''python
n = 10
plot = False
his = False
run_main = True
'''

- Hàm main random id từ tập test, sau đó train 2 tập model và dùng id từ tập test random lúc đầu để dự đoán
- Lưu vào lịch sử
- Show ảnh nếu plot = True
- data.py: 'setup_data()' đầu vào cho main

'''python
def main():
'''

Object LR_PREDICT chứa các hàm dự đoán

'''pyhon
class LR_PREDICT:
'''

### data.py
Chứa các import

Setup data đầu vào, lọc dữ liệu và PCA

'''python
def setup_data():
'''

Load file lịch sử

'''python
def history():
'''

Object DATA chứ các hàm lọc dữ liệu và PCA

'''python
class DATA:
'''

### analysis.py
Phân tích dữ liệu các giữa các lần dự đoán trong lịch sử

### .history.csv
Ghi lại lịch sử các lần dự đoán

### Other
Chứa các file test cũ để tham khảo

### Linear_regression
Chứa file tham khảo về cách hồi quy tuyến tính

## Hiện tại
- Chọn lọc dữ liệu đầu vào
- Tập trung tăng độ chính xác cho hồi quy tuyến tính
- Đưa ra kết luận về PCA
