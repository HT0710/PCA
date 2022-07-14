# Mục tiêu
- Tối ưu hóa
- Tăng cường thử nghiệm
- Đưa ra kết luận về PCA

## main.py
Setup data từ file data.py

```python
X_train, X_test, X_pca_train, X_pca_test, y_train, y_test, feature = setup_data()
LR = LinearRegression()
```

Các lựa chọn để thực thi chương trình

```python
# Số lượng sample predict
n = 10
# Show Ảnh
plot = False
# Show lịch sử
his = False
# Thực thi main
run_main = True
```

- Hàm main random id từ tập test, sau đó train 2 tập model thường và PCA, sau đó dùng id từ tập test random lúc đầu để dự đoán
- Lưu vào lịch sử
- Show ảnh nếu plot = True
- data.py: `setup_data()` đầu vào cho main

```python
def main():
```

Object LR_PREDICT chứa các hàm dự đoán

```pyhon
class LR_PREDICT:
```

## data.py
Chứa các import

Setup data đầu vào, lọc dữ liệu và PCA

```python
def setup_data():
```

Load file lịch sử

```python
def history():
```

Object DATA chứ các hàm lọc dữ liệu và PCA

```python
class DATA:
```

## .history.csv
Ghi lại lịch sử các lần dự đoán

## analysis.py
Phân tích dữ liệu các giữa các lần dự đoán trong lịch sử

## conclusion.txt
Kết luận về quá trình làm dự án

## Tệp ./Other
Chứa các file test cũ để tham khảo

## Tệp ./Linear_regression
Chứa file tham khảo về cách hồi quy tuyến tính

# Liên kết khác
- Repo về Machine Learning: 
https://github.com/HT0710/ML-From-Scratch.git
- Repo chi tiết về PCA: 
https://github.com/HT0710/dsc-performing-principal-component-analysis.git
