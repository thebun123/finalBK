# Mạng nơ-ron nhận dạng các từ rời rạc số đếm tiếng việt từ không đến chín 

## 1. Cấu trúc

Thư mục bao gồm các file/thư mục con: 

* `sample.py`: thu mẫu âm thanh các số đếm  `.wav`, sample rate = 16000, mono.

* `train.py`: train model và đánh giá model.

* `split.py`: phân chia dữ liệu cho _tập huấn luyện_ (training set) và _tập xác thực_ (validation set).

* `data.py`: tạo vector đặc trưng và DataLoader.

* `net.py`: định nghĩa mạng nơ-ron.

* `utils.py`: các hàm phụ trợ khác như lưu ma trận lỗi,....

* `extract_feature.py`: trích xuất đặc trưng log spectrogram từ mẫu

* `./model`: thư mục chứa các model.

* `./figure`: thư mục chứa các biểu đồ quá trình train, validation, test.

* `./conf_matrix`: thư mục chứa ma trận lỗi.

* `./data`: thư mục chứa toàn bộ file âm thanh.

* `./data_to_train`: thư mục chứa dữ liệu train và validation.

* `./train`: thư mục chứa dữ liệu train.

* `./test`: thư mục chứa dữ liệu test.

* `./val`: thư mục chứa dữ liệu validation.
## 2. Thu mẫu âm thanh

```bash
python sample.py 
```

mẫu sẽ được thu trong 2s. Sau khi phát âm, cần chọn thời điểm để bắt đầu cắt lấy mẫu trong khoảng 2s đó tuỳ vào thời điểm phát âm. Nếu mẫu ổn thì bấm phím 1 để lưu, 0 để huỷ mẫu đó.
Sau khi thu một mẫu, bấm phím 0 nếu muốn dừng thu mẫu, thu tiếp bấm phím bất kỳ.
## 3. Phân chia dữ liệu
Nếu muốn chia lại mẫu chạy

```bash
python split_data.py --make_new True
```

## 4. Huấn luyện

```bash
python train.py
```
Mô hình sau khi huấn luyện sẽ lưu ở trong thư mục model. Sau khi huấn luyện xong một số biểu đồ về các thông số trong quá trình huấn luyện sẽ được tạo ở thư mục figure.
--Truong--
