Đây là một tệp script thực hiện tải dữ liệu, tiền xử lý, kỹ thuật tính năng, huấn luyện mô hình và dự đoán cho bài toán dự đoán sống sót trên tàu Titanic.
* **Kiểm tra sơ lược về data**:
![](/images/2025-06-24-140638_hyprshot.png)
* **Nhập thư viện**:
* **Nhập thư viện**:
* **Nhập thư viện**:
* **Nhập thư viện**:

## Phần 1: Thiết lập và tải dữ liệu ban đầu

* **Nhập thư viện**: Nhập các thư viện cần thiết để thao tác dữ liệu, trực quan hóa và học máy.
* **Tạo thư mục đầu ra**: Kiểm tra xem thư mục có tên "titanic\_images" có tồn tại không và tạo nếu chưa có để lưu các biểu đồ được tạo.
* **Tải dữ liệu**: Tải `train.csv` và `test.csv` vào các DataFrame của pandas.

## Phần 2: Phân tích dữ liệu thăm dò và kỹ thuật tính năng

* **Hiển thị thông tin cơ bản**: Hiển thị phần đầu, thông tin và thống kê mô tả của DataFrame huấn luyện, bao gồm cả số lượng giá trị rỗng.
* **Phân tích sự sống sót theo các tính năng phân loại**:
    * Tính toán tỷ lệ sống sót trung bình được nhóm theo `Pclass`, `Sex`, `SibSp` và `Parch`.
* **Kỹ thuật tính năng: Kích thước gia đình**:
    * Tạo tính năng `Family_Size` bằng cách cộng `SibSp` (anh chị em/vợ chồng) và `Parch` (cha mẹ/con cái) và thêm 1 (cho chính hành khách).
    * Ánh xạ `Family_Size` thành `Family_Size_Group` (Alone, Small, Medium, Large) để phân loại tốt hơn.
    * Phân tích tỷ lệ sống sót theo `Family_Size_Group`.
    * **Biểu đồ**: Tạo và lưu biểu đồ tần suất phân bố `Family_Size`.
    * **Ảnh chụp màn hình 1: `family_size_distribution.png` sẽ ở đây.**
* **Phân tích sự sống sót theo tuổi**:
    * **Biểu đồ**: Tạo và lưu biểu đồ phân bố `Age` theo `Survived`.
    * **Ảnh chụp màn hình 2: `age_distribution_by_survival.png` sẽ ở đây.**
    * **Kỹ thuật tính năng: Phân nhóm tuổi**:
        * Tạo `Age_Cut` bằng cách phân vị tính năng `Age` thành 8 nhóm.
        * Phân nhóm: Thay thế các giá trị `Age` liên tục bằng các danh mục số nguyên (0-8) dựa trên các điểm cắt phân vị này.
        * **Biểu đồ**: Tạo và lưu biểu đồ tần suất `Age` ban đầu với các điểm cắt phân vị.
        * **Ảnh chụp màn hình 3: `original_age_with_cuts_distribution.png` sẽ ở đây.**
        * **Biểu đồ**: Tạo và lưu biểu đồ tần suất các danh mục `Age` đã được phân nhóm.
        * **Ảnh chụp màn hình 4: `binned_age_distribution.png` sẽ ở đây.**
* **Phân tích sự sống sót theo giá vé**:
    * **Biểu đồ**: Tạo và lưu biểu đồ phân bố `Fare` theo `Survived`.
    * **Ảnh chụp màn hình 5: `fare_distribution_by_survival.png` sẽ ở đây.**
    * **Kỹ thuật tính năng: Phân nhóm giá vé**:
        * Tạo `Fare_Cut` bằng cách phân vị tính năng `Fare` thành 8 nhóm.
        * Phân nhóm: Thay thế các giá trị `Fare` liên tục bằng các danh mục số nguyên (0-8) dựa trên các điểm cắt phân vị này.
        * **Biểu đồ**: Tạo và lưu biểu đồ tần suất `Fare` ban đầu với các điểm cắt phân vị.
        * **Ảnh chụp màn hình 6: `original_fare_with_cuts_distribution.png` sẽ ở đây.**
        * **Biểu đồ**: Tạo và lưu biểu đồ tần suất các danh mục `Fare` đã được phân nhóm.
        * **Ảnh chụp màn hình 7: `binned_fare_distribution.png` sẽ ở đây.**
* **Kỹ thuật tính năng: Tiêu đề từ tên**:
    * Trích xuất `Title` (ví dụ: Mr., Mrs., Miss) từ cột `Name`.
    * Chuẩn hóa các tiêu đề khác nhau thành các danh mục rộng hơn (ví dụ: "Mlle" thành "Miss", "Capt" thành "Military").
    * Phân tích tỷ lệ sống sót theo `Title`.
* **Kỹ thuật tính năng: Độ dài tên**:
    * Tính toán `Name_Length`.
    * **Biểu đồ**: Tạo và lưu các biểu đồ KDE về `Name_Length` cho hành khách sống sót và không sống sót.
    * **Ảnh chụp màn hình 8: `Name_length_Survived.png` sẽ ở đây.**
    * Phân nhóm: Chia `Name_Length` thành các danh mục `Name_Size` (0-8) dựa trên các phân vị.
    * **Biểu đồ**: Tạo và lưu biểu đồ tần suất phân bố `Name_Length`.
    * **Ảnh chụp màn hình 9: `name_length_distribution.png` sẽ ở đây.**
* **Kỹ thuật tính năng: Thông tin vé**:
    * Trích xuất `TicketNumber` (phần cuối cùng của chuỗi vé).
    * Tính toán `TicketNumberCounts` (số lần một số vé xuất hiện).
    * Trích xuất `TicketLocation` (tiền tố của chuỗi vé nếu có).
    * Chuẩn hóa các giá trị `TicketLocation` khác nhau.
* **Kỹ thuật tính năng: Thông tin cabin**:
    * Điền các giá trị `Cabin` bị thiếu bằng "U" (không xác định).
    * Trích xuất chữ cái đầu tiên của `Cabin` làm danh mục `Cabin` mới.
    * Tạo `Cabin_Assigned` (0 nếu 'U', 1 nếu khác).
    * Phân tích tỷ lệ sống sót theo `Cabin` và `Cabin_Assigned`.
* **Xử lý các giá trị số bị thiếu**:
    * Điền các giá trị `Age` và `Fare` bị thiếu bằng giá trị trung bình tương ứng của chúng.

## Phần 3: Chu trình tiền xử lý mô hình

* **Xác định các bước tiền xử lý**:
    * `OrdinalEncoder`: Đối với các tính năng như `Family_Size_Group`.
    * `OneHotEncoder`: Đối với các tính năng phân loại như `Sex` và `Embarked`.
    * `SimpleImputer`: Để xử lý các giá trị bị thiếu (mặc dù hầu hết đã được xử lý, nhưng đây là một biện pháp bảo vệ).
* **Xác định mục tiêu và tính năng**:
    * `X`: Các tính năng để huấn luyện (bỏ `Survived`).
    * `y`: Biến mục tiêu (`Survived`).
    * `X_test`: Các tính năng để dự đoán.
* **Chia dữ liệu**: Chia dữ liệu huấn luyện thành `X_train`, `X_valid`, `y_train`, `y_valid` bằng cách sử dụng `train_test_split` với phân tầng.
* **Tạo các chu trình tiền xử lý**:
    * `ordinal_pipeline`: Điền các giá trị bị thiếu bằng giá trị phổ biến nhất, sau đó áp dụng `OrdinalEncoder`.
    * `ohe_pipeline`: Điền các giá trị bị thiếu bằng giá trị phổ biến nhất, sau đó áp dụng `OneHotEncoder`.
* **ColumnTransformer**:
    * Áp dụng các bước tiền xử lý khác nhau cho các cột khác nhau:
        * `Age` được điền.
        * `Family_Size_Group` sử dụng `ordinal_pipeline`.
        * `Sex` và `Embarked` sử dụng `ohe_pipeline`.
        * `Pclass`, `TicketNumberCounts`, `Cabin_Assigned`, `Name_Size` được truyền qua mà không chuyển đổi.

## Phần 4: Huấn luyện và đánh giá mô hình (với GridSearchCV)

* Phần này huấn luyện và điều chỉnh một cách có hệ thống một số mô hình phân loại bằng cách sử dụng `GridSearchCV` để tối ưu hóa siêu tham số và `StratifiedKFold` để xác thực chéo mạnh mẽ.
* **RandomForestClassifier (rfc)**:
    * Xác định lưới tham số cho `n_estimators`, `min_samples_split`, `max_depth`, `min_samples_leaf` và `criterion`.
    * Thực hiện `GridSearchCV`.
    * In các tham số tốt nhất và điểm xác thực chéo tốt nhất.
    * **Ảnh chụp màn hình 10: Đầu ra in cho `CV_rfc.best_params_` và `CV_rfc.best_score_` sẽ ở đây.**
* **DecisionTreeClassifier (dtc)**:
    * Xác định lưới tham số.
    * Thực hiện `GridSearchCV`.
    * In các tham số tốt nhất và điểm xác thực chéo tốt nhất.
    * **Ảnh chụp màn hình 11: Đầu ra in cho `CV_dtc.best_params_` và `CV_dtc.best_score_` sẽ ở đây.**
* **KNeighborsClassifier (knn)**:
    * Xác định lưới tham số.
    * Thực hiện `GridSearchCV`.
    * In các tham số tốt nhất và điểm xác thực chéo tốt nhất.
    * **Ảnh chụp màn hình 12: Đầu ra in cho `CV_knn.best_params_` và `CV_knn.best_score_` sẽ ở đây.**
* **SVC (svc)**:
    * Xác định lưới tham số.
    * Thực hiện `GridSearchCV`.
    * In các tham số tốt nhất và điểm xác thực chéo tốt nhất.
    * **Ảnh chụp màn hình 13: Đầu ra in cho `CV_svc.best_params_` và `CV_svc.best_score_` sẽ ở đây.**
* **LogisticRegression (lr)**:
    * Xác định lưới tham số.
    * Thực hiện `GridSearchCV`.
    * In các tham số tốt nhất và điểm xác thực chéo tốt nhất.
    * **Ảnh chụp màn hình 14: Đầu ra in cho `CV_lr.best_params_` và `CV_lr.best_score_` sẽ ở đây.**
* **GaussianNB (gnb)**:
    * Xác định lưới tham số.
    * Thực hiện `GridSearchCV`.
    * In các tham số tốt nhất và điểm xác thực chéo tốt nhất.
    * **Ảnh chụp màn hình 15: Đầu ra in cho `CV_gnb.best_params_` và `CV_gnb.best_score_` sẽ ở đây.**

## Phần 5: Dự đoán và xuất kết quả

* **Thực hiện dự đoán**: Sử dụng `pipefinaldtc` (Mô hình cây quyết định với các tham số tốt nhất từ GridSearchCV) để đưa ra dự đoán trên tập dữ liệu `X_test`.
* **Lưu kết quả**:
    * Tạo một DataFrame mới `test_results_df` bằng cách sao chép `test_df`.
    * Thêm cột `Survived_Prediction` vào `test_results_df`.
    * Sắp xếp lại các cột để dễ đọc hơn.
    * Lưu `test_results_df` vào tệp CSV có tên `detailed_predictions.csv`.
    * Đọc lại tệp CSV đã lưu để xác minh.
    * In phần đầu của một DataFrame con chứa `PassengerId` và `Survived_Prediction`.
    * **Ảnh chụp màn hình 16: Phần đầu của `subset_df` (PassengerId và Survived_Prediction) được in ra console sẽ ở đây.**
