# **Module 01 - Natural Language Processing with Classification and Vector Spaces**
## **Week 1: Sentiment Analysis with Logistic Regression**

- Biểu diễn văn bản dưới dạng text thành vector và xây dựng một bộ phân loại sẽ phân loại văn bản mẫu thành hai loại (Tâm lý tích cực hoặc Tâm lý tiêu cực). Sử dụng Logistic Regression. 

### **Logistic Regression**
#### 1. Supervised Machine Learning (Học có giám sát)

![M1_W1_01_Supervised ML](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_01_Supervised%20ML.png)
> Kiến trúc của Supervise ML

- Trong máy học giám sát bạn có các tính năng đầu vào **X** và tập hợp các nhãn **Y**
- Để đảm bảo rằng bạn nhận được **dự đoán chính xác nhất** dựa trên dữ liệu của bạn.
- Mục tiêu của bản là **giảm thiểu** tỷ lệ **lỗi** của bạn hoặc **chi phí** càng nhiều càng tốt. 
- Và để làm được điều này, phải chạy được **Prediction Function** của bạn, cái mà lấy trong dữ liệu tham số để gán các Feature của bạn để đầu ra nhãn Y^
- Mapping tốt nhất từ các Features đến nhãn đạt được khi sự khác biệt giữa các giá trị kỳ vọng Y và giá trị dự đoán Y^ được **giảm thiểu**.
- **Hàm chi phí** thực hiện bằng cách so sánh mức độ gần gũi giữa Output Y^ của bạn với nhãn Y.
- Sau đó bạn có thể cập nhật tham số và lặp lại toàn bộ quá trình xử lý cho đến khi tối ưu được chi phí thấp nhất. 



#### 2. Sentiment Analysis (Phân tích tình cảm)

![Sentiment analysis Ví dụ](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_02_Sentiment%20analysis.png)
>Ví dụ về Sentiment analysis 

- Trong ví dụ này bạn có: 

    ```
    Tweet: Tôi rất hạnh phúc bởi vì tôi đang học NLP
    ```

- Và mình cần chứng minh rằng câu nói này đang là **tâm lý tích cực** hay **tâm lý tiêu cực**
- Để thực hiện bạn cần chuẩn bị tập training: 

    ```
    Postive (Tâm lý tích cực): -> Lable: 1
    Negative (Tâm lý tiêu cực): -> Lable: 0
    ```

- Dùng Logistic Regression đã dán nhãn, gán quan sát của nó cho hai lớp khác biệt

![Cách xây dựng Sentiment analysis](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_02_Sentiment%20analysis.png)
> Cách xây dựng Sentiment analysis

- Để xây dựng một bộ phân loại hồi quy logistic, có khả năng dự đoán tình cảm của một tweet tùy ý. 
- Xử lý các tweet thô trong training set và trích xuất các tính năng hữu ích
- Sau đó train Logistic Regression của bạn cùng với đó phải giảm thiểu chi phí. 
- Classify, cuối cùng bạn sẽ có thể đưa ra dự đoán của bạn

### **Vocabulary & Feature Extraction**
#### 3. Vocabulary 
- Để biểu diễn text dưới dạng vector, cần phải xây dựng một bộ từ vựng và nó sẽ cho phép bạn mã hóa bất kỳ text nào hoặc bất kỳ tweet nào dưới dạng một mảng số.

![M1_01_01_Vocabulary](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_03_Vocabulary.png)

> Cách lưu các text thành mảng số

- Hình ảnh một danh sách các tweet, trực quan hóa nó sẽ là các câu. 
- Sau đó từ vựng V sẽ là danh sách các từ duy nhất trong danh sách các tweet của bạn. 
- Để có được list đó thì bạn phải xem qua tất cả các từ vựng từ tất cả các tweet của bạn và lưu mọi từ mới xuất hiện trong tìm kiếm của bạn. 
- Lưu ý trong 2 câu có lặp từ thì chỉ lấy 1 từ duy nhất, không lặp lại hai từ đó. 

#### 4. Feature Extraction

![Biểu diễn Feature Extraction](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_04_Feature%20Extraction.png)
> Giải thích ảnh trên: 

- Chúng ta có một câu (I am happy because I am learning NLP)
- Làm sao để xác định câu này có trong list. 
- Thì nó sẽ gán giá trị là 1 nếu từ vựng trong câu trên có xuất hiện trong list, còn tất cả các giá trị còn lại không xuất hiện trong list sẽ hiểu là 0
- Những nó sẽ sinh ra vấn đề -> **Quá nhiều số 0 và biễu diễn quá thưa thớt.**

#### 5. Vấn đề biểu diễn thưa thớt 

![M1_W1_05_Problem Spare Representation](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_05_Problem%20Spare%20Representation.png)
> Giải thích ảnh trên: 

- Với sử biểu diễn thưa thớt, thì mô hình hồi quy Logistic sẽ phải học **n + 1 parameter**. 
- Trong đó n sẽ bằng kích thước từ vựng -> Nếu **kích thước từ vựng lớn** điều này sẽ là 1 vấn đề. 
- Mô hình mất rất nhiều thời gian để training và mất nhiều thời gian hơn cần thiết để đưa ra dự đoán

- Với text đã được học cách biểu diễn dứi dạng một Vector có kích thước v cụ thể -> 1 Tweet có thể xây dựng dưới một từ vựng của chiều V. 
- Nếu V trở nên lớn hơn -> **bạn sẽ gặp phải một số vấn đề**
- Như bạn có thể thấy, khi V trở nên lớn hơn, vector trở nên thưa thớt hơn. Hơn nữa, chúng ta sẽ có nhiều đặc trưng hơn và kết quả là phải **huấn luyện nhiều tham số θ** của V hơn. 
- Điều này có thể dẫn đến **thời gian huấn luyện lâu hơn** và **thời gian dự đoán cũng lớn hơn**.

**Negative and Positive** 

#### 6. Feature Extraction with Frequencies

- Mỗi hàng text sẽ là 1 Tweet 

- Cho một tập hợp dữ liệu với các tweet tích cực và tiêu cực như sau

![M1_W1_06_Feature Extraction with Frequencies_01](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_06_Feature%20Extraction%20with%20Frequencies_01.png)

- Bạn phải mã hóa mỗi tweet dưới dạng một vectơ. 
- Trước đây, vectơ này có kích thước V. 
- Bây giờ, như bạn sẽ thấy trong các video sắp tới, bạn sẽ biểu diễn nó bằng một vectơ có kích thước 3. 
- Để làm được điều này, bạn phải tạo một từ điển để gán từ và lớp mà nó xuất hiện (tích cực hoặc tiêu cực) với số lần từ đó xuất hiện trong lớp tương ứng của nó.

![M1_W1_06_Feature Extraction with Frequencies_02](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_06_Feature%20Extraction%20with%20Frequencies_02.png)

- Trong hai video trước, chúng tôi gọi từ điển này là `freqs`. 
- Trong bảng trên, bạn có thể thấy các từ như happy và sad có xu hướng nghiêng về một thái cực rõ ràng, trong khi các từ khác như "I, am" thường có xu hướng trung lập hơn. 
- Dựa trên từ điển này và tweet, "I am sad, I am not learning NLP", bạn có thể tạo một vector tương ứng với đặc trưng như sau:

![M1_W1_06_Feature Extraction with Frequencies_03](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_06_Feature%20Extraction%20with%20Frequencies_03.png)

- Để mã hóa đặc điểm tiêu cực, bạn có thể làm việc tương tự

![M1_W1_06_Feature Extraction with Frequencies_04](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_06_Feature%20Extraction%20with%20Frequencies_04.png)

- Do đó, bạn sẽ nhận được vectơ đặc trưng sau đây[1,8,11] . 1, tương ứng với độ lệch (bias), 8 là đặc trưng dương, và 11 là đặc trưng âm.

#### 7. Preprocessing
- Học về hai khái niệm chính về tiền xử lý: 
    + Khái niệm thứ nhất gọi là `steaming` (**bắt nguồn từ**) 
    + Khái niệm thứ hai gọi là `stop word` (**dừng từ**)
- Học về cách sử dụng `steaming` và `stop word` để xử lý trước văn bản

- Khi tiền xử lý, bạn phải thực hiện theo các bước sau: 
    + Loại bỏ các handle và URL
    + Phân tách chuỗi thành các từ
    + Xóa các `stop word` như: "and, is, a, o, v.v"
    + `steaming` hoặc chuyển đổi từng từ thành gốc của nó. Ví dụ như dancer, dancing, danced, thành **danc**. Bạn có thế sử dụng `porter stemmer` để xử lý việc này. 
    + Chuyển đổi tất cả các từ của bạn sang **chữ thường**. 

- Ví dụ như dòng tweet sau
   
    ```
    "@YMourri and @AndrewYNg are tuning a GREAT AI model at https://deeplearning.ai!!!"
    ```

- Sau khi tiền xử lý nó sẽ trở thành như sau: 

![M1_W1_07_Preprocessing](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_07_Preprocessing.png)

[*tun, great, ai, model*] Do đó, bạn có thể thấy cách chúng tôi **loại bỏ các ký tự xử lý**, phân tách thành các từ, **loại bỏ các từ dừng**, thực hiện chuyển đổi gốc và **chuyển đổi mọi thứ thành chữ thường.**

#### 8. Putting it All Together
- Nhìn chung, bạn bắt đầu với một văn bản cho trước, bạn thực hiện tiền xử lý, sau đó bạn trích xuất đặc điểm để chuyển đổi văn bản thành biểu diễn số như sau:

![M1_W1_08_PuttingIAT](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_08_PuttingIAT.png)

- **X** của bạn, trở thành kích thước (*m*, 3) như sau:

![M1_W1_08_PuttingIAT_01](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_08_PuttingIAT_01.png)

- Khi triển khai bằng code, nó sẽ như sau:

![M1_W1_08_PuttingIAT_02](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_08_PuttingIAT_02.png)

- Bạn có thể thấy ở bước cuối cùng, bạn đang lưu trữ các **tính năng được trích xuất dưới dạng các hàng** (extract_features) trong ma trận **X** và bạn có *m* ví dụ này.

### **Logistic Regression Overview**

- Đây là cái nhìn tổng quan về **hồi quy logistic** (logistic regression).
- Bạn sẽ sử dụng các **tính năng** (features) đã trích xuất để dự đoán một **tweet** có tâm lý tích cực hay tiêu cực.
- Hồi quy logistic sử dụng một **hàm sigmoid** (sigmoid function), xuất ra một xác suất từ 0 đến 1.
> Overview of logistic regression

![M1_W1_09_Logistic Regression Overview](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_09_Logistic_Regression_Overview.png)

- Trong **máy học được giám sát** (supervised machine learning), bạn có các tính năng đầu vào và nhãn. Bạn dùng một hàm với các tham số để ánh xạ các đối tượng với nhãn đầu ra.
- Để có ánh xạ tối ưu, bạn **giảm thiểu hàm chi phí** (cost function) bằng cách so sánh đầu ra **Y hat** với nhãn thật **Y**. Các **tham số** (parameters) được cập nhật lặp lại cho đến khi chi phí được giảm thiểu.
- Đối với hồi quy logistic, Hàm F trong hình là hàm Sigmoid
> Biểu đạt bằng phương trình

![M1_W1_10_Logistic_Regression_Overview_02](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_10_Logistic_Regression_Overview_02.png)
- Đối với hàm hồi quy logistic (H) là **hàm sigmoid**, phụ thuộc vào tham số **Theta** và vector tính năng **X dấu trên i** (quan sát thứ i, hoặc tweet thứ i).
- Hàm sigmoid tiếp cận 0 khi tích chấm của **Theta transpose X** ($\theta^T X$) tiến đến âm vô cực, và tiếp cận 1 khi nó tiến đến dương vô cực.
- Để phân loại, cần một **ngưỡng** (threshold), thường là **0.5**.
- Giá trị 0.5 tương ứng với **tích chấm** ($\theta^T X$) bằng 0.
- Khi tích chấm $\ge 0$, dự đoán là dương. Khi tích chấm $< 0$, dự đoán là âm.
> Một ví dụ được đưa ra trong bối cảnh **phân tích tình cảm** (sentiment analysis) tweet.

![M1_W1_10_Logistic_Regression_Overview_03](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_10_Logistic_Regression_Overview_03.png)

- Sau **tiền xử lý** (preprocessing) (ví dụ: chữ thường, giảm từ về gốc như 'tun'), bạn trích xuất các tính năng thành một vector.
- Vector này bao gồm một **đơn vị thiên vị** (bias unit) và các tính năng (như tổng tần số tích cực và tiêu cực).
- Giả sử đã có bộ tham số **Theta** tối ưu, bạn có thể nhận được giá trị của **hàm sigmoid** (ví dụ 4.92 trong script) và dự đoán một tình cảm tích cực.
- Bây giờ bạn đã biết ký hiệu (notation) để **đào tạo** (train) một yếu tố trọng lượng **Theta**.
- Video tiếp theo sẽ nói về cơ chế (mechanics) đằng sau việc đào tạo bộ phân loại hồi quy logistic.

#### 1. Logistic Regression: Training
- Video này hướng dẫn bạn học (tìm) **theta** ($\theta$) của riêng bạn từ đầu.
- Để đào tạo bộ phân loại **hồi quy logistic**, bạn lặp lại cho đến khi tìm thấy **theta** ($\theta$) giúp **giảm thiểu hàm chi phí** (cost function) J.
- Một ví dụ trực quan cho thấy hàm chi phí (giả sử phụ thuộc vào **theta1** và **theta2**) và sự tiến hóa của nó qua các lần lặp (100, 200,...) để tiến gần đến chi phí tối ưu.
> Kiến trúc Training LR

![M1_W1_11_Logistic_Regression_Training](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_11_Logistic_Regression_Training.png)

- Quá trình chi tiết bao gồm:
    1.  **Khởi tạo** (Initialize) vector tham số **theta** ($\theta$).
    2.  Sử dụng **hàm logistic** (logistic function) để lấy giá trị cho mỗi quan sát.
    3.  Tính toán **gradient** (gradient) của hàm chi phí.
    4.  **Cập nhật** (Update) các tham số.
    5.  Tính toán **chi phí J** (cost J) và quyết định xem có cần lặp thêm không (dựa trên tham số dừng hoặc số lần lặp tối đa).
- Thuật toán này được gọi là **gradient descent**.
> Thông thường bạn sẽ tiếp tục huấn luyện cho đến khi chi phí hội tụ. Nếu bạn vẽ biểu đồ số lần lặp so với chi phí, bạn sẽ thấy điều gì đó như sau:

![M1_W1_12_Logistic_Regression_Training_02](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_12_Logistic_Regression_Training_02.png)
- Sau khi có **theta** ($\theta$), bạn cần **đánh giá** (evaluate) nó, tức là đánh giá bộ phân loại của bạn (tốt hay xấu).
- Video tiếp theo sẽ chỉ cách đánh giá này.

#### 2. Logistic Regression: Testing

- Bạn sẽ sử dụng dữ liệu để dự đoán các điểm dữ liệu mới (ví dụ: tweet tích cực hay tiêu cực) và phân tích xem mô hình có **khái quát hóa** (generalize) tốt không.
> Phần này chỉ cách tính **độ chính xác** (accuracy) của mô hình.

![M1_W1_13_Logistic_Regression_Testing](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_13_Logistic_Regression_Testing.png)

- Để tính độ chính xác, bạn cần **X_Val** và **Y_Val** (dữ liệu để riêng trong quá trình đào tạo, còn gọi là **bộ xác nhận** - validation set) và **Theta** ($\theta$) (bộ thông số tối ưu).
- Các bước thực hiện:
    1.  Tính **hàm sigmoid** (sigmoid function) cho **X_Val** với các tham số **Theta**.
    2.  Đánh giá nếu mỗi giá trị $h(X, \theta)$ $\ge$ **ngưỡng** (threshold), thường là **0.5**.
    3.  Ví dụ: 0.3 $< 0.5$ $\rightarrow$ dự đoán là 0; 0.8 $\ge 0.5$ $\rightarrow$ dự đoán là 1; 0.5 $\ge 0.5$ $\rightarrow$ dự đoán là 1.
    4.  Kết quả là một **vector dự đoán** (prediction vector) gồm các số 0 (tiêu cực) và 1 (tích cực).
> Công thức tính độ chính xác

![M1_W1_14_Logistic_Regression_Testing_02](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_14_Logistic_Regression_Testing_02.png)

- Tính toán độ chính xác:
    1.  So sánh **vector dự đoán** với **giá trị thực sự** (Y_Val). Nếu khớp (ví dụ: dự đoán 0, nhãn 0) = 1; nếu không khớp (ví dụ: dự đoán 1, nhãn 0) = 0.
    2.  Tạo một vector so sánh (ví dụ: [1, 0, ...]).
    3.  Tổng hợp vector so sánh (tính tổng số lần dự đoán đúng).
    4.  Chia tổng đó cho **tổng số m quan sát** từ bộ xác nhận.
- Số liệu này ước tính thời gian **hồi quy logistic** (logistic regression) hoạt động chính xác trên **dữ liệu không nhìn thấy** (unseen data). (Ví dụ: độ chính xác 0.5 = 50%; ví dụ khác: 80%).
- Tổng kết Tuần 1: Bạn đã học cách **xử lý trước** (preprocess) văn bản, **trích xuất đối tượng/đặc điểm** (extract features), **đào tạo** (train) mô hình, và **kiểm tra** (test) mô hình.
- **Bài tập lập trình** (programming exercise) tuần này sẽ giúp thực hiện các khái niệm đã học.
* Có một **video tùy chọn** (optional video) về trực giác (intuition) đằng sau **hàm chi phí** (cost function) cho hồi quy logistic.
- Tuần tới sẽ học về thuật toán **Bayes ngây thơ** (Naive Bayes).

#### 3. Logistic Regression: Cost Function

- Video này là **tùy chọn** và giải thích **trực giác** (intuition) đằng sau **hàm chi phí hồi quy logistic** (logistic regression cost function).

> **Phương trình hàm chi phí** được phân tích thành các thành phần:

![M1_W1_15_Logistic_Regression_Cost_Function](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_15_Logistic_Regression_Cost_Function.png)


![M1_W1_16_Logistic_Regression_Cost_Function_2](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_16_Logistic_Regression_Cost_Function_2.png)

- **Tổng** ($\sum$) trên **m** ví dụ đào tạo, cho thấy chi phí được tính trên mỗi ví dụ.
- Hệ số **-1/m** phía trước có nghĩa là tính giá trị trung bình.
- **Dấu trừ** đảm bảo chi phí tổng thể luôn là một **số dương**.

- Bên trong phương trình có **hai thuật ngữ**:

    + **Thuật ngữ bên trái** ($y^{(i)} \log(h(x^{(i)}, \theta))$):
        * Có liên quan khi **nhãn (y) là 1**.
        * Khi y=1 và dự đoán (h) gần 1, chi phí gần 0 (dự đoán tốt).
        * Khi y=1 và dự đoán (h) gần 0, chi phí tăng lên rất lớn (tiến đến vô cực) (dự đoán sai).
        * Khi y=0, thuật ngữ này bằng 0.
    + **Thuật ngữ bên phải** ($(1-y^{(i)}) \log(1-h(x^{(i)}, \theta))$):
        * Có liên quan khi **nhãn (y) là 0**.
        * Khi y=0 và dự đoán (h) gần 0, chi phí gần 0 (dự đoán tốt).
        * Khi y=0 và dự đoán (h) gần 1, chi phí tăng lên rất lớn (tiến đến vô cực) (dự đoán sai).
        * Khi y=1, thuật ngữ này bằng 0.

- **Đồ thị hàm chi phí** cho một ví dụ duy nhất:
    + Khi **nhãn là 1**, chi phí ($-\log(h)$) gần 0 nếu dự đoán gần 1, và tiến đến **vô cùng** nếu dự đoán gần 0.
    + Khi **nhãn là 0**, chi phí ($-\log(1-h)$) gần 0 nếu dự đoán gần 0, và tiến đến **vô cùng** nếu dự đoán gần 1.
- Bạn đã hiểu cách hàm chi phí hoạt động khi dự đoán đúng và sai.
- Tuần tiếp theo sẽ tìm hiểu về **Naive Bayes**, một thuật toán phân loại khác.

#### 4. Optional Logistic Regression: Gradient

- This is an optional reading where I explain gradient descent in more detail. Remember, previously I gave you the gradient update step, but did not explicitly explain what is going on behind the scenes.

- The general form of gradient descent is defined as:

![M1_W1_17_Gradient_1](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_17_Gradient_1.png)

- For all j. We can work out the derivative part using calculus to get:

![M1_W1_18_Gradient_2](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_18_Gradient_2.png)

- A vectorized implementation is:

![M1_W1_19_Gradient_3](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_19_Gradient_3.png)

#### Partial derivative of J(θ)

First calculate derivative of sigmoid function (it will be useful while finding partial derivative of J(θ)):

![M1_W1_20_Gradient_4](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_20_Gradient_4.png)


- Note that we computed the partial derivative of the sigmoid function. If we were to derive `$h(x^{(i)}, \theta)$` with respect to `$\theta_j$`, you would get `$h(x^{(i)}, \theta) (1 - h(x^{(i)}, \theta)) x_j^{(i)}$`. Note that we used the chain rule there, because we multiply by the derivative of `$\theta^T x^{(i)}$` with respect to `$\theta_j$`. Now we are ready to find out resulting partial derivative:

$$
\frac{\partial}{\partial \theta_j} h(x^{(i)}, \theta) = h(x^{(i)}, \theta) (1 - h(x^{(i)}, \theta)) x_j^{(i)}
$$

![M1_W1_21_Gradient_5](https://github.com/DazielNguyen/NLP301c/blob/main/Module%2001/Image_Module_01/M1_W1/M1_W1_21_Gradient_5.png)

- The vectorized version:

$$
\nabla J(\theta) = \frac{1}{m} \cdot X^T \cdot (H(X, \theta) - Y)
$$

### **Phỏng vấn của bác Andrew Nguyen với ông Chris Manning**

##### Giới thiệu Chris Manning

* Buổi phỏng vấn với **Chris Manning**, nhà nghiên cứu **NLP** (Xử lý ngôn ngữ tự nhiên) được trích dẫn nhiều nhất thế giới.
* Ông là Giáo sư Khoa học Máy tính và Ngôn ngữ học tại **Đại học Stanford** và là Giám đốc của **Stanford AI Lab**.
* Ông nổi tiếng là người đi đầu trong việc áp dụng **học sâu (deep learning)** vào NLP.
* Các nghiên cứu nổi tiếng của ông bao gồm **mạng thần kinh đệ quy cây (tree-recursive neural networks)**, **phân tích tình cảm (sentiment analysis)**, **phân tích phụ thuộc (dependency parsing)**, và thuật toán **GLOVE**.
* Ông từng giảng dạy tại **Carnegie Mellon (CMU)**, **Đại học Sydney**, và **Stanford**.

---

##### Con đường đến với AI và Máy học

* Andrew (người phỏng vấn) hỏi Chris về việc ông bắt đầu với AI như thế nào, đặc biệt là với nền tảng về **ngôn ngữ học (linguistics)**.
* Chris cho biết ban đầu ông không phải là người làm AI. Khi học đại học, ông học khoa học máy tính, toán học và cả ngôn ngữ học.
* Điểm khởi đầu của ông là góc nhìn **khoa học nhận thức (cognitive science)**: Làm thế nào con người học được ngôn ngữ?
* Vào nửa sau thế kỷ 20, tư duy thống trị trong ngôn ngữ học là của **Noam Chomsky**.
* Chomsky cho rằng con người không thể học ngôn ngữ chỉ từ **dữ liệu (data)** mà phải có **máy móc bẩm sinh (innate machinery)** trong não.
* Chris không tin vào điều này và bắt đầu quan tâm đến việc làm thế nào bạn có thể *học* ngôn ngữ, dẫn ông đến **máy học (machine learning)** vào cuối những năm 1980.
* Thời điểm đó, máy học là một lĩnh vực rất nhỏ bé, "thận trọng" bên lề AI, không giống như bây giờ khi AI và máy học "là hai phần ba cùng một thứ".
* Các công trình máy học sơ khai thời đó bao gồm sách của **Jaime Carbonell** và **Tom Mitchell** (từ CMU) và các **thuật toán cây quyết định (decision tree algorithms)** như **ID3**.

---

##### Bối cảnh AI/NLP thời kỳ đầu

* Andrew lưu ý rằng vào thời điểm đó, việc sử dụng dữ liệu để học ngôn ngữ là không trực quan; thay vào đó, mọi người cố gắng **lập trình bằng tay (hand-programming)** một **ngữ pháp CFG (CFG grammar)**.
* Chris đồng ý, cách tiếp cận thống trị trong AI lúc đó là **hệ thống dựa trên tri thức (knowledge-based systems)**, nơi các **kỹ sư kiến thức (knowledge engineers)** mã hóa tri thức chuyên gia.
* Chris đã là "tín đồ thời kỳ đầu" (early believer) vào việc sử dụng máy học cho NLP.

---

##### Transformer và Dịch máy Thống kê (SMT)

* Chris đề cập (có phần lạc đề) rằng **kiến trúc dựa trên biến áp (transformer-based architectures)** hiện đang thống trị, được xây dựng xung quanh ý tưởng về **sự chú ý (attention)**, thứ tạo ra "cấu trúc cây mềm (soft tree structure)".
* Nghiên cứu (ví dụ của **John Hewitt**) cho thấy **mô hình biến áp (transformer models)** có thể học được cấu trúc ngôn ngữ (như **đồng tham khảo (co-reference)** và **cấu trúc ngữ pháp không có ngữ cảnh phân cấp**) chỉ từ dữ liệu văn bản thô.
* Quay lại chủ đề, Andrew đề cập đến công việc có ảnh hưởng của Chris về **dịch máy thống kê (statistical machine translation - SMT)**.
* Khi học sâu nổi lên, nhóm của Chris đã xuất bản một trong những bài báo sớm nhất về **dịch máy thần kinh (neural machine translation - NMT)**, giúp đặt nền tảng (ví dụ: **ma trận chú ý bi-tuyến tính - bilinear attention matrix**).
* Chris xác nhận rằng trong những năm 2000, ông chủ yếu sử dụng **kỹ thuật mô hình xác suất (probabilistic model techniques)**, đây là cách tiếp cận thống trị.
* Mô hình MT thống trị khi đó là **dịch máy dựa trên cụm từ thống kê (statistical phrase-based machine translation)**.
* Các hệ thống này sử dụng **bảng cụm từ (phrase tables)** (xác suất dịch cụm từ) kết hợp với **mô hình ngôn ngữ (language model - LM)**.
* **Mô hình ngôn ngữ** (được định nghĩa là một **phân phối xác suất (probability distribution)** qua chuỗi các từ) là một ý tưởng rất mạnh mẽ trong NLP, được sử dụng cho **hiệu chỉnh chính tả (spell correction)**, **nhận dạng giọng nói (speech recognition)**, và MT.
* Khi Google ra mắt dịch máy dựa trên ML, họ đã sử dụng các hệ thống **dựa trên cụm từ thống kê** này (sau khi ban đầu cấp phép hệ thống **dựa trên quy tắc** của **SYSTRAN**).
* **Franz Och** là người đã lãnh đạo Google mở rộng quy mô các mô hình MT dựa trên cụm từ thống kê trên "tấn dữ liệu", làm cho Google Translate hoạt động "khá hợp lý" trong giai đoạn 2007-2010.

---

##### Giai đoạn chững lại và Bước đột phá của Mạng Nơ-ron

* Trong giai đoạn 2010-2014, **dịch máy dựa trên cụm từ thống kê bị đình trệ (stalled)**.
* Những nỗ lực cải tiến bằng cách sử dụng **cấu trúc ngữ pháp (grammatical structure)** (tức là **hệ thống dịch máy dựa trên cú pháp - syntax-based machine translation systems**) "hầu như không hiệu quả".
* Giải pháp (trớ trêu thay) là "chú ý ít hơn đến cú pháp và chú ý nhiều hơn đến dữ liệu".
* Thành công lớn đầu tiên của các phương pháp thần kinh trong NLP (dựa trên văn bản) là **dịch thuật máy thần kinh (NMT)**.
* (Chris lưu ý rằng **nhận dạng giọng nói (speech recognition)** mới thực sự là thành công lớn *đầu tiên* của mạng nơ-ron đối với ngôn ngữ con người).
* NMT thành công vì có sẵn lượng lớn dữ liệu (văn bản song ngữ).
* Các mô hình ban đầu sử dụng **tái phát mạng thần kinh (recurrent neural networks - RNN)**, được xem như "phiên bản thần kinh liên tục của một mô hình Markov ẩn", và chúng hoạt động tốt mà không cần (lập trình sẵn) cấu trúc ngôn ngữ của con người.




