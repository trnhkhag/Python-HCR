import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# Load model nhận diện ký tự
model = tf.keras.models.load_model("HCR_English.h5")
word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',
             16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N', 24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',
             31:'V',32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'c',39:'d',40:'e',41:'f',42:'g',43:'h',44:'i',45:'j',
             46:'k',47:'l',48:'m',49:'n', 50:'o',51:'p',52:'q',53:'r',54:'s',55:'t',56:'u',57:'v',58:'w',59:'x',60:'y',
             61:'z'}

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    """
    Tiền xử lý ảnh viết tay để chuẩn bị đầu vào cho mô hình.
    - Đọc ảnh
    - Chuyển đổi sang grayscale
    - Resize về 28x28
    - Chuyển đổi ảnh sang binary (dùng Otsu Thresholding)
    - Invert background
    - Flatten ảnh và chuẩn hóa giá trị pixel về [0, 1]
    """
    # Đọc ảnh dưới dạng grayscale
    # img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.imread(image_path, 0)

    # Resize ảnh về kích thước 28x28
    img_gray = cv2.resize(img_gray, (28, 28))
    
    # Chuyển đổi ảnh grayscale sang binary (Otsu Thresholding)
    ret, img_binary=cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Đảo ngược màu sắc (đổi trắng thành đen và ngược lại)
    img = cv2.bitwise_not(img_binary)

    # Flatten ảnh
    value = img.flatten()
    fl_img = np.array(value)
    
    # Flatten ảnh và chuẩn hóa giá trị pixel
    fl_img = fl_img.astype('float32')
    
    # Thêm trục cho phù hợp với đầu vào của mô hình
    img = np.reshape(fl_img, (1, 28, 28, 1))
    return img

# Hàm dự đoán ký tự
def predict_character(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_label = word_dict[np.argmax(predictions)]
    return predicted_label

def capture_from_camera():
    # Mở camera
    cap = cv2.VideoCapture(1)  # 0 là camera mặc định
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập camera")
            break
        
        # Hiển thị ảnh từ camera
        cv2.imshow("Nhấn 'c' để chụp ảnh, 'q' để thoát", frame)
        
        key = cv2.waitKey(1)
        if key == ord('c'):  # Nhấn 'c' để chụp ảnh
            # Lưu ảnh và đóng camera
            img_path = "captured_image.jpg"
            cv2.imwrite(img_path, frame)
            print("Đã chụp ảnh!")
            break
        elif key == ord('q'):  # Nhấn 'q' để thoát
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return img_path

def on_button_1_click(canvas, image_3, result_text):
    """
    Xử lý sự kiện khi nhấn nút "Chụp ảnh".
    - Chụp ảnh từ camera.
    - Hiển thị ảnh trong `image_3`.
    - Hiển thị kết quả nhận diện ký tự trong `result_text`.
    
    Args:
        canvas: Canvas của giao diện GUI.
        image_3: ID của vùng ảnh trong canvas.
        result_text: ID của vùng hiển thị kết quả.
    """
    # Chụp ảnh từ camera
    img_path = capture_from_camera()
    if img_path:
        # Hiển thị ảnh trong image_image_3
        img = Image.open(img_path)
        img = img.resize((200, 200))  # Resize ảnh cho phù hợp
        img = ImageTk.PhotoImage(img)
        canvas.itemconfig(image_3, image=img)
        canvas.image = img  # Lưu trữ tham chiếu để không bị garbage collected

        # Dự đoán ký tự và hiển thị kết quả
        predicted_char = predict_character(img_path)
        canvas.itemconfig(result_text, text=predicted_char)

# Tích hợp nhận diện ký tự từ camera
if __name__ == "__main__":
    img_path = capture_from_camera()
    if img_path:
        predicted_char = predict_character(img_path)
        print(f"Ký tự dự đoán: {predicted_char}")
