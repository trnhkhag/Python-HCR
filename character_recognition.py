import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from flask import Flask, request
import cv2
import threading
# from character_recognition import on_button_1_click, predict_character

app = Flask(__name__)

# Biến toàn cục để giao tiếp giữa Flask và OpenCV
capture_requested = False
frame = None

def start_server():
    """
    Khởi động Flask server trong một thread riêng.
    """
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

@app.route('/trigger', methods=['POST'])
def trigger_function():
    """
    Endpoint để nhận trigger từ client.
    Khi nhận yêu cầu POST, đánh dấu yêu cầu chụp ảnh.
    """
    global capture_requested
    capture_requested = True
    print("Chụp ảnh từ camera...")
    return {"status": "success", "message": "Chụp màn hình thành công!"}, 200


# Load model nhận diện ký tự
model = tf.keras.models.load_model("models/hcr_model.h5")
word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',
             16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N', 24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',
             31:'V',32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'c',39:'d',40:'e',41:'f',42:'g',43:'h',44:'i',45:'j',
             46:'k',47:'l',48:'m',49:'n', 50:'o',51:'p',52:'q',53:'r',54:'s',55:'t',56:'u',57:'v',58:'w',59:'x',60:'y',
             61:'z'}
def preprocess_image(image_path):
    """
    Preprocess a single handwritten character image with padding for CNN recognition.
    
    :param image_path: Path to the input image.
    :return: Preprocessed image ready for CNN input (1, 28, 28, 1).
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image (binary inversion)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Find the bounding box around the character
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Select the largest contour (assuming it corresponds to the character)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Crop the character region
        char = thresh[y:y+h, x:x+w]
    else:
        # If no contours are found, use the whole image
        char = thresh

    # Add padding to make the character centered and preserve aspect ratio
    h, w = char.shape
    max_dim = max(h, w)
    padded_char = np.zeros((max_dim, max_dim), dtype=np.uint8)
    pad_y = (max_dim - h) // 2
    pad_x = (max_dim - w) // 2
    padded_char[pad_y:pad_y+h, pad_x:pad_x+w] = char

    # Resize to match model input size (28x28)
    char = cv2.resize(padded_char, (28, 28))

    # Normalize pixel values to [0, 1]
    char = char / 255.0

    # Add a channel dimension for CNN input (28, 28, 1)
    char = np.expand_dims(char, axis=-1)

    # Add batch dimension (1, 28, 28, 1)
    char = np.expand_dims(char, axis=0)

    return char

# Hàm dự đoán ký tự
def predict_character(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_label = word_dict[np.argmax(predictions)]
    return predicted_label

def capture_from_camera():
    # Mở camera
    
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    print("Server 5000 started!")
    url = "http://192.168.1.2:8080/video"
    
    # url = "http://192.168.1.47:8080/photo.jpg"
    cap = cv2.VideoCapture(url)  
    fail_time = 0
    print("Starting video stream...")
    while True:
        ret, frame = cap.read()
        fail_time +=1
        if not ret:
            if fail_time >= 10:
                print("Retry too much. Exiting...")
                exit()
                break
            fail_time += 1
            print("Failed to retrieve frame. Retrying {fail_time}:...")
            continue
        fail_time = 0
        
        # Hiển thị ảnh từ camera
        cv2.imshow("Nhấn 'c' để chụp ảnh, 'q' để thoát", frame)
        
        key = cv2.waitKey(1)
        global capture_requested
        if capture_requested:
            img_path = "captured_image.jpg"
            cv2.imwrite(img_path, frame)
            print("Đã chụp ảnh và lưu tại:", img_path)
            capture_requested = False  # Reset trạng thái sau khi xử lý
            predicted_char = predict_character(img_path)
            print(f"Ký tự dự đoán: {predicted_char}")
            break
        # if key == ord('c'):  # Nhấn 'c' để chụp ảnh
        #     # Lưu ảnh và đóng camera
        #     img_path = "captured_image.jpg"
        #     cv2.imwrite(img_path, frame)
        #     print("Đã chụp ảnh!")
        #     break
        if key == ord('q'):  # Nhấn 'q' để thoát
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
