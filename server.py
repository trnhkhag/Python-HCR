from flask import Flask, request

from character_recognition import on_button_1_click
app = Flask(__name__)

# Hàm Python sẽ được kích hoạt
def my_python_function():
    print("Triggered by Tasker!")
    # Thực hiện các tác vụ tại đây
    return "Function executed!"

# Endpoint để nhận trigger
@app.route('/trigger', methods=['POST'])
def trigger_function():
    response = my_python_function()
    # on_button_1_click("captured_image.jpg", "image_3", "result_text")
    return {"status": "success", "message": response}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
