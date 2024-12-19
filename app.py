
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from character_recognition import on_button_1_click


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"F:\Desktop\XuLyAnh\Python-HCR\assets\frame0")



def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()

window.geometry("500x800")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 800,
    width = 500,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    250.0,
    40.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    250.0,
    370.0,
    image=image_image_2
)

canvas.create_rectangle(
    35.0,
    445.0,
    315.0,
    765.0,
    fill="#4076FF",
    outline="")

canvas.create_text(
    140.0,
    450.0,
    anchor="nw",
    text="Image",
    fill="#FFFFFF",
    font=("Inter Bold", 24 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: on_button_1_click(canvas, image_3, result_text),
    relief="flat"
)
button_1.place(
    x=35.0,
    y=105.0,
    width=200.0,
    height=200.0
)

image_image_3 = canvas.create_image(
    175.0,
    625.0,
    image=None  # Để trống ảnh ban đầu
)

result_text = canvas.create_text(
    370.0,
    530.0,
    anchor="nw",
    text="",
    fill="#FFFFFF",
    font=("Inter Bold", 24 * -1)
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=265.0,
    y=105.0,
    width=200.0,
    height=200.0
)

canvas.create_rectangle(
    345.0,
    525.0,
    465.0,
    685.0,
    fill="#00B23B",
    outline="")

canvas.create_text(
    370.0,
    530.0,
    anchor="nw",
    text="Result",
    fill="#FFFFFF",
    font=("Inter Bold", 24 * -1)
)

result_text = canvas.create_text(
    398.0,
    598.0,
    anchor="nw",
    text="result here!",  # Nội dung mặc định
    fill="#FFFFFF",
    font=("Inter Bold", 24 * -1)
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    175.0,
    625.0,
    image=image_image_3
)
window.resizable(False, False)
window.mainloop()
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("500x800")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 800,
    width = 500,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    250.0,
    40.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    250.0,
    370.0,
    image=image_image_2
)

canvas.create_rectangle(
    35.0,
    445.0,
    315.0,
    765.0,
    fill="#4076FF",
    outline="")

canvas.create_text(
    140.0,
    450.0,
    anchor="nw",
    text="Image",
    fill="#FFFFFF",
    font=("Inter Bold", 24 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))

button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)

button_1.place(
    x=35.0,
    y=105.0,
    width=200.0,
    height=200.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))

button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=265.0,
    y=105.0,
    width=200.0,
    height=200.0
)

canvas.create_rectangle(
    345.0,
    525.0,
    465.0,
    685.0,
    fill="#00B23B",
    outline="")

canvas.create_text(
    370.0,
    530.0,
    anchor="nw",
    text="Result",
    fill="#FFFFFF",
    font=("Inter Bold", 24 * -1)
)

canvas.create_text(
    398.0,
    598.0,
    anchor="nw",
    text="A",
    fill="#FFFFFF",
    font=("Inter Bold", 24 * -1)
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    175.0,
    625.0,
    image=image_image_3
)
window.resizable(False, False)
window.mainloop()
