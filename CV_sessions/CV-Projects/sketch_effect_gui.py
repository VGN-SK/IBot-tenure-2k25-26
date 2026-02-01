import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# ===================== EFFECTS ===================== #

def pencil_sketch(image, blur_kernel=99, noise_std=5):

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray

    blur_inv_gray = cv2.GaussianBlur(inv_gray,(blur_kernel,blur_kernel),0)
    inv_blur_inv_gray = 255 - blur_inv_gray

    dodge = (gray/(inv_blur_inv_gray + 1e-6))*256
    sketch = np.clip(dodge,0,255).astype(np.uint8)

    noise = np.random.normal(0, noise_std, sketch.shape).astype(np.int16)
    sketch = np.clip(sketch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return sketch


def color_sketch(image, saturation_factor=0.8, blur_kernel=99, noise_std=5):

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hue,sat,val = cv2.split(image_hsv)

    inv_gray = 255 - val
    blur_inv_gray = cv2.GaussianBlur(inv_gray,(blur_kernel,blur_kernel),0)
    inv_blur_inv_gray = 255 - blur_inv_gray

    dodge = (val/(inv_blur_inv_gray + 1e-6))*256
    sketch_val = np.clip(dodge,0,255).astype(np.uint8)

    noise = np.random.normal(0,noise_std,sketch_val.shape)
    sketch_val = np.clip(sketch_val + noise,0,255).astype(np.uint8)

    final_sat = np.clip(sat.astype(np.float32) * saturation_factor,0, 255).astype(np.uint8)

    final_image_hsv = cv2.merge([hue,final_sat.astype(np.uint8),sketch_val])
    final_image = cv2.cvtColor(final_image_hsv,cv2.COLOR_HSV2BGR)

    return final_image

# ===================== GUI APP ===================== #

class SketchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gallery V1.0")
        self.root.geometry("1200x900")

        self.cap = None
        self.mode = None
        self.paused = False
        self.static_image = None
        self.prev_vid_path = None
        self.allow_save = False

        self.build_ui()
        self.update_frame()

    # ---------- UI ---------- #
    def build_ui(self):
        global control
        control = tk.Frame(self.root)
        control.pack(side="left", fill="y", padx=10)

        self.effect = tk.StringVar(value="original")

        ttk.Label(control, text="Effect").pack()
        ttk.Radiobutton(control, text="Original", variable=self.effect, value="original").pack(anchor="w")
        ttk.Radiobutton(control, text="Pencil Sketch", variable=self.effect, value="pencil").pack(anchor="w")
        ttk.Radiobutton(control, text="Color Sketch", variable=self.effect, value="color").pack(anchor="w")

        self.blur = tk.Scale(control, from_=1, to=201, resolution=2, orient="horizontal", label="Blur Kernel")
        self.blur.set(99)
        self.blur.pack(fill="x")

        self.noise = tk.Scale(control, from_=0, to=20, orient="horizontal", label="Noise")
        self.noise.set(5)
        self.noise.pack(fill="x")

        self.sat = tk.Scale(control, from_=0.1, to=1.5, resolution=0.1, orient="horizontal", label="Saturation")
        self.sat.set(0.8)
        self.sat.pack(fill="x")

        ttk.Button(control, text="Open Image", command=self.open_image).pack(fill="x", pady=5)
        ttk.Button(control, text="Open Video", command=self.open_video).pack(fill="x", pady=5)
        ttk.Button(control, text="Live Camera", command=self.open_camera).pack(fill="x", pady=5)
        ttk.Button(control, text="Play / Pause", command=self.toggle_pause).pack(fill="x", pady=5)
        ttk.Button(control, text="Quit", command=self.quit).pack(fill="x", pady=5)

        self.save_button = ttk.Button(control, text="Save Image", command=self.save_current_image)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(expand=True)

    # ---------- SOURCE LOADERS ---------- #
    def open_image(self):
        path = filedialog.askopenfilename()
        if not path or not path.strip().endswith(('.jpg','.jpeg','.png')):
            return
        self.prev_vid_path = None
        self.release_cap()
        self.static_image = cv2.imread(path)
        self.mode = "image"
        self.allow_save = True

    def open_video(self):
        path = filedialog.askopenfilename()
        self.prev_vid_path = path
        if not path or not path.strip().endswith(('.mp4','.mkv','.avi')):
            return
        self.release_cap()
        self.cap = cv2.VideoCapture(path)
        self.mode = "video"
        self.paused = False

    def replay_prev_video(self) :
        self.release_cap()
        self.cap = cv2.VideoCapture(self.prev_vid_path)
        self.paused = False

    def save_current_image(self):
        path = filedialog.asksaveasfilename(
            title="Save Image",
            initialdir=os.getcwd(),
            initialfile="sketch.png",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
            ],
            parent=self.root
        )

        if path:
            cv2.imwrite(path, self.static_image)

    def open_camera(self):
        self.prev_vid_path = None
        self.release_cap()
        self.cap = cv2.VideoCapture(0)
        self.mode = "live"
        self.paused = False

    # ---------- CORE LOOP ---------- #
    def update_frame(self):
        frame = None

        if self.mode in ("video", "live") and self.cap and not self.paused:
            ret, frame = self.cap.read()
            if not ret:
                self.release_cap()

        elif self.mode == "image":
            frame = self.static_image

        if frame is not None:
            processed = self.apply_effect(frame)
            self.show_frame(processed)
        elif self.mode == 'video' :
            self.replay_prev_video()

        # UI - Update
        if self.mode == 'image' and self.allow_save :
            self.save_button.pack(fill="x", pady=5)
            self.allow_save = False

        if self.mode != 'image' :
            self.save_button.pack_forget()

        self.root.after(30, self.update_frame)

    # ---------- PROCESS ---------- #
    def apply_effect(self, frame):
        if self.effect.get() == "pencil":
            return pencil_sketch(
                frame,
                blur_kernel=self.blur.get(),
                noise_std=self.noise.get()
            )
        elif self.effect.get() == "color":
            return color_sketch(
                frame,
                saturation_factor =self.sat.get(),
                blur_kernel=self.blur.get(),
                noise_std=self.noise.get()
            )
        else :
            return frame

    # ---------- DISPLAY ---------- #
    def show_frame(self, frame):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(img)

        self.image_label.img = img
        self.image_label.config(image=img)

    # ---------- UTIL ---------- #
    def toggle_pause(self):
        self.paused = not self.paused

    def release_cap(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def quit(self):
        self.release_cap()
        self.root.destroy()


# ===================== RUN ===================== #

if __name__ == "__main__":
    root = tk.Tk()
    app = SketchApp(root)
    root.mainloop()