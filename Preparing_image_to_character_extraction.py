import train_model

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # Using PIL to open and process images


class ImageSelectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Selector")

        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        self.select_button = tk.Button(self.master, text="Select Image", command=self.load_image)
        self.select_button.pack()

        self.selected_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.selected_image = Image.fromarray(train_model.project(file_path))

            self.selected_image = self.selected_image.resize((1000, 1000))  # Resize for display
            self.display_image()

    def display_image(self):
        if self.selected_image:
            # Convert the Image object to PhotoImage object
            photo_image = ImageTk.PhotoImage(self.selected_image)
            self.image_label.config(image=photo_image)
            self.image_label.image = photo_image  # Keeping reference to avoid garbage collection


def main():
    root = tk.Tk()
    app = ImageSelectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# train_model.project("/Users/eliyahunezri/Desktop/AI_applications/test1.jpeg")