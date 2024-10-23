import os
import shutil  # New import for copying files
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk  # Pillow library for handling images
import subprocess

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

        # Bind events to show/hide tooltip
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window is not None:
            return  # Tooltip is already visible

        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Remove window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip_window, text=self.text, background="yellow", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

# Function to show instructions
def show_instructions():
    instructions_window = Toplevel(window)
    instructions_window.title("Instructions")
    instructions_window.geometry("600x300")
    instructions_window.resizable(False, False)
    
    # Add instructions text
    instructions_text = """
    How to Use This System:

    1. Recognize Faces: After using "Create New" or "Train Photo," 
        this option ensures the face is recognized by the system.
    2. Create New: Click this button to create a new entry in the system.
    3. Upload Photo: Ensure that you upload only one person's photo.
    4. Train Photo: Use this button to train the system with a photo after uploading it.
    5. Delete Photo: Click this button to remove the photo from the system.


    Follow the prompts on the screen to complete each task.
    """
    
    instructions_label = tk.Label(instructions_window, text=instructions_text, font=("Arial", 12), justify="left")
    instructions_label.pack(pady=20)

# Function to run recognize face script
def recognize_face():
    try:
        subprocess.run([r'venv\Scripts\python.exe', 'recognize_face_v1.3.py'], check=True)
        messagebox.showinfo("Success", "Face recognition completed!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error occurred: {e}")

# Function to run the training script
def train_faces():
    try:
        subprocess.run([r'venv\Scripts\python.exe', 'train_v1.3.py'], check=True)
        messagebox.showinfo("Success", "Done")
    except KeyError as e:
        messagebox.showerror("Error", str(e))
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error occurred: {e}\nReturn Code: {e.returncode}\nOutput: {e.output}")
    except FileNotFoundError:
        messagebox.showerror("Error", "Python executable not found. Please check the virtual environment path.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Function to train from phone
def train_from_phone():
    try:
        subprocess.run([r'venv\Scripts\python.exe', 'train_from_smart_phone1.4.py'], check=True)
        messagebox.showinfo("Success", "Training completed.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error occurred: {e}\nReturn Code: {e.returncode}\nOutput: {e.output}")
    except FileNotFoundError:
        messagebox.showerror("Error", "Python executable not found. Please check the virtual environment path.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Function to add photos to the phone folder (copying instead of moving)
def add_photos():
    files = filedialog.askopenfilenames(title="Select Photos", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if files:
        dest_folder = os.path.join(os.getcwd(), "phone")
        os.makedirs(dest_folder, exist_ok=True)
        for file in files:
            try:
                dest_path = os.path.join(dest_folder, os.path.basename(file))
                shutil.copy(file, dest_path)  # Now using shutil.copy to copy the files instead of os.rename
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy {file}: {e}")
        messagebox.showinfo("Success", "Photos copied to 'phone' folder successfully!")

# Function to clean the phone folder
def clean_phone_folder():
    phone_folder = os.path.join(os.getcwd(), "phone")
    if os.path.exists(phone_folder):
        for file_name in os.listdir(phone_folder):
            file_path = os.path.join(phone_folder, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete {file_name}: {e}")
        messagebox.showinfo("Success", "Phone folder cleaned successfully!")
    else:
        messagebox.showinfo("Info", "The phone folder does not exist.")

# Create main application window
window = tk.Tk()
window.title("FAST Attendance Facial Registration")
window.geometry("500x520")
window.configure(bg="#f0f0f0")
window.resizable(False, False)

# Add a logo image (if you have one)
logo_image = Image.open("logo.jpg")
logo_image = logo_image.resize((150, 150), Image.Resampling.LANCZOS)
logo_photo = ImageTk.PhotoImage(logo_image)

logo_label = tk.Label(window, image=logo_photo, bg="#f0f0f0")
logo_label.pack(pady=10)

# Add title with a nice font
title_label = tk.Label(window, text="FAST Attendance Facial Registration", font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#333333")
title_label.pack(pady=10)

# Frame for buttons to group them nicely
button_frame = tk.Frame(window, bg="#f0f0f0")
button_frame.pack(pady=20)

# Define the same width and height for all buttons
button_width = 20
button_height = 2

# Create buttons with a nicer layout and better colors
recognize_button = tk.Button(button_frame, text="Recognize Faces", command=recognize_face, font=("Arial", 12), bg="#4CAF50", fg="white", width=button_width, height=button_height)
recognize_button.grid(row=0, column=0, padx=10, pady=10)

train_button = tk.Button(button_frame, text="Create New", command=train_faces, font=("Arial", 12), bg="#2196F3", fg="white", width=button_width, height=button_height)
train_button.grid(row=1, column=0, padx=10, pady=10)

train_phone_button = tk.Button(button_frame, text="Train Photo", command=train_from_phone, font=("Arial", 12), bg="#FF9800", fg="white", width=button_width, height=button_height)
train_phone_button.grid(row=1, column=1, padx=10, pady=10)

add_photos_button = tk.Button(button_frame, text="Upload Photo", command=add_photos, font=("Arial", 12), bg="#9C27B0", fg="white", width=button_width, height=button_height)
add_photos_button.grid(row=0, column=1, padx=10, pady=10)

# Clean phone folder button
clean_phone_button = tk.Button(button_frame, text="Delete photo", command=clean_phone_folder, font=("Arial", 12), bg="#F44336", fg="white", width=button_width, height=button_height)
clean_phone_button.grid(row=2, column=1, padx=10, pady=10)

# Create the tooltip
tooltip = Tooltip(add_photos_button, "Please upload and ensure only 1 person's photo")

# Add an instructions button with a book icon
book_image = Image.open("book.png")
book_image = book_image.resize((30, 30), Image.Resampling.LANCZOS)  # Resize the image (optional)
book_photo = ImageTk.PhotoImage(book_image)

# Add an instructions button with a book icon
instructions_button = tk.Button(window, image=book_photo, command=show_instructions, bg="#f0f0f0", borderwidth=0)
instructions_button.place(relx=1.0, rely=0.0, x=-10, y=10, anchor='ne')  # Position at the top right


# Footer
footer_label = tk.Label(window, text="© 2024 The Everly Group", font=("Arial", 10), bg="#f0f0f0", fg="#888888")
footer_label.pack(side="bottom", pady=10)

# Start the GUI loop
window.mainloop()
