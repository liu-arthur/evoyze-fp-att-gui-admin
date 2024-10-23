# Project Setup Instructions

This document provides instructions for setting up a Python virtual environment, installing the required packages, and packaging the application with PyInstaller.

## Prerequisites

- Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
- Make sure to add Python to your PATH during installation.

### Build Tools for Visual Studio

Make sure the Build Tools for Visual Studio are installed on your PC. If not, you can download them from the following link (note that the link may change over time):

[Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)

## Cloning the Repository

To get started, clone the repository from GitHub using the following command:

```bash
git clone https://github.com/liu-arthur/evoyze-fp-att-gui-admin.git
```

## Steps to Set Up the Project

### 1. Create a Virtual Environment

Open your command prompt (Windows) or terminal (macOS/Linux) and navigate to the directory where you want to create your virtual environment. Run the following command:

```bash
python -m venv venv
```

Replace `venv` with the desired name for your virtual environment.

### 2. Activate the Virtual Environment

#### On Windows:
```bash
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
source venv/bin/activate
```

You should see the name of your virtual environment in your terminal prompt, indicating that it is active.

### 3. Install Required Packages

Ensure you are in the directory containing your `requirements.txt` file. Then run:

```bash
pip install -r requirements.txt
```

This command will install all the packages specified in `requirements.txt`.

### 4. Update `gui_script.py`

Make sure to update the virtual environemnt directory path in `gui_script.py`, as it requires the folder name of the virtual environment. 

## Packaging with PyInstaller

If you want to package your application using PyInstaller, ensure that you have it installed:

```bash
pip install pyinstaller
```

You can then run PyInstaller with the desired options:

```bash
pyinstaller --onefile --noconsole --windowed --icon=logo.ico --distpath /path/to/dist/ gui_script.py
```

Replace or remove `/path/to/dist/`. By default, PyInstaller creates a dist folder in the current working directory to store the output files.

## Notes

- If you encounter any issues while installing packages, ensure that pip is up to date by running:

```bash
pip install --upgrade pip
```