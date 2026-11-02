# 🖼️ AI Image Classifier Using a Pretrained Model

## 📌 Project Overview

The AI Image Classifier is a beginner-friendly application developed as part of my Artificial Intelligence internship at CodeOrbit Tech.

It uses a pretrained **MobileNetV2** deep learning model to analyze uploaded images and predict their most likely object or animal category.

The application provides an interactive interface where users can upload images and view predicted labels, confidence scores, and alternative predictions.

## ✨ Features

* Upload images in JPG, JPEG, and PNG formats.
* Preview the uploaded image.
* Classify images using a pretrained MobileNetV2 model.
* Display the predicted object or animal category.
* Display an animal type and a possible breed when supported by the predicted label.
* Show the model's prediction confidence score.
* Display the top three predictions.
* Explain how the pretrained model works.
* Simple, interactive web interface.

## 🛠️ Technologies Used

* **Python** — Main programming language.
* **TensorFlow/Keras** — Loads and runs the pretrained model.
* **MobileNetV2** — Image classification model pretrained on ImageNet.
* **Streamlit** — Creates the web application interface.
* **NumPy** — Processes image data as numerical arrays.
* **Pillow** — Loads and processes uploaded images.

## ⚙️ How It Works

1. The user uploads an image.
2. The application resizes the image to 224 × 224 pixels.
3. The image is converted into a numerical array and preprocessed.
4. MobileNetV2 analyzes the image and generates prediction scores.
5. The application displays the top predicted categories and their scores.
6. The application uses the predicted label to infer an animal type and possible breed when applicable.

### What Is MobileNetV2?

MobileNetV2 is a convolutional neural network designed for efficient image recognition.

It is pretrained on the ImageNet dataset, which contains many object and animal categories.

Instead of training a model from scratch, this project uses the pretrained model to classify images. This makes the project simpler and more suitable for beginners.

## 💻 Installation and Setup

### Prerequisites

* Python installed on your computer.
* Git, if you want to clone the repository.
* An internet connection for downloading the pretrained model weights if they are not already available.

### Step 1: Clone the Repository

Replace the example repository URL below with your actual GitHub repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd AI-Image-Classifier
```

If your repository uses a different folder name, use that name instead.

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install the Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## 🧪 Testing

The application can be tested using different images, such as:

* Dogs
* Cats
* Birds
* Cars
* Other recognizable objects

The predicted results depend on the uploaded image and the categories supported by the pretrained model.

## ⚠️ Limitations

* Predictions may be incorrect for unclear or unusual images.
* The model recognizes categories learned during ImageNet training.
* Animal type and breed are inferred from the predicted label; they are not independently verified.
* A specific breed may not always be available or correctly identified.
* Confidence scores represent model prediction scores, not guaranteed accuracy.

## 🎯 Project Objective

The objective of this project is to demonstrate how a pretrained deep learning model can be integrated into a simple application to perform image classification.

It provides practical experience with Python, deep learning, image preprocessing, and building an interactive AI application.

## 👨‍💻 Internship

Developed as part of the **Artificial Intelligence Internship at CodeOrbit Tech**.

## 🔗 GitHub Repository

https://github.com/Praneeth-mikklenani/AI-Image-Classifier

