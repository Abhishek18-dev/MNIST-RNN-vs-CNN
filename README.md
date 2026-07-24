# 🧠 MNIST CNN vs RNN - Handwritten Digit Recognition

An end-to-end Deep Learning project that compares **Convolutional Neural Networks (CNNs)** and **Recurrent Neural Networks (RNNs)** for handwritten digit recognition using the **MNIST dataset**, along with an interactive **Streamlit web application** for real-time predictions.

> This project was built to understand **why CNNs outperform RNNs on image classification tasks**, while also learning the complete workflow from model training to deployment.

---

## 🚀 Live Demo

🔗 **http://mnist-cnn-rnn.streamlit.app/**

---

## 📌 Project Overview

This project explores two different Deep Learning architectures on the same dataset:

- **CNN** – Designed for extracting spatial features from images.
- **RNN** – Originally designed for sequential data, adapted here for image classification by treating each image as a sequence.

The project also includes a Streamlit application where users can:

- Upload MNIST digit images
- Upload handwritten digit images
- Compare CNN and RNN predictions
- View prediction confidence
- Explore Top-3 predicted classes

---

## 🎯 Why I Built This

Instead of building another "MNIST classifier", I wanted to understand the engineering and architectural differences between CNNs and RNNs on the same computer vision task.

This project also gave me an opportunity to learn:

- PyTorch model development
- Model saving and loading
- Image preprocessing
- Streamlit application development
- Deploying machine learning applications

---

## ✨ Features

- CNN and RNN model comparison
- MNIST image prediction
- Handwritten digit prediction
- Confidence score visualization
- Top-3 predictions
- Interactive Streamlit interface
- Modular project structure

---

## 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pillow (PIL)
- Matplotlib

---

## 📁 Project Structure

```text
MNIST-RNN-vs-CNN/
│
├── app.py
├── architecture.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── mnist_cnn.pth
│   └── mnist_rnn.pth
│
├── notebooks/
│   ├── mnist_cnn.ipynb
│   └── mnist_rnn.ipynb
│
└── images/
```

---

## 🧠 Model Architectures

### CNN

- Convolution Layers
- ReLU Activation
- Max Pooling
- Fully Connected Layers

CNN preserves spatial information and is naturally suited for image classification.

---

### RNN

The 28×28 image is treated as a sequence of rows and processed using an RNN before classification.

Although RNNs can learn sequential patterns, they are not specifically designed for preserving spatial relationships in images.

---

## 📊 Key Observations

During experimentation, I observed that:

- CNN consistently performed better than RNN on image classification.
- CNN effectively captures local spatial features using convolution operations.
- RNN loses important spatial relationships because it processes images sequentially.
- Even small differences in preprocessing can significantly affect predictions.

---

## ⚙️ Engineering Challenges

Some practical challenges I encountered while building this project included:

- Model serialization and loading
- Image preprocessing consistency
- Handling handwritten image normalization
- Integrating trained models into a Streamlit application
- Organizing the project into reusable modules

Working through these challenges helped me understand that building an ML application involves much more than just training a model.

---

## 🤝 AI Collaboration

AI was used as a collaborative engineering assistant throughout this project.

It helped with:

- Brainstorming implementation approaches
- Explaining Deep Learning and PyTorch concepts
- Debugging critical runtime errors
- Reviewing preprocessing and inference logic
- Improving documentation
- Refining the Streamlit interface

All implementation decisions, project integration, testing, debugging, repository organization, and final validation were performed by me. AI suggestions were reviewed, modified where necessary, and verified before being incorporated into the project.

---

## 📚 What I Learned

This project taught me much more than building a digit classifier.

Some key lessons include:

- CNNs are naturally better suited for image classification because they preserve spatial information.
- Model training is only one part of an ML project; preprocessing and inference pipelines are equally important.
- Proper model saving/loading is essential for deployment.
- Small preprocessing differences can noticeably affect prediction quality.
- Building interactive ML applications requires combining machine learning with software engineering practices.

---

## 🔮 Future Improvements

Possible future enhancements include:

- Better handwritten digit preprocessing
- Improved handwritten prediction accuracy
- Confusion matrix and evaluation dashboard
- Docker containerization
- Cloud deployment improvements
- Support for additional handwritten datasets (e.g., EMNIST)
- Extending the application to compare more deep learning architectures

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/MNIST-RNN-vs-CNN.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🙏 Acknowledgements

- MNIST Dataset
- PyTorch
- Streamlit

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ If you found this project interesting, consider giving it a star!