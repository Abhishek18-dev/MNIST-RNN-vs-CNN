import streamlit as st
import torch
import numpy as np

from PIL import Image, ImageOps
from torchvision import transforms

from architecture import CNNModel, RNNModel


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="MNIST Handwritten Digit Classifier",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MNIST Handwritten Digit Classifier")
st.markdown(
    "Compare predictions using **CNN** and **RNN** models trained on the **MNIST Dataset**."
)

# ---------------- DEVICE ---------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LOAD MODELS ---------------- #

@st.cache_resource
def load_models():

    cnn = torch.load(
        "models/mnist_cnn.pth",
        map_location=device,
        weights_only=False
    )

    cnn.eval()

    rnn = torch.load(
        "models/mnist_rnn.pth",
        map_location=device,
        weights_only=False
    )

    rnn.eval()

    return cnn, rnn


cnn_model, rnn_model = load_models()


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("⚙ Settings")

model_name = st.sidebar.radio(
    "Choose Model",
    ["CNN", "RNN"]
)

input_type = st.sidebar.radio(
    "Image Type",
    [
        "MNIST Image",
        "Handwritten Paper"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Model Accuracy")

st.sidebar.success("CNN : 98.96%")
st.sidebar.info("RNN : 96.95%")

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset")

st.sidebar.write("""
**MNIST**

• 70,000 Images

• 10 Classes (0–9)

• 28 × 28 Grayscale
""")


# ---------------- ABOUT ---------------- #

with st.expander("About this Project"):

    st.write("""
This project compares two Deep Learning models trained on the MNIST handwritten digit dataset.

Models Used:

• Convolutional Neural Network (CNN)

• Recurrent Neural Network (RNN)

Framework:

• PyTorch

Frontend:

• Streamlit
""")


# ---------------- TRANSFORM ---------------- #

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ---------------- PREPROCESS : MNIST ---------------- #

def preprocess_mnist(image):
    image = image.convert("L")

    # Resize ONLY if image isn't already 28x28
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.NEAREST)

    processed_image = image.copy()

    tensor = transform(image)

    return tensor.unsqueeze(0), processed_image


# ---------------- PREPROCESS : HANDWRITTEN ---------------- #
def preprocess_handwritten(image):

    # Convert to grayscale
    image = image.convert("L")

    # Convert to numpy
    img = np.array(image)

    # If background is dark, invert
    if img.mean() < 127:
        img = 255 - img

    # Threshold
    threshold = img.mean() + img.std() * 0.25

    img = np.where(img > threshold, 255, 0).astype(np.uint8)

    # Invert to MNIST style
    img = 255 - img

    # Find bounding box of digit
    coords = np.argwhere(img > 0)

    if len(coords) == 0:
        img = np.zeros((28, 28), dtype=np.uint8)
    else:

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        img = img[y0:y1, x0:x1]

        h, w = img.shape

        # Maintain aspect ratio
        if h > w:
            new_h = 20
            new_w = max(1, int(w * 20 / h))
        else:
            new_w = 20
            new_h = max(1, int(h * 20 / w))

        img = Image.fromarray(img)
        img = img.resize((new_w, new_h), Image.Resampling.NEAREST)

        canvas = Image.new("L", (28, 28), 0)

        left = (28 - new_w) // 2
        top = (28 - new_h) // 2

        canvas.paste(img, (left, top))

        img = np.array(canvas)

    processed_image = Image.fromarray(img)

    tensor = transform(processed_image)

    return tensor.unsqueeze(0), processed_image


# ---------------- FILE UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:

    original = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")
        st.image(original, use_container_width=True)

    if input_type == "MNIST Image":
        input_tensor, processed_image = preprocess_mnist(original)
    else:
        input_tensor, processed_image = preprocess_handwritten(original)

    input_tensor = input_tensor.to(device)

    with col2:
        st.subheader("Processed Image")
        st.image(
            processed_image,
            use_container_width=True,
            width=400,
            clamp=True,
            caption="28×28 Processed"
        )

    # Prediction

    with torch.inference_mode():

        if model_name == "CNN":
            outputs = cnn_model(input_tensor)
        else:
            outputs = rnn_model(input_tensor.squeeze(1))

    # st.write(input_tensor.shape)
    probabilities = torch.softmax(outputs, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)



    # ---------------- RESULTS ---------------- #

    st.markdown("---")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            label="🎯 Predicted Digit",
            value=str(prediction.item())
        )

    with metric2:
        st.metric(
            label="📊 Confidence",
            value=f"{confidence.item()*100:.2f}%"
        )

    if confidence >= 0.90:
        st.success("Very High Confidence")

    elif confidence >= 0.70:
        st.warning("Moderate Confidence")

    else:
        st.error("Low Confidence")

    # ---------------- TOP 3 ---------------- #

    top_prob, top_class = torch.topk(probabilities, 3)

    st.subheader("🏆 Top 3 Predictions")

    for i in range(3):

        digit = top_class[0][i].item()
        prob = top_prob[0][i].item()

        st.write(f"**{i+1}. Digit {digit}** — {prob*100:.2f}%")

    # ---------------- CLASS PROBABILITIES ---------------- #

    st.subheader("📈 Probability Distribution")

    for digit in range(10):

        prob = probabilities[0][digit].item()

        left, right = st.columns([1, 6])

        with left:
            st.write(f"**{digit}**")

        with right:
            st.progress(float(prob), text=f"{prob*100:.2f}%")

    # ---------------- MODEL INFO ---------------- #

    st.markdown("---")

    with st.expander("ℹ Model Information"):

        if model_name == "CNN":

            st.success("""
    **Convolutional Neural Network**

    • Accuracy : 98.96%

    • Best suited for image data

    • Learns spatial features using convolution filters

    • Faster and more accurate for MNIST
    """)

        else:

            st.info("""
    **Recurrent Neural Network**

    • Accuracy : 96.95%

    • Treats image as a sequence of rows

    • Good for sequential data

    • Used here for comparison with CNN
    """)

    # ---------------- FOOTER ---------------- #

    st.markdown("---")

    st.caption(
        "Developed by Abhishek | PyTorch • Streamlit • MNIST Digit Classifier"
    )