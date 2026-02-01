from flask import *
import cv2
import PIL.Image as Image
import io
import base64

import torch
import torchvision.transforms as T

from model import MyViT

# ================= CONFIG =================
USE_WEBCAM = True   # True → webcam, False → mp4
VIDEO_PATH = "Terrain-Recognition-using-Vision-Transformer/F4.mp4"
# ==========================================


class readCamera:
    def __init__(self, use_webcam=False, video_path=None):

        if use_webcam:
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video source")

        print("Camera opened:", self.cap.isOpened())

    def grab(self):
        ret, frame = self.cap.read()

        # Loop video if it ends
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        ret, self.buffer = cv2.imencode(".jpg", frame)
        self.blob = base64.b64encode(self.buffer).decode("utf-8")

    def getTensor(self):
        image = Image.open(io.BytesIO(self.buffer))
        image = T.Resize((256, 256))(image)
        image = T.ToTensor()(image)
        return image.unsqueeze(0)

    def close(self):
        self.cap.release()


# ================= FLASK APP =================
app = Flask(__name__, template_folder="template", static_folder="static")

cam = None


def get_camera():
    global cam
    if cam is None:
        cam = readCamera(
            use_webcam=USE_WEBCAM,
            video_path=VIDEO_PATH
        )
    return cam


# ================= MODEL (LOAD ONCE) =================
device = torch.device("cpu")

model = MyViT(
    (3, 256, 256),
    n_patches=16,
    n_blocks=4,
    hidden_d=64,
    n_heads=8,
    out_d=3
).to(device)

state = torch.load(
    "./Terrain-Recognition-using-Vision-Transformer/model.pt",
    map_location=device
)
model.load_state_dict(state["model_state_dict"])
model.eval()
# ====================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/image")
def camera():
    cam = get_camera()
    cam.grab()

    with torch.no_grad():
        pred = model(cam.getTensor())

    cls = torch.argmax(pred[0]).item()

    if cls == 0:
        text = "Desert"
    elif cls == 1:
        text = "Forest"
    elif cls == 2:
        text = "Mountain"
    else:
        text = ""

    return jsonify({
        "text_prediction": text,
        "image": cam.blob
    })


@app.route("/stop", methods=["POST"])
def stopcam():
    global cam
    if cam:
        cam.close()
        cam = None
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
