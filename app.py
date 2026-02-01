from flask import *
import cv2
import PIL.Image as Image
import io
import base64

import torch
import torchvision.transforms as T

from model import MyViT

USE_WEBCAM = True   # True → webcam, False → mp4
VIDEO_PATH = "F4.mp4"

class readCamera:
    def __init__(self, use_webcam=False, video_path=None):

        if use_webcam:
            # 0 = default webcam
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video source")
        
        print("Camera opened:", self.cap.isOpened())

    def grab(self):
        ret, frame = self.cap.read()

        # loop video if finished
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

app = Flask(__name__, template_folder="template", static_folder="static")

cam = None

@app.route("/")
def index():
    global cam
    if cam is None:
        cam = readCamera(
            use_webcam=USE_WEBCAM,
            video_path=VIDEO_PATH
        )
    return render_template("index.html")

@app.route('/image')
def camera():
    cam.grab()

    device = torch.device("cpu")
    model = MyViT(
        (3, 256, 256),
        n_patches=16,
        n_blocks=4,
        hidden_d=64,
        n_heads=8,
        out_d=3
    ).to(device)
    d = torch.load("./model.pt", map_location=device)
    model.load_state_dict(d["model_state_dict"])
    model.eval()

    pred = model(cam.getTensor())

    cls = torch.argmax(pred[0]).item()
    Text = None

    if cls == 0:
        Text = "Desert"
    elif cls == 1:
        Text = "Forest"
    elif cls == 2:
        Text = "Mountain"
    else:
        Text = ""

    response_data = {
        'text_prediction': Text,
        'image': cam.blob
    }

    return jsonify(response_data)

@app.route("/stop", methods=["POST"])
def stopcam():
    global cam
    if cam:
        cam.close()
        cam = None
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    app.run(debug=False, threaded=False)

