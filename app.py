from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

app = Flask(__name__)

# cargar modelo
model = load_model("modelo.h5", custom_objects={'KerasLayer': hub.KerasLayer})
classes = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

def analizar_imagen(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    pred = model.predict(img)[0]

    nsfw_score = pred[1] + pred[3] + pred[4]

    return {
        "resultados": {classes[i]: float(pred[i]*100) for i in range(len(classes))},
        "nsfw": float(nsfw_score*100),
        "estado": "NSFW" if nsfw_score > 0.6 else "Seguro"
    }

@app.route("/", methods=["GET", "POST"])
def index():
    data = None
    if request.method == "POST":
        file = request.files["file"]
        data = analizar_imagen(file)
    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)