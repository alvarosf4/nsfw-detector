from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# cargar modelo (CORREGIDO)
model = tf.keras.models.load_model(
    "modelo.h5",
    custom_objects={"KerasLayer": hub.KerasLayer},
    compile=False
)

classes = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']


def analizar_imagen(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "error": "Imagen inválida"
        }

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    nsfw_score = float(pred[1] + pred[3] + pred[4])

    return {
        "resultados": {
            classes[i]: float(pred[i] * 100) for i in range(len(classes))
        },
        "nsfw": float(nsfw_score * 100),
        "estado": "NSFW" if nsfw_score > 0.6 else "Seguro"
    }


@app.route("/", methods=["GET", "POST"])
def index():
    data = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", data={"error": "No file uploaded"})

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", data={"error": "Empty file"})

        data = analizar_imagen(file)

    return render_template("index.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)