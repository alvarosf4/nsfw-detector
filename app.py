from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# ================== CONFIGURACIÓN PARA RENDER ==================
# Asegurar que Flask escuche en el puerto correcto de Render
port = int(os.environ.get('PORT', 5000))

# Cargar modelo (con manejo de errores)
try:
    model = load_model("modelo.h5", custom_objects={'KerasLayer': hub.KerasLayer})
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    model = None

classes = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

def analizar_imagen(file):
    if model is None:
        return {"error": "Modelo no cargado"}
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    pred = model.predict(img, verbose=0)[0]

    nsfw_score = pred[1] + pred[3] + pred[4]   # hentai + porn + sexy

    return {
        "resultados": {classes[i]: float(pred[i]*100) for i in range(len(classes))},
        "nsfw": float(nsfw_score*100),
        "estado": "NSFW" if nsfw_score > 0.6 else "Seguro"
    }

@app.route("/", methods=["GET", "POST"])
def index():
    data = None
    if request.method == "POST":
        if 'file' not in request.files:
            data = {"error": "No se subió ningún archivo"}
        else:
            file = request.files["file"]
            if file.filename == '':
                data = {"error": "Archivo vacío"}
            else:
                data = analizar_imagen(file)
    return render_template("index.html", data=data)

# ================== PARA PRODUCCIÓN ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)