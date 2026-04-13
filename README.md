# FaceSight Studio

FaceSight Studio is a Python desktop application that uses OpenCV to detect faces and estimate age and gender from webcam or image input. The interface is built to feel like a professional analysis dashboard instead of a plain demo window.

## Features

- Real-time face detection with OpenCV DNN
- Age estimation and gender classification for each detected face
- Webcam mode and image upload mode
- Modern desktop UI with live metrics and analysis panel
- Automatic download of required model files on first launch

## Tech Stack

- Python 3.12+
- OpenCV DNN
- CustomTkinter
- Pillow
- NumPy

## Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
python app.py
```

## Model Files

The app downloads these files into `models/` automatically:

- `face_detection_yunet_2023mar.onnx`
- `age_deploy.prototxt`
- `age_net.caffemodel`
- `gender_deploy.prototxt`
- `gender_net.caffemodel`

## Notes

- Age and gender outputs are estimates from pretrained models, not exact labels.
- Good lighting and a frontal face improve accuracy.
- The first launch may take a little longer while the model files are downloaded.
