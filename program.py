import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import mss
import numpy as np
import cv2

# Paths
BASE_MODEL = "google/siglip-base-patch16-224"
MODEL_PATH = "./checkpoints/checkpoint-28130"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Only load image processor — NO tokenizer
processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device)

print("Model loaded on", device)

# Screen capture tool
sct = mss.mss()

monitor = sct.monitors[1]  # primary screen (0 is global virtual)
print("Capturing screen:", monitor)

def get_screen_frame():
    img = np.array(sct.grab(monitor))
    img = img[..., :3]  # drop alpha
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    topk = torch.topk(probs, k=5)  # top 5 labels
    scores = topk.values[0].cpu().numpy()
    labels = [model.config.id2label[idx] for idx in topk.indices[0].cpu().numpy()]

    return list(zip(labels, scores))

while True:
    frame = get_screen_frame()

    # Resize preview (not input shape – input handled by processor)
    preview = cv2.resize(frame, (800, 450))
    cv2.imshow("SIGLIP Screen Feed", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

    preds = predict(frame)

    print("\033c")  # clear terminal
    print("Top predictions:")
    for label, score in preds:
        print(f"{label}: {score:.4f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()