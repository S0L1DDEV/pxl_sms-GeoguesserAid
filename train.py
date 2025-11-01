# 1. Load datasets
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="dataset")
label_names = dataset["train"].features["label"].names

# 2. Extract labels
id2label = {i: l for i, l in enumerate(label_names)}
label2id = {l: i for i, l in enumerate(label_names)}
num_classes = len(label_names)

# 3. Split 90% train, 10% validation split
dataset = dataset["train"].train_test_split(test_size=0.1)

# 4. Load basemodel
from transformers import AutoModelForImageClassification, AutoImageProcessor

model_name = "google/siglip2-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)


# 5. define transports
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

image_transforms = Compose([
    Resize((224,224)),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std),
])

# 6. preprocess
def preprocess(batch):
    images = batch["image"]
    batch["pixel_values"] = [image_transforms(img.convert("RGB")) for img in images]
    del batch["image"]
    return batch

#dataset = dataset.with_transform(preprocess)
dataset = dataset.map(preprocess, batched=True)

# 7. load model
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=num_classes,           # fill after dataset load
    label2id=label2id,
    id2label=id2label,
)

# 8. Training config
from transformers import TrainingArguments, Trainer
import torch

args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    learning_rate=5e-5,
    num_train_epochs=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    load_best_model_at_end=True,
)

# 9. trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# 10. train
trainer.train()
#trainer.train(resume_from_checkpoint=True)