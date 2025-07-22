from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose, Normalize, ToTensor, ColorJitter

import config

data = load_dataset(config.DATASET_CHECKPOINT) # load dataset from hugging face hub
data = data['train'].train_test_split(test_size=0.2) # split into 80:20 ratio

# label mapping
labels = data["train"].features["label"].names
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

def process_data():
    image_processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT)

    normalize = Normalize(mean = image_processor.image_mean, std = image_processor.image_std)

    size = (
        image_processor.size['shortest_edge']
        if 'shortest_edge' in image_processor.size
        else (image_processor.size['height'], image_processor.size['width'])
    )

    emotion_transforms = Compose([RandomResizedCrop(size),
                       RandomHorizontalFlip(),
                       ColorJitter(brightness=config.BRIGHTNESS, contrast=config.CONTRAST, saturation=config.SATURATION, hue=config.HUE),
                       ToTensor(),
                       normalize
    ])

    def transforms(examples):
        examples['pixel_values'] = [emotion_transforms(img.convert('RGB')) for img in examples['image']]
        del examples['image']
        return examples

    data = data.map(transforms, batched=True)

    train_data = data["train"]
    test_data = data["test"]

    train_data.save_to_disk(config.PROCESSED_TRAIN_DATA)
    test_data.save_to_disk(config.PROCESSED_TEST_DATA)

    return labels, id2label, label2id

if __name__ == "__main__":
    process_data()