from PIL import Image
import time


def get_average_inference_time(model, path):
    inference_images = path.ls()
    average_time = 0

    for image in inference_images:
        start_time = time.time()

        img = Image.open(image)
        model.predict(img)

        average_time = average_time + (time.time() - start_time)

    average_time = average_time / len(inference_images)

    return average_time


def get_inference_time(model, sample_path):
    start_time = time.time()

    img = Image.open(sample_path)
    model.predict(img)