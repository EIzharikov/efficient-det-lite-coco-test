import cv2
import numpy as np
import tensorflow as tf


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Error. Image was not loaded.")
    return img


def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path, num_threads=12)
    interpreter.allocate_tensors()
    return interpreter


def prepare_input(interpreter, img):
    input_index = interpreter.get_input_details()[0]["index"]
    in_frame = cv2.resize(img, (640, 640))
    in_frame = in_frame.reshape((1, 640, 640, 3)).astype(np.float32)
    interpreter.set_tensor(input_index, in_frame)
    return interpreter


def run_inference(interpreter):
    interpreter.invoke()
    bboxes = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    class_ids = interpreter.get_tensor(interpreter.get_output_details()[1]["index"])
    confs = interpreter.get_tensor(interpreter.get_output_details()[2]["index"])
    return bboxes, class_ids, confs


def draw_boxes(img, bboxes):
    h, w = img.shape[:2]
    for box in bboxes[0]:
        if np.all(box == 0):
            continue
        ymin = int(box[0] * h)
        xmin = int(box[1] * w)
        ymax = int(box[2] * h)
        xmax = int(box[3] * w)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return img


def display_and_save_result(img, output_path):
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_path, img)


def main():
    image_path = "dog.jpg"
    model_path = "model_float32.tflite"
    output_path = "dog_result_tflite.jpg"

    img = load_image(image_path)
    interpreter = load_model(model_path)
    interpreter = prepare_input(interpreter, img)
    bboxes, class_ids, confs = run_inference(interpreter)

    print("Bounding boxes shape:", bboxes.shape)
    print("Bounding boxes:", bboxes)
    print("Class IDs shape:", class_ids.shape)
    print("Class IDs:", class_ids)
    print("Confidences shape:", confs.shape)
    print("Confidences:", confs)

    img = draw_boxes(img, bboxes)
    display_and_save_result(img, output_path)


if __name__ == "__main__":
    main()
