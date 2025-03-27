import json
import os
import time

import cv2
import numpy as np
from main import load_model, prepare_input, run_inference
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_on_coco(interpreter, coco_gt, image_dir, max_images=None):
    results = []
    total_time = 0.0
    PROCESSED_IMAGES = 0

    img_ids = coco_gt.getImgIds()

    if max_images is not None:
        img_ids = img_ids[:max_images]

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        image_path = os.path.join(image_dir, img_info["file_name"])

        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Image {image_path} not found, skipping...")
                continue
            start_time = time.perf_counter()
            orig_h, orig_w = img.shape[:2]

            interpreter = prepare_input(interpreter, img)
            bboxes, class_ids, confs = run_inference(interpreter)
            elapsed = time.perf_counter() - start_time
            total_time += elapsed
            PROCESSED_IMAGES += 1

            for i in range(bboxes.shape[1]):
                box = bboxes[0][i]
                if np.all(box == 0):
                    continue

                ymin, xmin, ymax, xmax = box
                xmin = xmin * orig_w
                xmax = xmax * orig_w
                ymin = ymin * orig_h
                ymax = ymax * orig_h

                width = xmax - xmin
                height = ymax - ymin

                results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(class_ids[0][i]) + 1,
                        "bbox": [xmin, ymin, width, height],
                        "score": float(confs[0][i]),
                    }
                )

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    avg_latency = total_time / PROCESSED_IMAGES if PROCESSED_IMAGES > 0 else 0
    fps = PROCESSED_IMAGES / total_time if total_time > 0 else 0

    with open("coco_results.json", "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes("coco_results.json")

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats, avg_latency, fps, PROCESSED_IMAGES, total_time


def main():
    model_path = "model_float32.tflite"
    interpreter = load_model(model_path)

    coco_ann_path = "annotations/instances_val2017.json"
    coco_image_dir = "val2017"

    coco_gt = COCO(coco_ann_path)

    max_images = 300
    img_ids = coco_gt.getImgIds()[:max_images]

    selected_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_ids))
    selected_imgs = coco_gt.loadImgs(img_ids)
    selected_cats = coco_gt.loadCats(coco_gt.getCatIds())

    coco_gt_subset = COCO()
    coco_gt_subset.dataset = {
        "info": coco_gt.dataset.get("info", {}),
        "licenses": coco_gt.dataset.get("licenses", []),
        "categories": selected_cats,
        "images": selected_imgs,
        "annotations": selected_anns,
    }
    coco_gt_subset.createIndex()

    metrics, avg_latency, fps, processed_images, total_time = evaluate_on_coco(
        interpreter, coco_gt_subset, coco_image_dir, max_images=None
    )

    print("\nCOCO Evaluation Metrics:")
    print(f"mAP @ [0.5:0.95]: {metrics[0]:.3f}")
    print(f"mAP @ 0.5: {metrics[1]:.3f}")
    print(f"mAP @ 0.75: {metrics[2]:.3f}")
    print(f"mAP @ small: {metrics[3]:.3f}")
    print(f"mAP @ medium: {metrics[4]:.3f}")
    print(f"mAP @ large: {metrics[5]:.3f}")

    print("\nPerformance Metrics:")
    print(f"Average latency per image: {avg_latency * 1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print(f"Total processed images: {processed_images}")
    print(f"Total inference time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
