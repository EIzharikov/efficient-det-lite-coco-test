# efficient-det-lite-coco-test

All instructions are reproducible for Windows 10.

To reproduce results with EfficientDet Lite you should:
1. Create environment and activate:
```commandline
    python -m venv venv
    source ./venv/Scripts/python.exe
```
2. Install all requirements:
```commandline
pip install -r requirements.txt
```
3. If you want just check work of model, you can launch with one example
```commandline
python main.py
```
4. If you want to evaluate the model with 300 random images from COCO you should do the next steps:
   1. Download annotations from COCO https://cocodataset.org/#home
   2. Download val dataset from COCO 2017 from the same url
   3. Place all file in repository folder
   4. Launch evaluation:
   ```commandline
   python eval.py
   ```

Examples of running evaluation you can check in examples folder.

The results of reproducing are:

| Metric         | EfficientDet Lite result |
|----------------|--------------------------|
| mAP@[0.5:0.95] | 0.437                    |
| mAP@0.5        | 0.587                    |
| mAP-small      | 0.220                    |
| mAP-medium     | 0.504                    |
| mAP-large      | 0.584                    |

The next provided results are done with:

CPU:12th Gen Intel Core i5-1240P(12 cores: 4 perfomance, 8 efficiency, 4.4 GHz)

RAM: 16 GB DDR-4-3200 MHz

| Metric                   | EfficientDet Lite |
|--------------------------|-------------------|
| AVG Latency, ms          | 1694.59           |
| Throughput, FPS          | 0.59              |
| Total inference takes, s | 508.38            |