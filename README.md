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