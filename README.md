# SuperControlSystem
`Face_Recognization` (FaceNet) 

`TFHE` (HNP) 

`Hand_Face_Detection` (Mediapipe)

# IT CAN
- [X] FEATURE_EXTRACTOR
- [X] ENCRYPTION
- [X] SYSTEM

# VERSION OF PYTHON
`FEATURE` `API` python3.7
`SERVER` python3.8

# TIPS
* `weight document` is already in
* `path` needs to change 
* `image` needs be prepared by yourself (upload or use API.py to take pictures)
* `image` should be square, user same clear light 
* `API.py` can cut a almost square
```python
square = self.image[210:210+400,450:450+380] # change this line to get perfect square
```
* `requirement.txt` is ready
```python
pip install requirements.txt # to get all package (some package is nnot necessaire)
```
# HOW TO RUN
```python
python API.py [--user]
```
