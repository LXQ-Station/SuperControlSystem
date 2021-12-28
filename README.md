# SuperControlSystem
`Face_Recognization` (FaceNet) 

`TFHE` (HNP) 

`Hand_Face_Detection` (Mediapipe)
# DEMO (pesudo)
![result](https://raw.githubusercontent.com/liziyu0104/SuperControlSystem/main/SHOW_IMAGE/result.png)


# IT CAN
- [X] FEATURE EXTRACTOR
- [X] ENCRYPTION
- [X] SYSTEM WITHOUT TOUCH

# VERSION OF PYTHON
`FEATURE` `API` python3.7
`SERVER` python3.8

# TIPS
* `weight document` already include
* `path` needs to change 
* `image` needs be prepared by yourself (upload or use API.py to take pictures)
* `image` should be square and use same clear light 
* `API.py` can cut a almost square
```python
square = self.image[210:210+400,450:450+380] # change this line to get perfect square
```
* `requirement.txt` is ready and I suggest you to install in your virtual environnement
```python
pip install requirements.txt # to get all package (some package is not necessaire)
```
# HOW TO RUN
```python
python API.py [--user]
```
![STAR](https://raw.githubusercontent.com/liziyu0104/SuperControlSystem/main/SHOW_IMAGE/star.svg)

# RESULT 
## FHE
![STAR](https://raw.githubusercontent.com/liziyu0104/SuperControlSystem/main/SHOW_IMAGE/R_FHE.jpg)
## FACENET
![STAR](https://raw.githubusercontent.com/liziyu0104/SuperControlSystem/main/SHOW_IMAGE/R2_FACE.jpg)
![STAR](https://raw.githubusercontent.com/liziyu0104/SuperControlSystem/main/SHOW_IMAGE/R1_FACE.jpg)
