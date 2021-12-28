# SuperControlSystem
`Face_Recognization` (FaceNet) 

`面部识别` (FaceNet)

`Fully Homomorphic Encryption over the Torus` (HNP) 

`环面全同态加密` (TFHE)


`Hand_Face_Detection` (Mediapipe)

`手脸检测` (Mediapipe)

# STEP
```python
    1.run API.py: take a photo (close index and thumb finger to select and move ; close index and middle finger to select and push these two finger down together to realize clic on the botton)
    2.move the generated image to new folder in folder FEATURE named "./square"
    3.run batch_extractor.py to get the relative feature which saved as numpy
    4.to test the picture you have taken is ok or not, you can run dissimilarity.py before that, you need to make sure "features" are in your generated folder "./square_features"
    5.move "features" to folder SERVER and change the path in FHE.py
    6.run FHE.py
```
# DEMO (pesudo)
![result](https://raw.githubusercontent.com/liziyu0104/SuperControlSystem/main/SHOW_IMAGE/result.png)

### BOTTON
- [X] focus on your face (Blue one on the right side)
- [X] take a photo but you need to activate the blue botton at first (grey one the the right side)
- [ ] login (green one on the left side)
- [ ] logout (red one on the left side)

# IT CAN
- [X] FACE FEATURE EXTRACTOR
- [X] FULLY HOMOMORPHIC ENCRYPTION
- [X] SYSTEM WITHOUT TOUCH
- [X] LIVE DETECTION (BLINK DETECTION)

# VERSION OF PYTHON
`FEATURE` `API` python3.7
`SERVER` python3.8

# TIPS
* `weight document` already include
* `path` may need to change (relative path is in used, normally not necessary to change) 
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
* `number of photo` for each clic on botton can be changed:
```python
404|   photo = 6 # to take 5 picture each clic 
``` 
# HOW TO RUN
```python
python API.py [--user]
python batch_extractor.py # in the folder FEATURE, need to add your dataset folder 
python FHE.py # in the folder SERVER, need to add your feature's numpy doc
```
![STAR](https://raw.githubusercontent.com/liziyu0104/SuperControlSystem/main/SHOW_IMAGE/star.svg)

# RESULT 
## FHE
![STAR](https://raw.githubusercontent.com/liziyu0104/SuperControlSystem/main/SHOW_IMAGE/R_FHE.jpg)
## FACENET
![STAR](https://raw.githubusercontent.com/liziyu0104/SuperControlSystem/main/SHOW_IMAGE/R2_FACE.jpg)
![STAR](https://raw.githubusercontent.com/liziyu0104/SuperControlSystem/main/SHOW_IMAGE/R1_FACE.jpg)
