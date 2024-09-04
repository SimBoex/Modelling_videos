# SignLanguageDetection

Some important points:
- in real time, seems like there are problems for the detection part;
- yet, with the splitting in train and test we achieve an accuracy of 1;
- i guess the problem is related to how we extract the keypoints infact, if they are selected correctly we achieve the previous accuracy while
in real time, seeems like how we move the head influences a lot the decision of which gest is done.

- Resources:
1. https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md

    1.1 https://www.youtube.com/watch?v=I-UOrvxxXEk
    2.1 https://arxiv.org/pdf/1512.02325
    
2. https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md
