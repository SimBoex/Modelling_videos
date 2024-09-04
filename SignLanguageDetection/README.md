# SignLanguageDetection

## Some important points:
- In real time, seems like there are some problems for the detection part;
- Yet, with the splitting in train and test we achieve an accuracy of 1;
- I guess, the problem is related to how we extract the keypoints. In fact, seeems like how we move the head influences a lot the decision of which gesture is performed.

## References:
This project is inspired from [here](https://www.youtube.com/watch?v=doDUihpj6ro&t=33s&themeRefresh=1)

- Resources:
1. https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md

    1.1 https://www.youtube.com/watch?v=I-UOrvxxXEk
    2.1 https://arxiv.org/pdf/1512.02325
    
2. https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md


## What did I learn?
1. How is possible to detect and then track an object instead of repeating the prediction to improve the computation time;
2. How is possible to extract data from the webcam;
3. I revied the concept of how to interpret the loss plots from this (link)[https://towardsdatascience.com/learning-curve-to-identify-overfitting-underfitting-problems-133177f38df5], especially why we need to reach the convergence part;
