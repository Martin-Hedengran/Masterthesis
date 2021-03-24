import numpy as np


def checkEssentialScore(E, K, pts1, pts2, sigma=1.0):
    #essential to fundamental matrix
    kinv = np.linalg.inv(K)
    kinvT = np.transpose(kinv)
    F21 = kinvT * E * kinv

    #Outlier rejection threshold based on 1 pixel deviation at 95%
    th = 3.841
    thscore = 5.991
    score = 0
    #? what is this?
    invSigmaSquare = 1.0 / (sigma * sigma)

    for i in range(len(pts1)):

        #Reprojection error in first image
        a1 = F21[0,0] * pts1[i][0][0] + F21[0,1] * pts1[i][0][1] + F21[0,2]
        b1 = F21[1,0] * pts1[i][0][0] + F21[1,1] * pts1[i][0][1] + F21[1,2]
        c1 = F21[2,0] * pts1[i][0][0] + F21[2,1] * pts1[i][0][1] + F21[2,2]
        
        num1 = a1 * pts2[i][0][0] + b1 * pts2[i][0][1] + c1
        squareDist1 = (num1 ** 2) /(a1**2 + b1**2)
        chiSquare1 = squareDist1 * invSigmaSquare
        if chiSquare1 <= th:
            score = score + thscore - chiSquare1
        
        #Reprojection error in second image
        a2 = F21[0,0] * pts2[i][0][0] + F21[0,1] * pts2[i][0][1] + F21[0,2]
        b2 = F21[1,0] * pts2[i][0][0] + F21[1,1] * pts2[i][0][1] + F21[1,2]
        c2 = F21[2,0] * pts2[i][0][0] + F21[2,1] * pts2[i][0][1] + F21[2,2]
        
        num2 = a2 * pts1[i][0][0] + b2 * pts1[i][0][1] + c2
        squareDist2 = (num2 ** 2) /(a2**2 + b2**2)
        chiSquare2 = squareDist2 * invSigmaSquare

        if chiSquare2 <= th:
            score = score + thscore - chiSquare2
    
    return score

def checkHomographyScore(H, pts1, pts2, sigma=1.0):
    Hinv = np.linalg.inv(H)

    #Outlier rejection threshold based on 1 pixel deviation at 95%
    th = 3.841
    thscore = 5.991
    score = 0
    #? what is this?
    invSigmaSquare = 1.0 / (sigma * sigma)
    
    for i in range(len(pts1)):
    
        #Reprojection error in first image
        a1 = 1 / (Hinv[2,0] * pts2[i][0][0] + Hinv[2,1] * pts2[i][0][1] + Hinv[2,2])
        b1 = (Hinv[0,0] * pts2[i][0][0] + Hinv[0,1] * pts2[i][0][1] + Hinv[0,2]) * a1
        c1 = (Hinv[1,0] * pts2[i][0][0] + Hinv[1,1] * pts2[i][0][1] + Hinv[0,2]) * a1

        squareDist1 = (pts1[i][0][0] - b1)**2 + (pts1[i][0][1] - c1)**2

        chiSquare1  = squareDist1 * invSigmaSquare
        if chiSquare1 <= th:
            score = score + th - chiSquare1
        
        #Reprojection error in second image
        a2 = 1 / (Hinv[2,0] * pts1[i][0][0] + Hinv[2,1] * pts1[i][0][1] + Hinv[2,2])
        b2 = (Hinv[0,0] * pts1[i][0][0] + Hinv[0,1] * pts1[i][0][1] + Hinv[0,2]) * a2
        c2 = (Hinv[1,0] * pts1[i][0][0] + Hinv[1,1] * pts1[i][0][1] + Hinv[0,2]) * a2

        squareDist2 = (pts2[i][0][0] - b2)**2 + (pts2[i][0][1] - c2)**2

        chiSquare2 = squareDist2 * invSigmaSquare
        if chiSquare2 <= th:
            score = score + th - chiSquare2

    return score