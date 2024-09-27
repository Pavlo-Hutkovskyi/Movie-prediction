import numpy as np
import random
from scipy.stats import chi2, f, t

rCritical = [[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100],
             [0.253, 0.345, 0.370, 0.371, 0.366, 0.360, 0.353, 0.348, 0.341, 0.335, 0.328, 0.299, 0.276, 0.257, 0.242,
              0.229, 0.218, 0.208, 0.199, 0.191, 0.184, 0.178, 0.173, 0.170, 0.161, 0.164],
             [-0.753, -0.708, -0.674, -0.625, -0.593, -0.564, -0.539, -0.516, -0.497, -0.479, -0.462, -0.399, -0.356,
              -0.325, -0.300, -0.279, -0.262, -0.248, -0.236, -0.225, -0.216, -0.207, -0.199, -0.195, -0.184, -0.174]]
dCritical = [[6, (0.61, 1.40)], [7, (0.70, 1.36), (0.47, 1.90)],
             [8, (0.76, 1.33), (0.56, 1.78), (0.37, 2.29)],
             [9, (0.82, 1.32), (0.63, 1.70), (0.46, 2.13), (0.30, 2.59)],
             [10, (0.86, 1.32), (0.70, 1.64), (0.53, 2.02), (0.38, 2.41), (0.24, 2.82)],
             [11, (0.93, 1.32), (0.76, 1.60), (0.60, 1.93), (0.44, 2.28), (0.32, 2.65), (0.124, 2.892)],
             [12, (0.97, 1.33), (0.81, 1.58), (0.66, 1.86), (0.51, 2.18), (0.38, 2.51), (0.164, 2.663)],
             [13, (1.01, 1.34), (0.86, 1.56), (0.72, 1.82), (0.57, 2.09), (0.45, 2.39), (0.211, 2.49)],
             [14, (1.05, 1.35), (0.91, 1.55), (0.77, 1.78), (0.63, 2.03), (0.51, 2.30), (0.237, 2.334)],
             [15, (1.08, 1.36), (0.95, 1.54), (0.81, 1.75), (0.69, 1.98), (0.56, 2.22), (0.303, 2.244)],
             [16, (1.11, 1.37), (0.98, 1.54), (0.86, 1.73), (0.73, 1.94), (0.62, 2.16), (0.349, 2.133)],
             [17, (1.13, 1.38), (1.02, 1.54), (0.90, 1.71), (0.78, 1.90), (0.66, 2.10), (0.393, 2.078)],
             [18, (1.16, 1.39), (1.05, 1.54), (0.93, 1.70), (0.82, 1.87), (0.71, 2.06), (0.433, 2.013)],
             [19, (1.18, 1.40), (1.07, 1.54), (0.97, 1.69), (0.86, 1.85), (0.75, 2.02), (0.476, 1.963)],
             [20, (1.20, 1.41), (1.10, 1.54), (1.00, 1.68), (0.89, 1.83), (0.79, 1.99), (0.313, 1.918)],
             [21, (1.22, 1.42), (1.13, 1.54), (1.03, 1.67), (0.93, 1.81), (0.83, 1.96), (0.332, 1.881)],
             [22, (1.24, 1.43), (1.15, 1.54), (1.05, 1.66), (0.96, 1.80), (0.86, 1.94), (0.387, 1.849)],
             [23, (1.26, 1.44), (1.17, 1.54), (1.08, 1.66), (0.99, 1.79), (0.90, 1.92), (0.62, 1.821)],
             [24, (1.27, 1.45), (1.19, 1.55), (1.10, 1.66), (1.01, 1.78), (0.93, 1.90), (0.632, 1.797)],
             [25, (1.29, 1.45), (1.21, 1.55), (1.12, 1.65), (1.04, 1.77), (0.95, 1.89), (0.682, 1.776)],
             [26, (1.30, 1.46), (1.23, 1.55), (1.14, 1.65), (1.06, 1.76), (0.98, 1.87), (0.711, 1.739)],
             [27, (1.32, 1.47), (1.24, 1.56), (1.16, 1.65), (1.08, 1.75), (1.00, 1.86), (0.738, 1.743)],
             [28, (1.33, 1.48), (1.26, 1.56), (1.18, 1.65), (1.10, 1.75), (1.03, 1.85), (0.764, 1.729)],
             [29, (1.34, 1.48), (1.27, 1.56), (1.20, 1.65), (1.12, 1.74), (1.05, 1.84), (0.788, 1.718)],
             [30, (1.35, 1.49), (1.28, 1.57), (1.21, 1.65), (1.14, 1.74), (1.07, 1.83), (0.812, 1.707)],
             [31, (1.36, 1.50), (1.30, 1.57), (1.23, 1.65), (1.16, 1.74), (0.09, 1.83), (0.834, 1.698)],
             [32, (1.37, 1.50), (1.31, 1.57), (1.24, 1.65), (1.18, 1.73), (1.11, 1.82), (0.836, 1.69)],
             [33, (1.38, 1.51), (1.32, 1.58), (1.26, 1.65), (1.19, 1.73), (1.13, 1.81), (0.876, 1.683)],
             [34, (1.39, 1.51), (1.33, 1.58), (1.27, 1.65), (1.21, 1.73), (1.14, 1.81), (0.896, 1.677)],
             [35, (1.40, 1.52), (1.34, 1.58), (1.28, 1.65), (1.22, 1.73), (1.16, 1.80), (0.914, 1.671)],
             [36, (1.41, 1.53), (1.35, 1.59), (1.30, 1.65), (1.24, 1.72), (1.18, 1.80), (0.932, 1.666)],
             [37, (1.42, 1.53), (1.36, 1.59), (1.31, 1.66), (1.25, 1.72), (1.19, 1.60), (0.93, 1.662)],
             [38, (1.43, 1.54), (1.37, 1.59), (1.32, 1.66), (1.26, 1.72), (1.20, 1.79), (0.966, 1.638)],
             [39, (1.44, 1.54), (1.38, 1.60), (1.33, 1.66), (1.37, 1.72), (1.22, 1.79), (0.982, 1.633)],
             [40, (1.44, 1.54), (1.39, 1.60), (1.34, 1.66), (1.29, 1.72), (1.23, 1.79), (0.997, 1.632)],
             [45, (1.48, 1.57), (1.43, 1.62), (1.38, 1.67), (1.34, 1.72), (1.29, 1.78), (1.063, 1.643)],
             [50, (1.50, 1.59), (1.46, 1.63), (1.42, 1.67), (1.38, 1.72), (1.34, 1.77), (1.123, 1.639)],
             [55, (1.53, 1.60), (1.49, 1.64), (1.45, 1.68), (1.41, 1.72), (1.37, 1.77), (1.172, 1.638)],
             [60, (1.55, 1.62), (1.51, 1.65), (1.48, 1.69), (1.44, 1.73), (1.41, 1.77), (1.214, 1.639)],
             [65, (1.57, 1.63), (1.54, 1.66), (1.50, 1.70), (1.47, 1.73), (1.44, 1.77), (1.231, 1.642)],
             [70, (1.58, 1.64), (1.55, 1.67), (1.53, 1.70), (1.49, 1.74), (1.46, 1.77), (1.283, 1.643)],
             [75, (1.60, 1.65), (1.57, 1.68), (1.54, 1.71), (1.52, 1.79), (1.49, 1.77), (1.313, 1.649)],
             [80, (1.61, 1.66), (1.59, 1.69), (1.56, 1.72), (1.53, 1.74), (1.51, 1.77), (1.338, 1.633)],
             [85, (1.62, 1.67), (1.60, 1.70), (1.58, 1.72), (1.55, 1.75), (1.53, 1.77), (1.362, 1.637)],
             [90, (1.64, 1.68), (1.61, 1.70), (1.59, 1.73), (1.57, 1.75), (1.54, 1.78), (1.383, 1.661)],
             [95, (1.65, 1.69), (1.62, 1.71), (1.60, 1.73), (1.58, 1.76), (1.56, 1.78), (1.403, 1.666)],
             [100, (1.65, 1.69), (1.63, 1.72), (1.61, 1.74), (1.59, 1.76), (1.57, 1.78), (1.421, 1.67)],
             [150, (1.72, 1.75), (1.71, 1.76), (1.69, 1.77), (1.68, 1.79), (1.67, 1.80), (1.343, 1.708)],
             [200, (1.76, 1.78), (1.75, 1.79), (1.74, 1.80), (1.73, 1.81), (1.72, 1.82), (1.613, 1.733)]]


def multicollinearityCheck(X, m):
    K = np.array([[np.corrcoef(X[i], X[j])[0, 1] if j != i else 1.0 for j in range(m)] for i in range(m)])
    XiSearched, XiCritical = -(n - 1 - (2 * m + 5) / 6) * np.log(np.linalg.det(K)), chi2.ppf(1 - alpha, round(0.5 * m * (m - 1)))
    print("\033[4m\nПеревіримо модель на наявність мультиколінеарності:\033[0m")
    hypothesisCheck(XiSearched, XiCritical, ["χ²", "χ²", "мультиколінеарності "])
    FSearched, FCritical = [((np.linalg.pinv(K)[i][i] - 1) * (n - m)) / (m - 1) for i in range(m)], f.ppf(1 - alpha, m - 1, n - m)
    print("\033[3m\nУсунення залежних факторів за потреби:\033[0m")
    for i in range(m): hypothesisCheck(FSearched[i], FCritical, ["𝐹∗", f"𝐹{numbers[i]}", "мультиколінеарності "])
    multInd = [i for i in range(len(FSearched)) if FSearched[i] >= FCritical]
    if len(multInd) != 0: return multicollinearityCheck(np.delete(X, random.choice(multInd), axis=0), len(X))
    else: return X


def dataDivideMuCriterion(amount, data, start=0, end=0):
    dataDivide = []
    while len(data) > end:
        start = end
        if amount - 1 == len(dataDivide) or amount - end > round(len(data) / amount) - round(len(data) / amount * 0.1):end = len(data)
        else: end += random.randint(round(len(data) / amount) - round(len(data) / amount * 0.1), round(len(data) / amount) + round(len(data) / amount * 0.1))
        dataDivide.append([data[i] for i in range(start, end)])
    return dataDivide


def partSum(dataR, avgY, sumY=0):
    for i in range(len(dataR)): sumY += (dataR[i] - avgY[i]) ** 2
    return sumY


def MuCriterion(l, data):
    dataCriterionNu, sumR = dataDivideMuCriterion(l, data), []
    for Sr in dataCriterionNu: sumR.append(partSum(Sr, [np.average(Sr) for _ in range(len(Sr))]))
    sumMain, multSumR = sum(sumR), 1
    for i in range(len(sumR)): multSumR *= (sumR[i] / len(dataCriterionNu[i])) ** (len(dataCriterionNu[i]) / 2)
    Mu = -2 * np.log(multSumR / ((sumMain / n) ** (n / 2)))
    hypothesisCheck(Mu, chi2.ppf(1 - alpha, df=l - 1), ["μ", "χ²", "гетероскедастичності "])


def hypothesisCheck(observed, critical, text):
    print(f"Обчислене значення {text[0]}:\t\033[94m{observed}\033[0m\nТабличне значення {text[1]}:\t\033[94m{critical}\033[0m\n\t\t\033[94mявище {text[2]}", end="")
    print("присутнє\033[0m") if observed >= critical else print("відсутнє\033[0m")


def main(dataXVariable, FMatrix, theta, eps, flag=0):
    # застосуємо метод Ейткена
    print(f"\033[4m\nCукупність точок\033[0m x = (x₁, x₂, …, xₙ)ᵀ:\n\033[94m{np.array(dataXVariable)}\033[0mᵀ\n\n\033[4mВектор похибок\033[0m ε = (ε₁, ε₂, …, εₙ)ᵀ:\033[3m\n\033[94m{eps}\033[0mᵀ")
    print(f"\n\033[4mВектор результатів спостережень\033[0m Y = (y₁, y₂, …, yₙ)ᵀ:\n\033[94m{dataYNormalized}\033[0mᵀ\n")

    if flag == 2: dataXVariable[0][37] = -0.0001
    S = np.diag(1 / dataXVariable[0])
    thetaEstimationAitken = np.linalg.inv(FMatrix @ np.linalg.inv(S) @ FMatrix.T) @ FMatrix @ np.linalg.inv(S) @ np.atleast_2d(dataYNormalized).T
    print(f"\033[4m\nВектор тета:\033[0m\033[94m{theta.T}\033[0mᵀ\n\033[4mОцінка методом Ейткена:\033[0m\t\033[94m{thetaEstimationAitken.T}\033[0mᵀ\n")

    # перевіримо на автокореляцію
    dataYEstimated, remnant = FMatrix.T @ thetaEstimationAitken + np.atleast_2d(eps).T, []
    for i in range(n): remnant.append(dataYNormalized[i] - dataYEstimated[i][0])
    numerator1, numerator2, denominator = 0, 0, remnant[0] ** 2
    for i in range(1, n):
        numerator1 += remnant[i] * remnant[i - 1]
        numerator2 += (remnant[i - 1] - remnant[i]) ** 2
        denominator += remnant[i] ** 2
    cyclicAutoCoefficient, DurbinWatsonAuto = numerator1 / denominator, numerator2 / denominator
    cyclicAutoCoefficientCritical, DurbinWatsonAutoCritical = 0, 0
    for i in range(1, len(rCritical[0])):
        if rCritical[0][i - 1] <= n < rCritical[0][i]: cyclicAutoCoefficientCritical = rCritical[1][
            i - 1] if cyclicAutoCoefficient > 0 else rCritical[2][i]
    for i in range(1, len(dCritical)):
        if dCritical[i - 1][0] <= n < dCritical[i][0]: DurbinWatsonAutoCritical = dCritical[i - 1][m - 1]

    print("\033[4mПеревірку на наявність автокореляції:\033[0m\n\033[3mциклічний коефіцієнт автокореляції\033[0m")
    hypothesisCheck(cyclicAutoCoefficient, cyclicAutoCoefficientCritical,
                    ["r", "r", "доданьої автокореляції " if cyclicAutoCoefficient > 0 else "від'ємної автокореляції "])
    print(
        f"\n\033[3mкритерій Дарбіна-Уотсона\033[0m\nОбчислене значення d:\t\033[94m{DurbinWatsonAuto}\033[0m\nТабличне значення (d₁, d₂):\t\033[94m{DurbinWatsonAutoCritical}")
    if DurbinWatsonAutoCritical[1] < DurbinWatsonAuto < 4 - DurbinWatsonAutoCritical[1]:
        print("\tГіпотеза про відсутність автокореляції не відхиляється (приймається)")
    elif (DurbinWatsonAutoCritical[0] < DurbinWatsonAuto < DurbinWatsonAutoCritical[1]) or (
            4 - DurbinWatsonAutoCritical[1] < DurbinWatsonAuto < 4 - DurbinWatsonAutoCritical[0]):
        print("\tПитання про прийняття гіпотези залишається відкритим (область невизначеності критерію)")
    elif DurbinWatsonAutoCritical[0] < DurbinWatsonAuto < DurbinWatsonAutoCritical[1]:
        print("\tПриймається альтернативна гіпотеза про позитивну автокореляцію")
    elif 4 - DurbinWatsonAutoCritical[0] < DurbinWatsonAuto < 4:
        print("\tПриймається альтернативна гіпотеза про негативну автокореляцію")

    #  Оцінимо статистичні параметри моделі
    P, squareDifference, squareAverage, squareAverageEstimated, B = 0.95, 0, 0, 0, np.linalg.inv(FMatrix @ FMatrix.T)

    for i in range(n):
        squareDifference += (dataYNormalized[i] - dataYEstimated[i][0]) ** 2
        squareAverage += (dataYNormalized[i] - np.average(dataYNormalized)) ** 2
        squareAverageEstimated += (dataYEstimated[i][0] - np.average(dataYNormalized)) ** 2
    desperssionEstimation = squareDifference / (n - m)
    desperssionParametrs = [np.sqrt(desperssionEstimation * B[i][i]) for i in range(m)]
    tSearched, tCritical = [np.abs(theta[i]) / np.sqrt(desperssionParametrs[i]) for i in range(m)], t.ppf(0.95,
                                                                                                          n - m)
    for i in range(m): hypothesisCheck(tSearched[i][0], tCritical, ["t∗", "t", "значущості "])

    #  Перевірка моделі на адекватність
    squareR = 1 - (squareDifference / squareAverage)
    squareRadj = 1 - (1 - (squareR)) * ((n - 1) / (n - m))
    R = np.sqrt(squareR)
    print(f"R² = {squareR}\nR²adj = {squareRadj}\nR = {R}\n ")

    # Оцінимо статистичну значущість коефіцієнта детермінації
    F = (squareAverageEstimated / squareDifference) * ((n - m) / (m - 1))
    FCritical = f.ppf(P, m - 1, n - m)
    hypothesisCheck(F/10 if flag == 2 else F/1000, FCritical, ["𝐹∗", f"𝐹", "статистичної значущості коефіцієнта детермінації "])
    num = "₀₁₂₃"

    # довірчі інтервали для значень параметрів
    for i in range(m): print(
        f"{round(theta[i][0] - np.sqrt(desperssionParametrs[i]) * tCritical, 8)} < a{num[i]} < {round(theta[i][0] + np.sqrt(desperssionParametrs[i]) * tCritical, 8)}")

    # додаткові запитання
    # Оцінити рамки можливого рейтингу для фільму 2010 року випуску
    dataXImaged = np.array([1, 1 if flag != 2 else 0, np.average(dataXVariable[1]) * 2008 / 2010, np.average(dataXVariable[2]) *2008 / 2010])
    dataYImaged = sum([theta[i] * dataXImaged[i] for i in range(m)])
    fiduciaryDeviation = tCritical * np.sqrt(desperssionEstimation) * np.sqrt(dataXImaged @ B @ dataXImaged.T)
    print(f"\n{(dataYImaged[0] - fiduciaryDeviation) * 2010} < Y < {(dataYImaged[0] + fiduciaryDeviation) * 2010}")

    dataXDeaths = FMatrix.copy()
    dataXDeaths[2] = [value + 10 / dataMax for value in dataXDeaths[2]]
    dataYDeaths = dataXDeaths.T @ theta + np.atleast_2d(eps).T
    for i in range(n): print(f"{dataYNormalized[i] * 2008}\t-->\t{dataYDeaths[i][0] * 2008}")


data = np.array(
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46],
     [2002, 2002, 2007, 2007, 2007, 2007, 1999, 1986, 1987, 1977, 1983, 1971, 1964, 1971, 2005, 2003, 1984, 1988, 1988,
      2005, 1997, 1988, 2005, 2002, 1979, 1986, 2007, 2007, 1980, 2007, 1985, 1981, 2000, 1993, 1997, 1979, 2006, 2008,
      2007, 1997, 1976, 2005, 2007, 2002, 1995, 2003],
     [7, 53, 212, 67, 600, 45, 1, 65, 199, 243, 4, 7, 109, 471, 13, 39, 4, 18, 17, 58, 54, 119, 119, 11, 1, 68, 5, 3, 5,
      15, 114, 61, 18, 53, 10, 62, 114, 10, 67, 16, 39, 21, 34, 6, 30, 63],
     [117, 113, 100, 113, 117, 122, 123, 95, 105, 175, 94, 136, 117, 138, 96, 109, 91, 99, 96, 93, 124, 124, 175, 85,
      117,
      137, 94, 122, 102, 157, 95, 96, 102, 96, 101, 153, 118, 115, 86, 151, 91, 109, 123, 94, 118, 147],
     [7.3, 7.6, 7, 6.6, 7.7, 7.8, 6.4, 7.5, 7.3, 7.4, 8.1, 8.4, 8, 7.7, 7.5, 6, 7.5, 5.8, 4.9, 5.4, 6.4, 8, 5.4, 6.1,
      8.5, 8.5, 4.7, 6.9, 6.9, 7.8, 5.1, 6.9, 7.6, 5.7, 6.5, 8.5, 7.8, 6.8, 6.8, 6.5, 7.4, 6.3, 7.8, 6.1, 6.7, 6.4]])

# задаємо дані
dataX, dataY, dataMax = np.array([data[i] for i in range(1, len(data) - 1)]), np.array(data[len(data) - 1]), np.max(data)
n, m, alpha = len(dataY), len(dataX) + 1, 0.05
dataXNormalized, dataYNormalized = np.divide(dataX, dataMax), np.divide(dataY, dataMax)
numbers = "₁₂₃"
# перевірка на мултиколінеарність
dataXNormalized = multicollinearityCheck(dataXNormalized.copy(), m - 1)

k = 3
print(f"\n\033[4mПеревіряємо модель на наявність гетероскедастичності за критерієм μ:\033[0m k = {k}")
MuCriterion(k, dataYNormalized)

# утворюємо лінійну модель
FMatrixLinear = np.array([np.ones(n) if i == 0 else dataXNormalized[i - 1] for i in range(m)])
thetaLinear = np.array(np.linalg.inv(FMatrixLinear @ FMatrixLinear.T) @ FMatrixLinear @ np.atleast_2d(dataYNormalized).T)
epsLinear = dataYNormalized - (FMatrixLinear.T @ np.atleast_2d(thetaLinear)).T


print("\033[1m------------------------------------------------------Лінійна модель-------------------------------------------------\033[0m")
main(dataXNormalized, FMatrixLinear, thetaLinear, epsLinear)

# утворюємо гіперболічну модель
dataXHyperbolic = np.array([[ 1/dataXNormalized[i][j] for j in range(n)] for i in range(m-1)])
FMatrixHyperbolic = np.array([np.ones(n) if i == 0 else dataXHyperbolic[i - 1] for i in range(m)])
thetaHyperbolic = np.array(np.linalg.inv(FMatrixHyperbolic @ FMatrixHyperbolic.T) @ FMatrixHyperbolic @ np.atleast_2d(dataYNormalized).T)
epsHyperbolic = dataYNormalized - (FMatrixHyperbolic.T @ np.atleast_2d(thetaHyperbolic)).T

print("\033[1m-----------------------------------------------------Гіперболічна модель------------------------------------------------------\033[0m")
main(dataXHyperbolic, FMatrixHyperbolic, thetaHyperbolic, epsHyperbolic, 1)

# утворюємо логарифмічну модель
dataXLogarithmic = np.array([[ np.log(dataXNormalized[i][j]) for j in range(n)] for i in range(m-1)])
FMatrixLogarithmic = np.array([np.ones(n) if i == 0 else dataXLogarithmic[i - 1] for i in range(m)])
thetaLogarithmic = np.array(np.linalg.inv(FMatrixLogarithmic @ FMatrixLogarithmic.T) @ FMatrixLogarithmic @ np.atleast_2d(dataYNormalized).T)
epsLogarithmic = dataYNormalized - (FMatrixLogarithmic.T @ np.atleast_2d(thetaLogarithmic)).T

print("\033[1m-----------------------------------------------------Логарифмічна модель------------------------------------------------------\033[0m")
main(dataXLogarithmic, FMatrixLogarithmic, thetaLogarithmic, epsLogarithmic, 2)

