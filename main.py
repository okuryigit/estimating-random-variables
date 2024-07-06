import numpy as np

conv_blind=0
conv_givenA = 0
conv_optimal=0
conv_linear = 0
conv_ml = 0
conv_map=0


for i in range(0,1000):

    rv_x = np.random.uniform(4.61, 25.39, 1000)
    rv_y = np.random.uniform(-10.39, 10.39, 1000)  # limits of the samples found using
    # std. deviation and mean formulas of discrete uniform dist.
    M = np.column_stack((rv_x, rv_y))  # to create 1000x2 matrix with x and y samples
    matrix_R = np.array([[0.8, 0.75], [0.75, 1]])
    matrix_L = np.linalg.cholesky((matrix_R)) # cholesky factorization
    matrix_final = np.dot(M, matrix_L)

    X_f = matrix_final[:, 0]
    Y_f = matrix_final[:, 1]

    # blind estimate
    xsample_sum = 0
    ysample_sum = 0
    for i in range(0, 1000):
        xsample_sum = xsample_sum + matrix_final[i, 0]

    xsample_mean = xsample_sum / 1000
    blind_error = 0
    for i in range(0, 1000):
        blind_error += (matrix_final[i, 0] - xsample_mean) ** 2
    MSE_blind = blind_error / 1000

    # Assuming that Xf > mean(Yf)/2  estimation
    count = 0
    for j in range(0, 1000):
        ysample_sum = ysample_sum + matrix_final[j, 1]

    ysample_mean = ysample_sum / 1000
    cond = 0
    for i in range(0, 1000):
        if matrix_final[i, 0] > (ysample_mean / 2):
            cond = cond + matrix_final[i, 0]
            count = count + 1
    condsample_mean = cond / count
    square_error1 = 0
    for i in range(0, 1000):
        square_error1 += (X_f[i] - condsample_mean) ** 2
    MSE_givenA = square_error1 / 1000

    # linear and optimal estimation
    factor_a = np.cov(X_f, Y_f)[1, 0] / np.var(Y_f)
    factor_b = xsample_mean - factor_a * ysample_mean
    linear_estimate = factor_a * Y_f + factor_b
    error = 0
    for i in range(0, 1000):
        error += (X_f[i] - linear_estimate[i]) ** 2
    MSE_linear = error / 1000
    MSE_optimal=MSE_linear # optimal (the least MSE) estimation for Xf given Yf is also  the linear estimation.


    # ML estimation and MAP estimation
    index=np.argmax(rv_x)
    ml_estimate=X_f[index]
    ml_error=0

    for i in range(0,1000):
        ml_error+=(X_f[i]-ml_estimate)**2
    MSE_ml=ml_error/1000
    MSE_map=MSE_ml  # since prior probabilities are needed for the MAP, and the number of samples increase
    #the MAP estimate converges towards the maximum likelihood estimate.

    conv_blind += MSE_blind
    conv_givenA += MSE_givenA
    conv_optimal += MSE_optimal
    conv_linear += MSE_linear
    conv_ml += MSE_ml
    conv_map += MSE_map


print("MSE of Blind Estimation:", conv_blind/1000)
print("MSE of assuming that Xf > mean(Yf)/2  estimation:", conv_givenA/1000)
print("MSE of the optimal estimator:", conv_optimal/1000)
print("MSE of the linear estimator:", conv_linear/1000)
print("MSE of ML estimation:", conv_ml/1000)
print("MSE of MAP estimation:", conv_map/1000)