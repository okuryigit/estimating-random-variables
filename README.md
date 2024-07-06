# estimating random variables
In order to estimate random variable Xf from Yf via different methods:

1. Generate two uniform random variables X and Y. These random variables 
are discrete uniformly distributed variables with 𝜇𝑋 ≈ 15, 𝜇𝑌 ≈ 0, and 𝜎𝑋 ≈
 𝜎𝑌 ≈ 6. Each variable should include 1000 elements. Unite them in a final 
1000x2 initial matrix M. 
2. Now, you will implement Cholesky factorization to generate a correlation 
between X and Y. The correlation matrix for Cholesky factorization is  
R = [0.8 0.75; 0.75 1]  
Here, you will generate an L matrix by using Cholesky factorization with R.  
Multiply your M matrix with L and get your final 1000x2 
matrix which has Xf and Yf  as the columns. 
3. Now you have 2 random variables X and Y. Hereupon, you will 
estimate Xf from Yf by using below: 
● Blind estimation.
● Assuming that Xf > mean(Yf)/2 
● Optimal estimation given Yf 
● Linear estimation given Yf 
● Maximum likelihood (ML) estimation given Yf 
● Maximum a posteriori (MAP) estimation given Yf 
Repeat each case 1000 times (again monte-carlo), observe the 
convergence and calculate and print out the mean-squared error (MSE).
