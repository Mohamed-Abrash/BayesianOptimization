# BayesianOptimization
Implementing a Bayesian Optimization Algorithm In MATLAB



<img src="https://github.com/user-attachments/assets/0e7e2b95-7f38-4943-8a49-65aa717f1ba1" alt="Bayesian Optimization" width="700" />


This project explores Bayesian Optimization as an alternative to traditional gradient-based and gradient-free methods for finding the global minimum of a function.  The research is motivated by the need to optimize expensive-to-evaluate, black-box functions, where gradient information is unavailable or unreliable.  Bayesian Optimization offers a flexible and efficient approach to tackle such challenges.  

The project aims to investigate the effectiveness of Bayesian Optimization using Gaussian Processes as surrogate models.  

The project examines various aspects of this method, including:  

- Different covariance functions (e.g., Matérn, exponential, Gaussian) and their impact on optimization.
- Comparing acquisition functions (e.g., Lower Confidence Bound, Expected Improvement, Probability of Improvement) and their exploration-exploitation strategies.
- Addressing challenges like anisotropy and multiple global minima.  

The project presents case studies with various test functions to illustrate the capabilities and limitations of Bayesian Optimization, particularly in higher-dimensional problems.

Below is a figure showcasing the behavior of the algorithm when different acquisition functions are used to optimize functions in 2D:

<img src="https://github.com/user-attachments/assets/dd3821f4-48d0-49cf-b8b0-1b9c3f998518" alt="Bayesian Optimization 2D" width="700" />


\
This repository includes:
- Full written report
- Matlab implementaiton 
