import numpy as np 

def circle(x, hess = False):
    Q = np.array([[1,0],
                   [0,1]]) #identity metrix
    f_val = x.T @ Q @ x
    grad = 2 * Q @ x
    hess_out = 2 * Q if hess else None
    return f_val, grad, hess_out



def axis_aligned_ellipses(x, hess = False):
    Q = np.array([[1, 0],
                   [0, 100]])  # Ellipse matrix
    f_val = x.T @ Q @ x
    grad = 2 * Q @ x
    hess_out = 2 * Q if hess else None
    return f_val, grad, hess_out

def rotated_ellipse(x, hess=False):
    theta = np.pi / 6
    R = np.array([
        [np.sqrt(3)/2, -0.5],
        [0.5, np.sqrt(3)/2]
    ])
    D = np.diag([100, 1])
    Q = R.T @ D @ R
    f_val = x.T @ Q @ x
    grad = 2 * Q @ x
    hess_out = 2 * Q if hess else None
    return f_val, grad, hess_out


def Rosenbrock(x, hess = False):
    """
    Rosenbrock function: f(x,y) = 100(y - x^2)^2 + (1 - x)^2
    """
    f_val  = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    grad = np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
                     200 * (x[1] - x[0]**2)])
    if hess:
        hess_out = np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
                             [-400 * x[0], 200]])
    else:
        hess_out = None
    return f_val, grad, hess_out 



def smooth_triangle(x, hess=False):
    """
    f(x1,x2) = exp(x1 + 3x2 - 0.1) + exp(x1 - 3x2 - 0.1) + exp(-x1 - 0.1)
    Returns: f_val, grad, (optional) hessian
    """
    x1, x2 = x[0], x[1]

    e1 = np.exp(x1 + 3*x2 - 0.1)
    e2 = np.exp(x1 - 3*x2 - 0.1)
    e3 = np.exp(-x1 - 0.1)

    f_val = e1 + e2 + e3

    grad = np.array([
        e1 + e2 - e3,         
        3*e1 - 3*e2           
    ])

    if hess:
        h11 = e1 + e2 + e3      
        h22 = 9*e1 + 9*e2       
        h12 = 3*e1 - 3*e2       

        hessian = np.array([
            [h11, h12],
            [h12, h22]
        ])
    else:
        hessian = None

    return f_val, grad, hessian



def linear_function(x, hess=False):
    """
    f(x) = a^T x where a is a fixed non-zero vector.
    Gradient is constant; Hessian is zero.
    """
    a = np.array([1.0, 2.0])  

    f_val = np.dot(a, x)
    grad = a.copy()

    hessian = np.zeros((2, 2)) if hess else None

    return f_val, grad, hessian

