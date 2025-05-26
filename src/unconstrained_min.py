import numpy as np
class LineSearchMinimizer:
    
    def __init__(self, f, x0=None, obj_tol=1e-12, param_tol = 1e-8, max_iter=None, method = 'gradient' ):
        """
        Line search optimizer supporting gradient descent and Newton methods.

        Parameters:
        - f: function to minimize. Should return (f_val, grad, hess) given x and hess flag.
        - x0: starting point (default: [1.0, 1.0]).
        - obj_tol: objective tolerance (default: 1e-12).
        - param_tol: step-size tolerance (default: 1e-8).
        - max_iter: max iterations (default: 100).
        - method: 'gradient' or 'newton' (default: 'gradient').

        Attributes:
        - x: current point
        - iterates: list of visited points
        - objective_values: list of function values
        - success: True/False if converged
        """
    
        self.f = f 
        self.method = method.lower()

        self.x0 = np.array(x0 if x0 is not None else [1.0,1.0], dtype=float)
        self.obj_tol=obj_tol
        self.param_tol=param_tol
        self.max_iter=max_iter if max_iter is not None else 100

        self.x = self.x0.copy()
        self.iterates = [self.x.copy()]
        self.objective_values = []
        self.success  = None 


    def backtracking(self, x, f_val, grad, direction):
        t = 1.0
        alpha = 0.5
        c = 0.01
        min_t = 1e-12 # sefty stop
        # Wolf: f(x+tp) <= f(x) + c t ∇f(x)^T p
        while t > min_t:
            x_new = x + t * direction
            f_new, _,_ = self.f(x_new, hess = False) 
            if f_new <= f_val + c * t * np.dot(grad,direction):
                break # Wolf condition satisfied
            t *= alpha 
        return t 
    def minimize(self):
        """
        Run the optimization loop using the selected method.

        Returns
        -------
        x_final : ndarray
            Final estimated minimizer.
        f_final : float
            Final objective value.
        success : bool
            Whether termination criteria were satisfied.
                """
        for i in range(self.max_iter):
            # Choose method and compute values accordingly
            if self.method == 'gradient':
                f_val, grad, _ = self.f(self.x, hess=False)
                direction = -grad

            elif self.method == 'newton':
                f_val, grad, hess = self.f(self.x, hess=True)
                if np.linalg.det(hess) == 0:
                    raise ValueError('Hessian is singular, cannot use Newton method')
                direction = -np.linalg.solve(hess, grad)

            else:
                raise ValueError("Only 'gradient' and 'newton' methods are supported")

            self.objective_values.append(f_val)

            # Step size using Wolfe line search
            step_size = self.backtracking(self.x, f_val, grad, direction)
            x_new = self.x + step_size * direction

            f_new, _, _ = self.f(x_new, hess=False)
            self.iterates.append(x_new.copy())

            print(f"Iter{i}: x = {x_new}, f(x) = {f_new:.6e}")

            # Check stopping criteria
            stop_norm = np.linalg.norm(x_new - self.x)
            obj_diff = abs(f_new - f_val)

            if stop_norm < self.param_tol or obj_diff < self.obj_tol:
                self.success = True
                break

            self.x = x_new

        else:
            self.success = False

        return self.x, self.objective_values[-1], self.success
        
    