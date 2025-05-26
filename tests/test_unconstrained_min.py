from src.unconstrained_min import LineSearchMinimizer
from tests.examples import circle,axis_aligned_ellipses,rotated_ellipse,Rosenbrock, smooth_triangle
from src.utils import plot_contour, plot_func_vals_vs_iter

opt_circle_grad = LineSearchMinimizer(f=circle,x0=[1.0, 1.0],obj_tol=1e-12,param_tol=1e-8,max_iter=100,method='gradient')
x_min, f_min, success = opt_circle_grad.minimize()
print("\nFinal result circl gradient:")
print("x_min =", x_min)
print("f_min =", f_min)
print("success =", success)

opt_circle_newton = LineSearchMinimizer(f=circle,x0=[1.0, 1.0],obj_tol=1e-12,param_tol=1e-8,max_iter=100,method='gradient')
x_min, f_min, success = opt_circle_newton.minimize()
print("\nFinal resultn circle newton:")
print("x_min =", x_min)
print("f_min =", f_min)
print("success =", success)
plot_contour(f=circle,iterates1=opt_circle_grad.iterates,iterates2=opt_circle_newton.iterates,label1="Gradient Descent",label2="Newton",
title="Circle: Contour with Optimization Paths"
)


opt_axis_ellipses_grad = LineSearchMinimizer(f=axis_aligned_ellipses,x0=[1.0, 1.0],obj_tol=1e-12,param_tol=1e-8,max_iter=100,method='gradient')
x_min, f_min, success = opt_axis_ellipses_grad.minimize()
print("\nFinal result axis_ellipse gradient:")
print("x_min =", x_min)
print("f_min =", f_min)
print("success =", success)

opt_axis_ellipses_newton = LineSearchMinimizer(f=axis_aligned_ellipses,x0=[1.0, 1.0],obj_tol=1e-12,param_tol=1e-8,max_iter=100,method='newton')
x_min, f_min, success = opt_axis_ellipses_newton.minimize()
print("\nFinal result axis_ellipse newton:")
print("x_min =", x_min)
print("f_min =", f_min)
print("success =", success)

plot_contour(f=axis_aligned_ellipses,iterates1=opt_axis_ellipses_grad.iterates,iterates2=opt_axis_ellipses_newton.iterates,label1="Gradient Descent",label2="Newton",
title="Axis-Aligned Ellipses:"
)


opt_ellipse_grad = LineSearchMinimizer(f=rotated_ellipse,x0=[1.0, 1.0],obj_tol=1e-12,param_tol=1e-8,max_iter=100,method='gradient')
x_min, f_min, success = opt_ellipse_grad.minimize()
print("\nFinal result ellipse gradient:")
print("x_min =", x_min)
print("f_min =", f_min)
print("success =", success)

opt_ellipse_newton = LineSearchMinimizer(f=rotated_ellipse,x0=[1.0, 1.0],obj_tol=1e-12,param_tol=1e-8,max_iter=100,method='newton')
x_min, f_min, success = opt_ellipse_newton.minimize()
print("\nFinal result ellipse newton:")
print("x_min =", x_min)
print("f_min =", f_min)
print("success =", success)

plot_contour(f=rotated_ellipse,iterates1=opt_ellipse_grad.iterates,iterates2=opt_ellipse_newton.iterates,label1="Gradient Descent",label2="Newton",
title="Rotated_ellipse:"
)

opt_rosenbrock_grad = LineSearchMinimizer(f=Rosenbrock,x0=[-1.0, 2.0],obj_tol=1e-12,param_tol=1e-8,max_iter=10000,method='gradient')
x_min, f_min, success = opt_rosenbrock_grad.minimize()
print("\nFinal result rosnbrock gradient:")
print("x_min =", x_min)
print("f_min =", f_min)
print("success =", success)
opt_rosenbrock_newton = LineSearchMinimizer(f=Rosenbrock,x0=[-1.0, 2.0],obj_tol=1e-12,param_tol=1e-8,max_iter=100,method='newton')
x_min, f_min, success = opt_rosenbrock_newton.minimize()
print("\nFinal result rosenbrock newton:")
print("x_min =", x_min)
print("f_min =", f_min)
print("success =", success)
plot_contour(f=Rosenbrock,iterates1=opt_rosenbrock_grad.iterates,iterates2=opt_rosenbrock_newton.iterates,label1="Gradient Descent",label2="Newton",
title="Rosenbrock:"
)



opt_triangle_grad = LineSearchMinimizer(f=smooth_triangle,x0=[1.0, 1.0],obj_tol=1e-12,param_tol=1e-8,max_iter=100,method='gradient')
x_min, f_min, success = opt_triangle_grad.minimize()
print("\nFinal result traingle gradien:")
print("x_min =", x_min)
print("f_min =", f_min)
print("success =", success)
opt_traingle_newton = LineSearchMinimizer(f=smooth_triangle,x0=[1.0, 1.0],obj_tol=1e-12,param_tol=1e-8,max_iter=100,method='newton')
x_min, f_min, success = opt_traingle_newton.minimize()
print("\nFinal result traingle newton:")
print("x_min =", x_min)
print("f_min =", f_min)
print("success =", success)
plot_contour(f=smooth_triangle,iterates1=opt_triangle_grad.iterates,iterates2=opt_traingle_newton.iterates,label1="Gradient Descent",label2="Newton",
title="smooth_triangel"
)

opt_dict = {'smooth_triangle': [opt_triangle_grad,opt_traingle_newton],
            'Rosenbrock': [opt_rosenbrock_grad, opt_rosenbrock_newton], 
            'Rotated Ellipse': [opt_ellipse_grad, opt_ellipse_newton],
            'Axis-Aligned Ellipses': [opt_axis_ellipses_grad,opt_axis_ellipses_newton],
            'Circle': [opt_circle_grad, opt_circle_newton]
            }
for title, opts in opt_dict.items():
   plot_func_vals_vs_iter(
    {"Gradient Descent": opts[0].objective_values,
     "Newton": opts[1].objective_values},
    title=title
)

