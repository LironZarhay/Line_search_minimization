import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm



def plot_contour(f, iterates1, iterates2,
                            label1="Gradient Descent", label2="Newton",
                            title="rosenbrock_function"):
    x1 = [pt[0] for pt in iterates1]
    y1 = [pt[1] for pt in iterates1]
    x2 = [pt[0] for pt in iterates2]
    y2 = [pt[1] for pt in iterates2]

    if 'smooth_triangel' in title.lower():
        x_min, x_max = -2, 2
        y_min, y_max = -2, 2
    else:
    
        x_min, x_max = -2.5, 2.5
        y_min, y_max = -2.5, 2.5

    # Contour grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                         np.linspace(y_min, y_max, 1000))
    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = f(np.array([xx[i, j], yy[i, j]]), hess=False)[0]

    # Start plot
    plt.figure(figsize=(8, 6))
    levels = 500 if 'smooth_triangle' in title.lower() else 200
    cp = plt.contour(xx, yy, zz, levels=levels, cmap="plasma", linewidths=0.8)


    plt.clabel(cp, inline=True, fontsize=7)

    # Path lines with steps
    plt.plot(x1, y1, '-o', color='dodgerblue', markersize=3, linewidth=2,
             label=f"{label1} ({len(x1)} steps)")
    plt.plot(x2, y2, '-o', color='darkorange', markersize=3, linewidth=2,
             label=f"{label2} ({len(x2)} steps)")

    # Mark start and end of each method
    
    plt.scatter([x1[-1]], [y1[-1]], color="red", s=60, marker="*", label=f"{label1} end")

    
    plt.scatter([x2[-1]], [y2[-1]], color="black", s=60, marker="*", label=f"{label2} end")

    # Axis and layout
    plt.title(title, fontsize=14)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.axis("equal")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=5)
    plt.tight_layout()

    # Save once to disk
    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/{title.replace(' ', '_').lower()}.png"
    if not os.path.exists(save_path):
        plt.savefig(save_path, dpi=300)
        print(f"📷 Saved: {save_path}")
    else:
        print(f"✅ Already saved: {save_path}")

    plt.close()




def plot_func_vals_vs_iter(f_vals_dict, title):
    """
    f_vals_dict: dictionary like {'GD': [f1, f2, ...], 'NT': [f1, f2, ...]}
    title: string to describe the function/problem
    """
    plt.figure(figsize=(8, 5))
    for label, f_vals in f_vals_dict.items():
        plt.plot(range(len(f_vals)), f_vals, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Function value")
    plt.title(f"Function Value vs. Iteration for {title}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{title.lower().replace(' ', '_')}_func_vals.png")
    plt.close()
