# A simple example for the phi and phiplus indicator
import numpy as np
import matplotlib.pyplot as plt
from hypervolume2d import hypervolume_indicator
from hypervolume2d import remove_dominated_points
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def phiplus(points, L, R, U):
    if L is None or U is None or R is None:
        raise ValueError("L, R, and U must all be provided")

    # Let us check if the reference point R is within the bounds [L, U]
    if not (L[0] <= R[0] <= U[0] and L[1] <= R[1] <= U[1]):
        raise ValueError("Reference point R must be within the bounds [L, U]")

    # Let us check if the reference point R is not dominated by any of the points
    if any(point[0] < R[0] and point[1] < R[1] for point in points):
        reference_area = (U[0] - R[0]) * (U[1] - R[1])
    else:
        reference_area = (U[0]- L[0]) * (U[1] - L[1]) - (R[0]-L[0]) * (R[1]-L[1])

    print(f"Reference area: {reference_area}")
    # # Filter points within the bounds [L, U]
    # truncated_points = [point for point in points if L[0] <= point[0] <= U[0] and L[1] <= point[1] <= U[1]]

    # Remove points that are outside the ROI
    filtered_points = [point for point in points if (point[0] <= R[0] and point[1] <= R[1]) or (L[0] <= point[0] <= U[0] and L[1] <= point[1] <= U[1])]
    # Compute the hypervolume indicator for the filtered points with upper bound U
    hv_filtered_points = hypervolume_indicator(filtered_points, U)
    print(f"Filtered points: {filtered_points}")
    print(f"Hypervolume of filtered points: {hv_filtered_points}")

    if reference_area <= 0:
        return 0
    else:
        return hv_filtered_points / reference_area

def plot_points(points, L, R, U):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')

    # Plot grid
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 11, 1))
    ax.grid(True)
    ax.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)

    # Shade the dominated regions
    for point in points:
        if ((point[0] <= U[0] and point[1] <=  U[1]) and \
                (point[0] >= L[0] and point[1] >= L[1])) \
            and ((point[0] >= R[0] and point[1] >= R[1]) \
            or  (point[0] <= R[0] and point[1] > R[1] or \
                (point[0] > R[0] and point[1] <= R[1])) \
            or (point[0] <= R[0] and point[1] <= R[1])
                ):
            rect = patches.Rectangle(point, U[0]-point[0], U[1]-point[1], linewidth=0, edgecolor='none', facecolor='gray',
                                     alpha=0.6)
            ax.add_patch(rect)

    # Plot points
    for point in points:
        ax.plot(point[0], point[1], 'bo')

    # Highlight reference point
    ax.plot(R[0], R[1], 'ro', label='Reference Point R')

    # Draw bounds
    rect = patches.Rectangle(L, U[0] - L[0], U[1] - L[1], linewidth=3, edgecolor='blue', facecolor='none',
                             linestyle='--', label='Bounds [L, U]')
    ax.add_patch(rect)

    #plt.legend()
    plt.title("Calculation of Phiplus")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.show()


def main():
    points = [
        (9, 1),
        (5, 2),
        (4, 3),
        (3, 7),
        (1, 8)
    ]

    # Define lower bound, reference point, and upper bound
    L = (2, 2)
    R = (6, 6)
    U = (10, 10)

    # Calculate hypervolume indicator
    hv = phiplus(points, L=L, R=R, U=U)
    print(f"Hypervolume indicator with L={L}, R={R}, and U={U}: {hv}")

    # Plot points and shaded regions
    plot_points(points, L, R, U)

    # Second example
    points = [
        (8, 3),
        (4, 4),
        (3, 5),
        (12, 3),
        (1, 7)
        ]

    L = (0, 0)
    R = (6, 4)
    U = (10,6)

    hv = phiplus(points, L=L, R=R, U=U)
    print(f"Hypervolume indicator with L={L}, R={R}, and U={U}: {hv}")
    # plot
    plot_points(points, L, R, U)


if __name__ == "__main__":
    main()
