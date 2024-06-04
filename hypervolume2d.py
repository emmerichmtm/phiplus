import numpy as np

def remove_dominated_points(points):
    # iterate over all points i, j
    marker_for_removal = []
    for i in range(len(points)):
       for j in range(len(points)):
             if (points[i][0] < points[j][0] and points[i][1] <= points[j][1]
                 or points[i][0] <= points[j][0] and points[i][1] < points[j][1]):
                #print(f"point {i} dominates point {j}")
                # mark point j for removal
                marker_for_removal.append(j)
    #print(marker_for_removal)
    # copy points that are not in marker_for_removal to new list
    points = [points[i] for i in range(len(points)) if i not in marker_for_removal]
    return points

def hypervolume_indicator(points, reference_point):
    #"""
    #Compute the 2-D Hypervolume Indicator.
    #Args:
    #- points: A list of tuples representing the points in the objective space.
    #- reference_point: A tuple representing the reference point.
    #Returns:
    #- The 2-D Hypervolume Indicator.
    #"""

    # Remove dominated points
    points = remove_dominated_points(points)
    points = np.array(points)

    reference_point = np.array(reference_point)

    # Calculate the dominated points
    dominated_points = [point for point in points if all(point <= reference_point)]

    # Sort dominated points by the second objective (f2)
    dominated_points.sort(key=lambda x: x[0])

    hypervolume = 0.0
    last_point = reference_point

    # Iterate through the sorted dominated points to calculate hypervolume
    for point in dominated_points:
        if last_point is not None:
            height = last_point[1] - point[1]
            width = reference_point[0] - point[0]
            hypervolume += height * width
        last_point = point

    return hypervolume
