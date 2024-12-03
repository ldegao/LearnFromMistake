import numpy as np


def compute_ttc(p_A, v_A, steer_A, p_B, v_B, steer_B, d, epsilon=1e-6, max_time=50, time_steps=5000):
    """
    Compute TTC (Time to Collision) for linear and circular trajectories using incremental search.

    Parameters:
    - p_A, v_A, steer_A: Position, velocity, and steering angle of object A
    - p_B, v_B, steer_B: Position, velocity, and steering angle of object B
    - d: Collision threshold distance
    - epsilon: Tolerance for identifying straight-line motion
    - max_time: Maximum time range for TTC computation
    - time_steps: Number of steps for incremental time search

    Returns:
    - TTC: Time to collision, or None if no collision occurs
    """
    times = np.linspace(0, max_time, time_steps)
    R_A, R_B = None, None
    c_A, c_B = None, None

    # Compute circular motion parameters
    if abs(steer_A) > epsilon:
        R_A = np.linalg.norm(v_A) / np.tan(steer_A)
        c_A = p_A + np.array([-R_A * v_A[1], R_A * v_A[0]]) / np.linalg.norm(v_A)

    if abs(steer_B) > epsilon:
        R_B = np.linalg.norm(v_B) / np.tan(steer_B)
        c_B = p_B + np.array([-R_B * v_B[1], R_B * v_B[0]]) / np.linalg.norm(v_B)

    # Incrementally compute positions
    for t in times:
        if R_A is not None:  # A is circular
            theta_A = t * np.linalg.norm(v_A) / R_A
            pos_A = c_A + R_A * np.array([np.cos(theta_A), np.sin(theta_A)])
        else:  # A is linear
            pos_A = p_A + v_A * t

        if R_B is not None:  # B is circular
            theta_B = t * np.linalg.norm(v_B) / R_B
            pos_B = c_B + R_B * np.array([np.cos(theta_B), np.sin(theta_B)])
        else:  # B is linear
            pos_B = p_B + v_B * t

        # Check distance
        if np.linalg.norm(pos_A - pos_B) <= d:
            return t  # Return the first time collision occurs

    return None  # No collision detected within the time range


def compute_ttc_vehicle_to_rectangle(p_car, v_car, a_car, r_car, rect_vertices, max_time=50, time_steps=5000):
    """
    Compute the Time to Collision (TTC) between a circular vehicle and a rectangular obstacle.

    Parameters:
    - p_car: Initial position of the vehicle (numpy array)
    - v_car: Velocity of the vehicle (numpy array)
    - a_car: Acceleration of the vehicle (numpy array)
    - r_car: Radius of the vehicle
    - rect_vertices: List of vertices defining the rectangle (list of numpy arrays)
    - max_time: Maximum time to consider for collision
    - time_steps: Number of steps for incremental time search

    Returns:
    - t_collision: Time to collision, or None if no collision occurs
    """

    # Define the vehicle trajectory
    def vehicle_position(t):
        return p_car + v_car * t + 0.5 * a_car * t ** 2

    # Compute the distance to a rectangle edge
    def distance_to_edge(t, p1, p2):
        car_pos = vehicle_position(t)
        edge_vec = p2 - p1
        s = np.clip(np.dot(car_pos - p1, edge_vec) / np.dot(edge_vec, edge_vec), 0, 1)
        closest_point = p1 + s * edge_vec
        return np.linalg.norm(car_pos - closest_point) - r_car

    # Compute the minimum distance to the rectangle
    def min_distance_to_rectangle(t):
        distances = [distance_to_edge(t, rect_vertices[i], rect_vertices[(i + 1) % len(rect_vertices)])
                     for i in range(len(rect_vertices))]
        return min(distances)

    # Check for collision over time
    times = np.linspace(0, max_time, time_steps)
    for t in times:
        if min_distance_to_rectangle(t) <= 0:
            return t  # Collision time
    return None  # No collision within the time range
