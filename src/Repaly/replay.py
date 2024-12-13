import numpy as np
import carla
import math
import argparse
import sys

# Function to calculate the desired velocity vector based on the target point
def calculate_velocity(current_position, target_position, max_speed=10.0):
    direction = target_position - current_position
    distance = np.linalg.norm(direction)

    if distance > 0:
        direction = direction / distance  # Normalize the direction vector
        speed = min(max_speed, distance * 2)  # Speed is proportional to distance but capped at max_speed
        velocity = direction * speed
    else:
        velocity = np.array([0.0, 0.0])  # Stop if already at the target point

    return velocity


# Function to apply the velocity to the actor vehicle in CARLA
def apply_velocity(actor_vehicle, velocity):
    # Set the target velocity for the vehicle in the CARLA simulation
    carla_velocity = carla.Vector3D(velocity[0], velocity[1], 0.0)  # Z remains zero for 2D motion
    actor_vehicle.set_target_velocity(carla_velocity)


# Function to plan the movement along the path
def kinematic_path_planning(actor_vehicle, path, max_speed=10.0, stop_threshold=0.1):
    current_position = np.array([actor_vehicle.get_location().x, actor_vehicle.get_location().y])
    path_index = 0

    # Iterate through the path until reaching the goal or max iterations
    while path_index < len(path):
        # Get the target point from the path
        target_point = path[path_index]

        # Calculate the velocity to move toward the target point
        velocity = calculate_velocity(current_position, target_point, max_speed)

        # Apply the calculated velocity to the vehicle
        apply_velocity(actor_vehicle, velocity)

        # Update the current position (this would typically be updated each frame)
        current_position = np.array([actor_vehicle.get_location().x, actor_vehicle.get_location().y])

        # Check if the vehicle has reached the target point
        if np.linalg.norm(current_position - target_point) < stop_threshold:
            path_index += 1  # Move to the next point in the path

    # When the loop finishes, stop the vehicle (if needed)
    actor_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))  # Stop the vehicle


# Function to initialize the simulation with weather, friction, and other configurations
def initialize_simulation(client, args):
    # Set weather conditions based on arguments
    weather = carla.WeatherParameters(
        cloudiness=args.cloud,
        precipitation=args.rain,
        puddles=args.puddle,
        wind_intensity=args.wind,
        fog_density=args.fog,
        wetness=args.wetness,
        sun_altitude_angle=args.altitude,
        sun_azimuth_angle=args.angle
    )
    client.get_world().set_weather(weather)


# Function to parse the arguments and create the config
def parse_args():
    parser = argparse.ArgumentParser(description="Configure CARLA simulation")
    parser.add_argument("--cloud", type=int, default=0, help="Cloudiness level (0-100)")
    parser.add_argument("--rain", type=int, default=0, help="Rain level (0-100)")
    parser.add_argument("--puddle", type=int, default=0, help="Puddle level (0-100)")
    parser.add_argument("--wind", type=int, default=0, help="Wind intensity (0-100)")
    parser.add_argument("--fog", type=int, default=0, help="Fog density (0-100)")
    parser.add_argument("--wetness", type=int, default=0, help="Wetness level (0-100)")
    parser.add_argument("--altitude", type=int, default=0, help="Sun altitude angle")
    parser.add_argument("--angle", type=int, default=0, help="Sun azimuth angle")
    parser.add_argument("--spawn", nargs=6, type=float, help="Spawn location (x, y, z, pitch, yaw, roll)")
    parser.add_argument("--dest", nargs=3, type=float, help="Destination location (x, y, z)")
    parser.add_argument("--speed", type=float, default=10.0, help="Target speed of the vehicle")

    return parser.parse_args()


# Main function to set up the simulation and run it
if __name__ == '__main__':
    # Parse arguments for configuration
    args = parse_args()

    # Connect to the CARLA server
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Initialize simulation with weather and vehicle configuration
    initialize_simulation(client, args)

    # Create the spawn point for the vehicle
    if args.spawn:
        spawn_point = carla.Transform(carla.Location(args.spawn[0], args.spawn[1], args.spawn[2]),
                                      carla.Rotation(args.spawn[3], args.spawn[4], args.spawn[5]))
    else:
        spawn_point = carla.Transform(carla.Location(334.83, 217.1, 1.32), carla.Rotation(0.0, 90, 0.0))

    # Spawn a vehicle in the world
    blueprint = client.get_world().get_blueprint_library().filter('vehicle.*')[0]
    actor_vehicle = client.get_world().spawn_actor(blueprint, spawn_point)

    # Example path
    path = np.array([[0, 0], [10, 10], [20, 20]])

    # Perform kinematic path planning
    kinematic_path_planning(actor_vehicle, path)

    # Clean up: destroy the actor after the simulation
    actor_vehicle.destroy()

    print("Simulation completed.")
