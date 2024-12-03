import numpy as np
import carla
from TTC import compute_ttc, compute_ttc_vehicle_to_rectangle


def classify_ttc(ttc, divide=1.0):
    """Classify TTC into categories."""
    if ttc < divide:
        return "very short"
    elif ttc < divide * 2:
        return "short"
    elif ttc < divide * 3:
        return "moderate"
    elif ttc < divide * 5:
        return "long"
    else:
        return "very long"


def get_rectangle_vertices(waypoint):
    """Calculate the vertices of a restricted zone rectangle."""
    transform = waypoint.transform
    location = transform.location
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    lane_width = waypoint.lane_width
    length = 5.0
    half_width = lane_width / 2
    vertices = [
        location + forward * length / 2 + right * half_width,
        location + forward * length / 2 - right * half_width,
        location - forward * length / 2 + right * half_width,
        location - forward * length / 2 - right * half_width
    ]
    return vertices


def angle_to_direction(yaw):
    """Convert a yaw angle to a compass direction."""
    yaw = yaw % 360
    directions = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    index = int((yaw + 22.5) % 360 // 45)
    return directions[index]


def velocity_to_direction(velocity):
    """Convert velocity vector to a compass direction."""
    if np.linalg.norm([velocity.x, velocity.y, velocity.z]) < 0.1:
        return "stationary"
    angle = np.degrees(np.arctan2(velocity.y, velocity.x))
    return angle_to_direction(angle)


def weather_to_text(weather):
    """Convert weather data to a descriptive text."""
    if weather.precipitation > 0:
        return "rainy"
    if weather.cloudiness > 50:
        return "cloudy"
    if weather.fog_density > 0:
        return "foggy"
    return "sunny"


def calculate_lane_count(waypoint):
    """Calculate the total number of lanes at the waypoint."""
    lane_count = 1
    right_wp = waypoint.get_right_lane()
    while right_wp and right_wp.lane_type == carla.LaneType.Driving:
        lane_count += 1
        right_wp = right_wp.get_right_lane()
    left_wp = waypoint.get_left_lane()
    while left_wp and left_wp.lane_type == carla.LaneType.Driving:
        lane_count += 1
        left_wp = left_wp.get_left_lane()
    return lane_count


def get_road_type(waypoint):
    """
    Determine the RoadType based on the waypoint and its surroundings.
    Returns one of intersection, divided, undivided, two-way.
    """
    # Check if the waypoint is part of an intersection
    if waypoint.is_junction:
        return "intersection"
    # Check a lane type and driving direction
    left_lane = waypoint.get_left_lane()
    right_lane = waypoint.get_right_lane()

    if left_lane and right_lane:
        if left_lane.lane_type == carla.LaneType.Driving and right_lane.lane_type == carla.LaneType.Driving:
            # Check if the road is divided
            if left_lane.lane_id * waypoint.lane_id < 0:
                return "divided"
            else:
                return "undivided"
    # Default to two-way for bidirectional roads without specific division
    return "two-way"


def get_road_slope(waypoint):
    """
    Calculate the RoadSlope based on the waypoint's elevation and nearby points.
    Returns one of flat, uphill, downhill, level.
    """
    # Get the current waypoint location
    current_location = waypoint.transform.location

    # Get a waypoint slightly ahead on the same lane
    forward_waypoint = waypoint.next(5.0)[0]  # 5 meters ahead
    forward_location = forward_waypoint.transform.location

    # Calculate slope using the difference in z-coordinates
    slope = (forward_location.z - current_location.z) / 5.0  # Slope per meter

    # Classify the slope
    if abs(slope) < 0.01:  # Small threshold for flat
        return "flat"
    elif slope > 0.01:
        return "uphill"
    elif slope < -0.01:
        return "downhill"
    else:
        return "level"


class DSL2Parser:
    def __init__(self, world):
        self.world = world
        self.npcs = []
        self.ads = None
        self.previous_velocities = {}

    def set_ads(self, ads_vehicle):
        """Set the ADS vehicle."""
        self.ads = ads_vehicle

    def get_actors(self):
        """Retrieve all vehicles (NPCs) in the scene."""
        self.npcs = self.world.get_actors().filter('vehicle.*')

    def get_road_info(self):
        """Extract road information including RoadType, Lanes, RoadSlope, and SpeedLimit."""
        map_data = self.world.get_map()
        waypoint = map_data.get_waypoint(self.ads.get_location())
        lane_count = calculate_lane_count(waypoint)
        road_info = {
            "RoadType": get_road_type(waypoint),
            "Lanes": None if lane_count == 0 else [f"Lane {i}" for i in range(1, lane_count + 1)],
            "RoadSlope": get_road_slope(waypoint),
            "SpeedLimit": self.get_speed_limit()
        }
        return road_info

    def parse_scene(self):
        """Parse the entire scene and return the DSL2 description."""
        scene_data = {
            "Road": self.get_road_info(),
            "Environment": self.get_environment_info(),
            "NPCs": [self.get_npc_behavior(npc) for npc in self.npcs],
            "ADS": self.get_ads_behavior()
        }
        return scene_data

    def get_lane_departure_status(self, vehicle):
        """Calculate the lane departure status."""
        vehicle_location = vehicle.get_location()
        waypoint = self.world.get_map().get_waypoint(vehicle_location)
        lane_center = waypoint.transform.location
        lane_departure_distance = np.linalg.norm([
            vehicle_location.x - lane_center.x,
            vehicle_location.y - lane_center.y
        ])
        lane_width = waypoint.lane_width
        align_threshold = lane_width / 6
        deviate_threshold = lane_width / 3
        if lane_departure_distance <= align_threshold:
            return "Align"
        elif lane_departure_distance <= deviate_threshold:
            return "Deviate"
        else:
            return f"Depart ({lane_departure_distance:.2f} m)"

    def get_speed_limit(self):
        """Get the speed limit near the ADS vehicle."""
        waypoint = self.world.get_map().get_waypoint(self.ads.get_location())
        return waypoint.get_speed_limit()

    def get_environment_info(self):
        """Extract environment information."""
        weather = self.world.get_weather()
        road_condition = "wet" if weather.precipitation > 0 else "dry"
        if weather.fog_density > 50:
            road_condition = "icy"
        elif weather.snow > 0:
            road_condition = "snow-covered"
        return {
            "Weather": weather_to_text(weather),
            "RoadCondition": road_condition,
            "TrafficSignals": self.get_traffic_signal_state()
        }

    def get_traffic_signal_state(self):
        """Retrieve the traffic signal state near the ADS vehicle."""
        waypoint = self.world.get_map().get_waypoint(self.ads.get_location())
        traffic_lights = self.world.get_traffic_lights_from_waypoint(waypoint, 30.0)
        for light in traffic_lights:
            if light.is_active:
                return light.state.name.lower()
        return "none"

    def get_npc_behavior(self, npc):
        """Extract the behavior of a single NPC."""
        transform = npc.get_transform()
        velocity = npc.get_velocity()
        control = npc.get_control()
        lane = self.world.get_map().get_waypoint(transform.location)

        # Determine the type of NPC
        npc_type = "motorized vehicle"
        if velocity.x == 0 and velocity.y == 0 and velocity.z == 0:
            npc_type = "roadblock"  # Static obstacles
        # Optionally, add more logic to classify debris, animals, etc.

        # Simplify LaneDeparture to match DSL
        lane_departure = self.get_lane_departure_status(npc)
        if "Depart" in lane_departure:
            lane_departure = "Depart"

        # Behavior information
        behavior = {
            "Type": npc_type,
            "SideToADS": self.get_side_to_ads(npc),
            "HeadDirection": angle_to_direction(transform.rotation.yaw),
            "SpeedDirection": velocity_to_direction(velocity),
            "Speed": np.linalg.norm([velocity.x, velocity.y, velocity.z]),
            "MovingOnWhichWay": lane.lane_id,
            "MovingToWhichWay": lane.lane_id,
            "SteeringAngle": control.steer,
            "PriorAction": "Maneuver" if abs(control.steer) > 0.1 else "Stable",
            "Accelerate": self.calculate_acceleration(npc),
            "LaneDeparture": lane_departure,
            "TTCToADS": self.calculate_ttc_to_ads(npc)
        }
        return behavior

    def get_ads_behavior(self):
        """Extract the behavior of the ADS vehicle."""
        transform = self.ads.get_transform()
        velocity = self.ads.get_velocity()
        control = self.ads.get_control()
        lane = self.world.get_map().get_waypoint(transform.location)
        return {
            "HeadDirection": angle_to_direction(transform.rotation.yaw),
            "SpeedDirection": velocity_to_direction(velocity),
            "Speed": np.linalg.norm([velocity.x, velocity.y, velocity.z]),
            "MovingOnWhichWay": lane.lane_id,
            "MovingToWhichWay": lane.lane_id,
            "SteeringAngle": control.steer,
            "PriorAction": "Maneuver" if abs(control.steer) > 0.1 else "Stable",
            "Accelerate": self.calculate_acceleration(self.ads),
            "LaneDeparture": self.get_lane_departure_status(self.ads),
            "TTCToNPCs": self.calculate_ttc_to_npcs(),
            "TTCToRestrictedZone": self.calculate_ttc_to_restricted_zone()
        }

    def calculate_ttc_to_ads(self, npc):
        """Calculate TTC between the given NPC and the ADS vehicle."""
        ttc = compute_ttc(
            p_A=npc.get_location(),
            v_A=npc.get_velocity(),
            steer_A=npc.get_control().steer,
            p_B=self.ads.get_location(),
            v_B=self.ads.get_velocity(),
            steer_B=self.ads.get_control().steer,
            d=2.0
        )
        return classify_ttc(ttc)

    def calculate_ttc_to_npcs(self):
        """Calculate the minimum TTC between the ADS and any NPC."""
        min_ttc = float('inf')
        for npc in self.npcs:
            ttc = compute_ttc(
                p_A=self.ads.get_location(),
                v_A=self.ads.get_velocity(),
                steer_A=self.ads.get_control().steer,
                p_B=npc.get_location(),
                v_B=npc.get_velocity(),
                steer_B=npc.get_control().steer,
                d=2.0
            )
            if ttc < min_ttc:
                min_ttc = ttc
        return classify_ttc(min_ttc)

    def calculate_ttc_to_restricted_zone(self):
        """Calculate TTC between the ADS and the nearest restricted zone."""
        min_ttc = float('inf')
        map_data = self.world.get_map()
        waypoint = map_data.get_waypoint(self.ads.get_location())
        restricted_zones = []
        right_wp = waypoint.get_right_lane()
        while right_wp:
            if right_wp.lane_type != carla.LaneType.Driving:
                restricted_zones.append(right_wp)
            right_wp = right_wp.get_right_lane()
        left_wp = waypoint.get_left_lane()
        while left_wp:
            if left_wp.lane_type != carla.LaneType.Driving:
                restricted_zones.append(left_wp)
            left_wp = left_wp.get_left_lane()
        for zone in restricted_zones:
            rect_vertices = get_rectangle_vertices(zone)
            ttc = compute_ttc_vehicle_to_rectangle(
                p_car=self.ads.get_location(),
                v_car=self.ads.get_velocity(),
                a_car=self.ads.get_control().steer,
                r_car=2.0,
                rect_vertices=rect_vertices
            )
            if ttc < min_ttc:
                min_ttc = ttc
        return classify_ttc(min_ttc) if min_ttc != float('inf') else "very long"

    def calculate_acceleration(self, npc):
        """Calculate the acceleration status."""
        velocity = np.linalg.norm([npc.get_velocity().x, npc.get_velocity().y, npc.get_velocity().z])
        prev_velocity = self.previous_velocities.get(npc.id, 0)
        self.previous_velocities[npc.id] = velocity
        acceleration = velocity - prev_velocity
        if acceleration > 0:
            return "Accelerating"
        elif acceleration < 0:
            return "Braking"
        return "constant-speed"

    def get_side_to_ads(self, npc):
        """
        Determine the relative position of the NPC to the ADS.
        Returns one of front, rear, left, right, front-left, front-right, rear-left, rear-right.
        """
        ads_location = self.ads.get_location()
        npc_location = npc.get_location()
        diff = npc_location - ads_location

        # Calculate the angle in degrees from ADS to NPC
        angle = np.degrees(np.arctan2(diff.y, diff.x)) % 360

        # Define direction categories
        directions = [
            "front", "front-right", "right", "rear-right",
            "rear", "rear-left", "left", "front-left"
        ]

        # Map the angle to one of the 8 categories
        index = int((angle + 22.5) % 360 // 45)
        return directions[index]


# Example usage
if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    vehicles = world.get_actors().filter('vehicle.*')
    ads_vehicle = vehicles[0]
    parser = DSL2Parser(world)
    parser.set_ads(ads_vehicle)
    parser.get_actors()
    scene = parser.parse_scene()
    print(scene)
