# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32
# from std_msgs.msg import Header
# from sensor_msgs.msg import Image
# import numpy as np
# import carla
# from carla import WeatherParameters
# import time
# import queue
 
# class CarlaDriverNode(Node):
#     def __init__(self):
#         super().__init__('carla_driver_node')

#         # ROS 2 publishers / subscribers
#         self.image_pub = self.create_publisher(Image, '/carla/camera/image_cropped', 1)
#         self.steering_sub = self.create_subscription(Float32, '/carla/steering_cmd', self.steering_callback, 1)

#         # Initialize CARLA
#         self.init_carla()
#         self.i = 0
#         self.latency=0
#         self.sentTime=time.time()
#         self.receivedTime=time.time()
#         # Start camera listener
#         #self.camera.listen(self.camera_callback)
#         self.image_queue = queue.Queue()
#         self.camera.listen(self.image_queue.put)

#     def init_carla(self):
#         client = carla.Client("localhost", 2000)
#         client.set_timeout(10.0)

#         self.world = client.get_world()
#         settings = self.world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = 0.2  # 100 Hz
#         self.world.apply_settings(settings)

#         weather = WeatherParameters(
#         sun_altitude_angle=90.0,  # High noon = minimal shadows
#         sun_azimuth_angle=0.0,
#         cloudiness=0.0,           # Clear sky (no cloud shadows)
#         precipitation=0.0,
#         fog_density=0.0,
#         wetness=0.0
#         )
#         self.world.set_weather(weather)
#         # client.reload_world()
#         blueprint_library = self.world.get_blueprint_library()

#         # Spawn vehicle
#         bp = blueprint_library.find('vehicle.dodge.charger_police')
#         spawn_point = self.world.get_map().get_spawn_points()[144]
#         self.vehicle = self.world.spawn_actor(bp, spawn_point)

#         # Camera setup
#         camera_bp = blueprint_library.find("sensor.camera.rgb")
#         camera_bp.set_attribute("image_size_x", "720")
#         camera_bp.set_attribute("image_size_y", "405")
#         camera_bp.set_attribute("fov", "110")
#         camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
#         self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)

#         # Initialize steering command
#         self.current_steering = 0.0

#     def camera_callback(self, image):
#         # Convert raw CARLA image to numpy
#         img = np.frombuffer(image.raw_data, dtype=np.uint8)
#         img = img.reshape((image.height, image.width, 4))[:, :, :3]

#         # Crop only the region of interest
#         h, w, _ = img.shape
#         img_cropped = img[int(0.5*h):int(0.8*h), int(0.3*w):int(0.7*w), :]

#         # Convert to ROS Image message manually (no cv_bridge)
#         ros_msg = Image()
#         ros_msg.height, ros_msg.width = img_cropped.shape[:2]
#         ros_msg.encoding = "rgb8"
#         ros_msg.data = img_cropped.tobytes()
#         ros_msg.step = ros_msg.width * 3

#         self.i=self.i+1

#         ros_msg.header = Header()
#         ros_msg.header.frame_id = str(self.i)
#         #ros_msg.header.stamp = time.time()

#         self.sentTime=time.time()
#         print("Seq: ", ros_msg.header.frame_id, " | Time: ", self.sentTime)
#         #print(time.time())
#         self.image_pub.publish(ros_msg)

#         # Apply steering command
#         control = carla.VehicleControl()
#         control.steer = self.current_steering
#         control.throttle = 0.35
#         self.vehicle.apply_control(control)

#     def steering_callback(self, msg):
#         # Update steering command from model server
#         self.current_steering = msg.data
#         #self.receivedTime = time.time()
#         #self.latency = self.receivedTime - self.sentTime
#         #print("Latency: ", self.latency)

# def main(args=None):
#     rclpy.init(args=args)
#     node = CarlaDriverNode()
#     try:
#         #rclpy.spin(node)
#         node.world.tick()
#         oldtime=time.time()
#         while True:
#             image = node.image_queue.get()
#             node.camera_callback(image)
#             node.world.tick()
#             elapsedtime=time.time()-oldtime
#             print("Time elapsed: ", elapsedtime)
#             oldtime=time.time()

#     except KeyboardInterrupt:
#         print("Shutting down node.")
#     finally:
#         node.camera.stop()
#         node.vehicle.destroy()
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from std_msgs.msg import Header
from sensor_msgs.msg import Image

import numpy as np
import carla
from carla import WeatherParameters
import time
import queue

from carlaros_interfaces.srv import InferSteering   # <--- custom service

# img_save = 0 

class CarlaDriverNode(Node):
    def __init__(self):
        super().__init__('carla_driver_node')

        # Service client to query steering from the model node
        self.steering_client = self.create_client(InferSteering, '/carla/predict_steering')
        while not self.steering_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /carla/predict_steering service...')

        # Initialize CARLA
        self.init_carla()

        self.i = 0
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)

    def init_carla(self):
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        self.world = client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.2
        self.world.apply_settings(settings)

        weather = WeatherParameters(
            sun_altitude_angle=90.0,
            sun_azimuth_angle=0.0,
            cloudiness=0.0,
            precipitation=0.0,
            fog_density=0.0,
            wetness=0.0
        )
        self.world.set_weather(weather)

        blueprint_library = self.world.get_blueprint_library()

        # Spawn vehicle
        bp = blueprint_library.find('vehicle.dodge.charger_police')
        spawn_point = self.world.get_map().get_spawn_points()[6] #RT-LL 144; LT-LL 6
        self.vehicle = self.world.spawn_actor(bp, spawn_point)

        # Camera setup
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "720")
        camera_bp.set_attribute("image_size_y", "405")
        camera_bp.set_attribute("fov", "110")
        camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.base_cam_transform = camera_init_trans
        # camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7), carla.Rotation(pitch=0,yaw=0,roll=6)) #For fault injection: orientation fault
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)

    def process_image_and_request_steering(self, image: carla.Image):
        """
        Convert CARLA image → cropped ROS Image → call model service → apply steering.
        This is called once per simulation step from main().
        """
        # img_save = img_save + 1
        # Convert raw CARLA image to numpy
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))[:, :, :3]

        # Crop only the region of interest
        h, w, _ = img.shape
        img_cropped = img[int(0.5*h):int(0.8*h), int(0.3*w):int(0.7*w), :]
        
        # Convert to ROS Image message manually (no cv_bridge)
        ros_msg = Image()
        ros_msg.height, ros_msg.width = img_cropped.shape[:2]
        ros_msg.encoding = "rgb8"
        ros_msg.data = img_cropped.tobytes()
        ros_msg.step = ros_msg.width * 3

        self.i += 1
        ros_msg.header = Header()
        ros_msg.header.frame_id = str(self.i)

        sent_time = time.time()
        print(f"Seq: {ros_msg.header.frame_id} | Sent at: {sent_time}")

        # --- SERVICE CALL: block until steering is returned ---
        request = InferSteering.Request()
        request.image = ros_msg
        
        future = self.steering_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        received_time = time.time()
        # if future.result() is None:
        #     self.get_logger().error('Service call failed, using zero steering')
        #     steering_cmd = 0.0
        # else:
        steering_cmd = float(future.result().steering)

        # Apply steering command to CARLA
        control = carla.VehicleControl()
        control.steer = steering_cmd
        control.throttle = 0.5
        self.vehicle.apply_control(control)

        
        latency = received_time - sent_time
        print(f"Steering: {steering_cmd:.4f} | Latency: {latency:.4f}s")
    

def main(args=None):
    rclpy.init(args=args)
    node = CarlaDriverNode()

    try:
        # Prime the world once
        node.world.tick()
        oldtime = time.time()

        while rclpy.ok():
            # 1) Get the latest camera frame (produced by previous tick)
            image = node.image_queue.get()

            # 2) Send to model service and wait for steering
            node.process_image_and_request_steering(image)

            # 3) Advance CARLA world *only after* model returned steering
            #time.sleep(1)
            node.world.tick()

            elapsedtime = time.time() - oldtime
            print("Time elapsed per step: ", elapsedtime)
            oldtime = time.time()

    except KeyboardInterrupt:
        print("Shutting down node.")
        settings = node.world.get_settings()
        settings.synchronous_mode = False # Disables synchronous mode
        settings.fixed_delta_seconds = None
        node.world.apply_settings(settings)

        node.camera.stop()
        node.vehicle.destroy()
        node.destroy_node()
        rclpy.shutdown()

    except Exception as e:
        print("Exception: ", e)
        settings = node.world.get_settings()
        settings.synchronous_mode = False # Disables synchronous mode
        settings.fixed_delta_seconds = None
        node.world.apply_settings(settings)

        node.camera.stop()
        node.vehicle.destroy()
        node.destroy_node()
        rclpy.shutdown() 
    finally:
        node.camera.stop()
        node.vehicle.destroy()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
