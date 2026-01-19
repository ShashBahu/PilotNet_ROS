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
import csv
import os

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

        self.run_id = int(time.time())
        log_dir = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs")
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"RCB_Total_Time_log.csv")

        self.csv_f = open(self.csv_path, "w", newline="")
        self.csv_w = csv.writer(self.csv_f)
        self.csv_w.writerow([
            "seq",
            # "t_cam_cb_ns",
            # "t_req_send_ns",
            # "t_resp_done_ns",
            # "t_apply_ns",
            "lat_service_ms",
            # "lat_e2e_ms",
        ])
        self.csv_f.flush()
        self.i = 0
        self.image_queue = queue.Queue(maxsize=1)
        def _cam_cb(img):
            t_cam_cb_ns = time.perf_counter_ns()
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                pass
            # push newest frame
            try:
                self.image_queue.put_nowait((img, t_cam_cb_ns))
            except queue.Full:
                pass
        self.camera.listen(_cam_cb)

    def init_carla(self):
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        self.world = client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
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
        camera_bp.set_attribute("sensor_tick", "0.1")
        camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.base_cam_transform = camera_init_trans
        # camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7), carla.Rotation(pitch=0,yaw=0,roll=6)) #For fault injection: orientation fault
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)

    def process_image_and_request_steering(self, image: carla.Image, t_cam_cb_ns: int):

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

        # sent_time = time.time()
        # print(f"Seq: {ros_msg.header.frame_id} | Sent at: {sent_time}")

        # # --- SERVICE CALL: block until steering is returned ---
        # request = InferSteering.Request()
        # request.image = ros_msg
        
        # future = self.steering_client.call_async(request)
        # rclpy.spin_until_future_complete(self, future)
        # received_time = time.time()
        # # if future.result() is None:
        # #     self.get_logger().error('Service call failed, using zero steering')
        # #     steering_cmd = 0.0
        # # else:
        # steering_cmd = float(future.result().steering)

        # # Apply steering command to CARLA
        # control = carla.VehicleControl()
        # control.steer = steering_cmd
        # control.throttle = 0.5
        # self.vehicle.apply_control(control)

        
        # latency = received_time - sent_time
        # print(f"Steering: {steering_cmd:.4f} | Latency: {latency:.4f}s")
        seq = int(ros_msg.header.frame_id)

        
        request = InferSteering.Request()
        request.image = ros_msg
        t_req_send_ns = time.perf_counter_ns()
        future = self.steering_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        t_resp_done_ns = time.perf_counter_ns()

        steering_cmd = float(future.result().steering)

        # Apply steering command to CARLA
        control = carla.VehicleControl()
        control.steer = steering_cmd
        control.throttle = 0.3
        self.vehicle.apply_control(control)
        # t_apply_ns = time.perf_counter_ns()

        lat_service_ms = (t_resp_done_ns - t_req_send_ns) / 1e6
        # lat_e2e_ms = (t_apply_ns - t_cam_cb_ns) / 1e6

        print(f"Seq: {seq} | Steer: {steering_cmd:.4f} | service_ms: {lat_service_ms:.2f}")

        # write one row per frame
        self.csv_w.writerow([
            seq,
            # t_cam_cb_ns,
            # t_req_send_ns,
            # t_resp_done_ns,
            # t_apply_ns,
            lat_service_ms,
            # lat_e2e_ms,
        ])

        # flush occasionally so you don't lose data if it crashes
        if seq % 50 == 0:
            self.csv_f.flush()

    

def main(args=None):
    rclpy.init(args=args)
    node = CarlaDriverNode()

    try:
        # Prime the world once
        #node.world.tick()
        oldtime = time.time()

        while rclpy.ok():
            # 1) Get the latest camera frame (produced by previous tick)
            image, t_cam_cb_ns = node.image_queue.get()

            # 2) Send to model service and wait for steering
            node.process_image_and_request_steering(image, t_cam_cb_ns)

            # 3) Advance CARLA world *only after* model returned steering
            #time.sleep(1)
            #node.world.tick()

            elapsedtime = time.time() - oldtime
            print("Time elapsed per step: ", elapsedtime)
            oldtime = time.time()

    except KeyboardInterrupt:
        print("Shutting down node.")

    except Exception as e:
        print("Exception: ", e)
        
    finally:
        try:
            node.csv_f.flush()
            node.csv_f.close()
            print(f"Saved latency log to: {node.csv_path}")
        except Exception as e:
            print("CSV close error:", e)

        try:
            settings = node.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            node.world.apply_settings(settings)
        except:
            pass

        try: node.camera.stop()
        except: pass
        try: node.vehicle.destroy()
        except: pass
        try: node.destroy_node()
        except: pass
        try: rclpy.shutdown()
        except: pass


if __name__ == "__main__":
    main()