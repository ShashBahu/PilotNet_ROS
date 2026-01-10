import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import numpy as np
import tensorflow as tf
import time
import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

print("Is GPU available:", tf.config.list_physical_devices('GPU'))

class ModelServerNode(Node):
    def __init__(self):
        super().__init__('model_server_node')
        print("GPU: ", tf.test.is_gpu_available)
        with tf.device("gpu:0"):
            self.model = tf.keras.models.load_model("/home/krg6/Capstone/PilotNet_Train/pilotnet/models/PilotNet_v23.h5")
            print("Model loaded!")

        # ROS 2 subscribers / publishers
        self.image_sub = self.create_subscription(Image, '/carla/camera/image_cropped', self.image_callback, 10)
        self.steering_pub = self.create_publisher(Float32, '/carla/steering_cmd', 10)

    def image_callback(self, msg):
        # Convert ROS Image -> numpy array
        print("Seq: ", msg.header.frame_id, " | Time: ", time.time())
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))

        # Resize to model input
        img_resized = tf.image.resize(img, [66, 200])
        img_input = np.expand_dims(img_resized, axis=0)

        # Predict steering
        steering = float(self.model.predict(img_input)[0].item())

        # Publish steering command
        self.steering_pub.publish(Float32(data=steering))

def main(args=None):
    rclpy.init(args=args)
    node = ModelServerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
