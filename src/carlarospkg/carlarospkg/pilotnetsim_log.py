import carla
import numpy as np
import tensorflow as tf
import cv2
import time

import os
from carla import WeatherParameters
import math
import csv
import queue

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#tf.debugging.set_log_device_placement(True)
# from ctypes import *
# lib8 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcublas.so.11')
# lib1 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcudart.so.11.0')
# lib2 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcublasLt.so.11')
# lib3 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcufft.so.10')
# lib4 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcurand.so.10')
# lib5 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcusolver.so')
# lib6 = cdll.LoadLibrary('/home/krg/miniconda3/envs/carla9.13_py3.7/lib/libcusparse.so.11')
#lib7 = cdll.LoadLibrary('/usr/lib/x86_64-linux-gnu/libcudnn.so')

# import tensorrt as tr

# from tensorflow.python.compiler.tensorrt import trt_convert as trt
# print("Shou!")
# params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode="FP32")
# print("YYDS")
# converter = trt.TrtGraphConverterV2(input_saved_model_dir="./pilotnetopt", conversion_params=params)
# converter.convert()
# converter.save("./pilotnetopt_onxx")
# print("Saved")

#print("Is GPU available:", tf.config.list_physical_devices('GPU'))

i=0

def flip_bits_in_layer_kernel(layer, n_flips=10, bit_low=0, bit_high=5):
    """
    Flip n_flips random bits in the 'kernel' of a Keras layer (Conv2D / Dense).
    bit_low, bit_high: inclusive range of bit positions (mantissa bits).
    """
    var = layer.kernel  # tf.Variable
    w = var.read_value()  # tf.Tensor float32

    assert w.dtype == tf.float32

    int_view = tf.bitcast(w, tf.int32)
    flat = tf.reshape(int_view, [-1])
    size = tf.size(flat)

    for _ in range(n_flips):
        # random index in this weight tensor
        idx = tf.random.uniform([], 0, size, dtype=tf.int32)
        # random bit within [bit_low, bit_high]
        bit = tf.random.uniform([], bit_low, bit_high - 1, dtype=tf.int32)

        val = flat[idx]
        one = tf.constant(1, dtype=tf.int32)
        mask = tf.bitwise.left_shift(one, bit)
        val_flipped = tf.bitwise.bitwise_xor(val, mask)

        flat = tf.tensor_scatter_nd_update(flat, [[idx]], [val_flipped])

    # reshape and cast back
    int_flipped = tf.reshape(flat, tf.shape(int_view))
    w_flipped = tf.bitcast(int_flipped, tf.float32)
    var.assign(w_flipped)

    print(f"Flipped {n_flips} bits in layer {layer.name}, bits [{bit_low}, {bit_high}]")



def main():
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.2  # 100 Hz
        world.apply_settings(settings)

        weather = WeatherParameters(
        sun_altitude_angle=90.0,  # High noon = minimal shadows
        sun_azimuth_angle=0.0,
        cloudiness=0.0,           # Clear sky (no cloud shadows)
        precipitation=0.0,
        fog_density=0.0,
        wetness=0.0
        )
        world.set_weather(weather)
        #client.reload_world()
        blueprint_library = world.get_blueprint_library()

        # Spawn the vehicle
        #bp = blueprint_library.filter("model3")[0]
        bp = blueprint_library.find('vehicle.dodge.charger_police') 
        spawn_point = world.get_map().get_spawn_points()[63]                 # 144 -> RT-LL; 7 -> LT-RL; 6 -> LT-LL
        #spawn_point = carla.Transform(carla.Location(x=65.77, y=137.46, z=0.60), carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        #spawn_point = carla.Transform(carla.Location(x=64.06, y=-57.72, z=0.60))
        print("New spawn point: ", spawn_point)
        vehicle = world.spawn_actor(bp, spawn_point)
        # Set spectator to follow the vehicle
        spectator = world.get_spectator()
        # Attach a front-facing camera
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "720")
        camera_bp.set_attribute("image_size_y", "405")
        camera_bp.set_attribute("fov", "110")
        camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)


        # Load trained PilotNet model
        #model = tf.keras.models.load_model("./pilotnetopt")
        print("GPU: ", tf.test.is_gpu_available)
        
        # Create CSV Log file
        log_dir = "/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "SS_Total_and_Pred_log.csv")

        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.DictWriter(csv_f, fieldnames=["iter", "elapsed_time", "pred_time"])
        csv_w.writeheader()
        csv_f.flush()
        print("Logging elapsedtime to:", csv_path)
        iter_count = 0

        # Load Model
        with tf.device("gpu:0"):
            #model = tf.saved_model.load("./pilotnetopt")
            model = tf.keras.models.load_model("models/PilotNet_v23.h5")   
            model.summary()
            for i, layer in enumerate(model.layers):
                print(i, layer.name, type(layer))

            # first_conv = None
            # for layer in model.layers:
            #     if isinstance(layer, tf.keras.layers.Conv2D):
            #         first_conv = layer
            #         break

            # if first_conv is not None:
            #     flip_bits_in_layer_kernel(first_conv, n_flips=10, bit_low=30, bit_high=32)
            #     print("Injected 1 low-bit flip in layer:", first_conv.name)
            # else:
            #     print("WARNING: No Conv2D layer found for bit flip!") 

            # len_layers = len(model.layers)
            # print("Depth of layers: ", len_layers)

            # conv_layer = None
            # layer_chosen = 1
            # conv_layer = model.layers[layer_chosen]

            # if conv_layer is not None:
            #     flip_bits_in_layer_kernel(conv_layer, n_flips=0, bit_low=30, bit_high=32)
            #     print("Injected 1 low-bit flip in layer:", conv_layer.name)
            # else:
            #     print("WARNING: Invalid layer")
        
        print("Done loading those balls......")

        

        def preprocess(image):

            img = np.frombuffer(image.raw_data, dtype=np.uint8)
            #print("Shape-1: ",img.shape)
            
            img = img.reshape((image.height, image.width, 4))[:, :, :3]
            #print("Shape-2", img.shape)

            h, w, _ = img.shape
            top = int(0.5 * h)
            bottom = int(0.8 * h)
            left = int(0.3*w)
            right = int(0.7*w)
            img = img[top:bottom, left:right, :] 
            img = cv2.resize(img, (200, 66))    
            #print("Shape-3", img.shape)
            
            #img = img.astype(np.float32) / 255.0 #- 0.5
            return np.expand_dims(img, axis=0), img

        def drive(image):
            global i
            img, img_display = preprocess(image)
            #print("Output Values: ", model.predict(img))
            #print("Output Shape: ", model.predict(img)[0].shape)
            #print(model.predict(img).shape)
            
            pred_start = time.time()
            steering = float(model.predict(img)[0].item())
            prediction_time = time.time() - pred_start

            # csv_w.writerow({"pred_time": prediction_time})

            #steering = (steering * 2.0) - 1.0
            # throttling = float(model.predict(img)[1].item())
            # braking = float(model.predict(img)[2].item())
            #braking = 0 

            # print("Steering: ", steering)
            # print("Throttling: ", throttling)
            # print("Braking: ", braking)
            
            control = carla.VehicleControl()
            control.steer = steering    
            # control.brake = braking
            # control.throttle = throttling
            # control.steer = np.clip(steering, -1.0, 1.0)
            
            # if braking > 0.1:   
            #     control.throttle = 0.0
            #     #control.brake = np.clip(braking, 0.0, 1.0)
            # elif steering < 0.04 and steering > -0.04:
            #     control.steer = 0
            #     #control.throttle = np.clip(throttling, 0.0, 1.0)
            #     control.brake = 0.0
            # else:
            #     control.brake = 0.0

            if steering < 0.04 and steering > -0.04:
                control.steer = 0
            
            control.throttle = 0.4
            
            #print(control.steer, control.throttle, control.brake, sep = '\n')
            print("Steering: ", control.steer, "Throttle: ", control.throttle)
            vehicle.apply_control(control)

            i=i+1
            #cv2.imwrite(f"sim_images/{i}_{steering}.png",img_display*255)
            #cv2.imshow("Image Seen: ", img_display)
            #cv2.waitKey(1)

            return prediction_time
        
        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        
        world.tick()
        oldtime=time.time()
        #camera.listen(lambda image: drive(image))

        try:
            while True:

                image = image_queue.get()
                pred_time = drive(image)
                #transform = vehicle.get_transform()
                # spectator.set_transform(carla.Transform(
                #     transform.location + carla.Location(x=-10, z=2),
                #     carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
                # ))
                # try:
                #     image = image_queue.get()
                # except queue.Empty:
                #     print("WARNING: Did not receive camera image in time.")
                #     continue
                # # inference & control here
                # drive(image)

                v = vehicle.get_velocity()
                info_text = [
                    '',
                    'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
                    '']
                print(info_text)
                world.tick()
                elapsedtime=time.time()-oldtime
                print("Time elapsed: ", elapsedtime)
                oldtime=time.time()
                iter_count += 1
                csv_w.writerow({"iter":iter_count, "elapsed_time":elapsedtime, "pred_time":pred_time})
                if iter_count % 50 == 0:
                    csv_f.flush()
                time.sleep(0.1)
        except KeyboardInterrupt:
            camera.stop()
            vehicle.destroy()
            print("Simulation ended. Vehicle and camera destroyed.")
            settings = world.get_settings()
            settings.synchronous_mode = False # Disables synchronous mode
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
    except Exception as e:
            camera.stop()
            vehicle.destroy()
            print("Simulation ended. Vehicle and camera destroyed.")
            settings = world.get_settings()
            settings.synchronous_mode = False # Disables synchronous mode
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            print("Exception:", e)

if __name__ == '__main__':
    main()
        
