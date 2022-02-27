import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import ros2_numpy as rnp
import numpy as np
import cv2

import os
from ament_index_python.packages import get_package_share_directory

from .DDRNet_23_slim import get_seg_model
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DDRNetInferenceNode(Node):

    def __init__(self, weights_file):
        super().__init__('ddrnet_inference_node')

        self.model = self.load_model(weights_file)
        self.get_logger().info('Successfully loaded model')

        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]   
        
        self.color_image_sub = self.create_subscription(
            Image, 
            '/zed2/zed_node/rgb_raw/image_raw_color',
            self.rgb_image_callback,
            10
        )
    
    def load_model(self, weights_file):
        model = get_seg_model(cfg=None)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights_file, map_location=device)
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.to(device)
        model.eval()
        return model


    def decode_segmap(self, image, nc=19):
        label_colors = np.array([(128, 0, 128),  # 0=road
            # 1=sidewalk, 2=building, 3=wall, 4=fence, 5=poles
            (192, 0, 192), (64, 64, 64), (128, 64, 0), (128, 128, 0), (128, 128, 128),
            # 6=traffic light, 7=traffic sign, 8=vegetation, 9=ground, 10=sky
            (128, 128, 0), (192, 192, 0), (0, 128, 0), (0, 192, 0), (0, 192, 255),
            # 11=person, 12=rider, 13=car, 14=truck, 15=bus
            (192, 0, 0), (255, 0, 0), (0, 0, 192), (0, 0, 128), (0, 0, 64),
            # 16=on rails, 17=motorcycle, 18=bicycle
            (0, 64, 64), (128, 64, 0), (128, 0, 0)])
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

        
    def rgb_image_callback(self, msg: Image):
        image = rnp.numpify(msg)
        image = image[..., :3]
        cv2.imwrite('original.jpg', image)
        image = cv2.resize(image, (2048, 1024))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(device)

        t1 = time.time()
        seg_map = self.model(tensor)
        t2 = time.time()
        print(1 / (t2 - t1))
        seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()
        seg_image = self.decode_segmap(seg_map)
        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite('prediction.jpg', seg_image)


def main(args=None):
    rclpy.init(args=args)

    share_dir = os.path.dirname(get_package_share_directory('ddrnet_inference'))
    weights_file = os.path.join(share_dir, 'weights/best_val_smaller.pth')

    inference_node = DDRNetInferenceNode(weights_file)

    rclpy.spin(inference_node)

    inference_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()