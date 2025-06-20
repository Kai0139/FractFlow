from typing import List, Dict, Optional, Any
import os
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from dotenv import load_dotenv
import rclpy.duration
import rclpy.time
load_dotenv()
# Initialize FastMCP server
mcp = FastMCP("Visual_Question_Answering")

from PIL import Image
import base64
import io
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import Point
import ast

class ROSImageComms(Node):
    def __init__(self):
        super().__init__("ros_image_comms")
        self.cv_bridge = CvBridge()

        self.image = None

        self.cam_sub = self.create_subscription(
            ROSImage, "/camera/image_raw", self.ros_image_cb, 10
        )

    def ros_image_cb(self, msg):
        """Convert a ROS Image message to a numpy array.
        
        Args:
            msg: ROS Image message to convert
            
        Returns:
            A numpy array representing the image
        """
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg)
        self.image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

class ROSPosComms(Node):
    def __init__(self):
        super().__init__("ros_pos_comms")
        self.cv_bridge = CvBridge()

        self.px = None
        self.py = None

        self.cam_sub = self.create_subscription(
            Point, "/current_pos", self.ros_image_cb, 10
        )

    def ros_image_cb(self, msg):
        self.px = msg.x
        self.py = msg.y


class ROSGotoPos(Node):
    def __init__(self):
        super().__init__("ros_goto_pos")
        self.target_pos = None
        self.target_pos_pub = self.create_publisher(Point, "/goto_pos", 10)
        self.target_pos_sub = self.create_subscription(
            Point, "/current_pos", self.ros_pos_cb, 10
        )

    def ros_pos_cb(self, msg):
        self.target_pos = (msg.x, msg.y)

    def publish_target(self, x, y):
        msg = Point()
        msg.x = x
        msg.y = y
        self.target_pos_pub.publish(msg)
        time.sleep(0.1)

def encode_image(image: Image.Image, size: tuple[int, int] = (512, 512)) -> str:
    image.thumbnail(size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image


def ros_image_cb(msg) -> np.ndarray:
    """Convert a ROS Image message to a numpy array.
    
    Args:
        msg: ROS Image message to convert
        
    Returns:
        A numpy array representing the image
    """
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return cv_image

def load_image_cv(size_limit: tuple[int, int] = (512, 512)) -> tuple[str, dict]:
    """Load an image from a numpy array.
    
    Args:
        image: numpy array of the image
        
    Returns:
        A tuple containing:
            - Base64 encoded string of the resized image (max 512x512) (this can be put in the image_url field of the user message)
            - Dictionary with metadata including original width and height
    """
    rclpy.init()
    node = ROSImageComms()
    while rclpy.ok() and node.image is None:
        rclpy.spin_once(node)
    
    meta_info = {}
    image = Image.fromarray(node.image)
    meta_info['width'], meta_info['height'] = image.size
    base64_image = encode_image(image, size_limit)

    node.destroy_node()
    rclpy.shutdown()
    return base64_image, meta_info

@mcp.tool()
async def Visual_Question_Answering(prompt: str) -> str:
# def Visual_Question_Answering(prompt: str) -> str:
    '''
    This tool uses Qwen-VL-Plus model to perform visual question answering or image analysis.
    The image is automatically resized to a maximum of 480x480 pixels before processing.
    
    Args:
        prompt (str): Text prompt describing what you want to know about the image. This can be:
                     - A direct question about the image content (e.g., "What objects are in this image?")
                     - A request for detailed description (e.g., "Describe this image in detail")
                     - A specific analytical instruction (e.g., "The location of the chair in image")
    
    Returns:
        str: A detailed text response from the VLM model analyzing the image according to the prompt.
             The response format depends on the nature of the prompt.
    '''
    base64_image, meta_info = load_image_cv((480, 480))
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[{"role": "user","content": [
                {"type": "text","text": prompt},
                {"type": "image_url",
                "image_url": {"url": f'data:image/png;base64,{base64_image}'}}
                ]}]
    )
    return completion.choices[0].message.content

@mcp.tool()
async def Move_to_Position(x, y):
# def Move_to_Position(position: str):
    '''
    This tool moves the robot to a specified position.
    
    Args:
        x: x coordinate of the target position
        y: y coordinate of the target position
    '''
    rclpy.init()
    pos_pub = ROSGotoPos()
    time.sleep(0.1)
    pos_pub.target_pos = None
    pos_pub.publish_target(float(x), float(y))
    while rclpy.ok() and pos_pub.target_pos is None:
        rclpy.spin_once(pos_pub, timeout_sec=3)
    # pos_pub.destroy_node()
    rclpy.shutdown()
    return True

@mcp.tool()
async def Get_Current_Position():
    rclpy.init()
    pos_comms = ROSPosComms()
    while rclpy.ok() and pos_comms.px is None:
        rclpy.spin_once(pos_comms)
    rclpy.shutdown()
    return str([round(pos_comms.px, 3), round(pos_comms.py, 3)])
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio') 
    # res = Visual_Question_Answering(prompt="What is the localtion of the white single bed in image? " \
    # "Tell me the exact bounding box in form (cx, cy, w, h), do not return anything else")
    # Move_to_Position(position="[0.5, 0.5]")
    # print(res)


