"""
Visual Question Answering Tool - Unified Interface

This module provides a unified interface for visual question answering that can run in multiple modes:
1. MCP Server mode (default): Provides AI-enhanced VQA operations as MCP tools
2. Interactive mode: Runs as an interactive agent with VQA capabilities
3. Single query mode: Processes a single query and exits

Usage:
  python vqa_tool.py                        # MCP Server mode (default)
  python vqa_tool.py --interactive          # Interactive mode
  python vqa_tool.py --query "..."          # Single query mode
"""

import os
import sys
from pathlib import Path
# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Import the FractFlow ToolTemplate
from FractFlow.tool_template import ToolTemplate

class VQATool(ToolTemplate):
    """Visual Question Answering tool using ToolTemplate"""
    
    SYSTEM_PROMPT = """
You are a robot operator, I will ask you to control the robot to move to the position of an object.

# Robot Working Space Description
The robot working space is a 10m by 10m square, the robot starts at the center of the working space, center coordinate is (0m, 0m).
The camera is mounted at the top of the robot working space, the camera looks downward, which means the image is the top view of the working environment.
The camera pixel coordinate can be converted to working space coordinate by the following formula:
x = (pixel_x - 480 / 2) / 480 * 10
y = (480 / 2 - pixel_y) / 480 * 10

# Example
User: move to the black office chair. 
You should use Visual_Question_Answering to find the position of the black office chair by 
Visual_Question_Answering(prompt="What is the localtion of the black office chair in image? 
Tell me the exact bounding box in form (cx, cy, w, h), do not return anything else"). 
 Then, you convert the pixel coordinate of the object to the coordinate of the object in the working space. 
For instance, if the returned bounding box is (360, 240, 100, 100), then the center of the object in working space is (2.5m, 0m),
 you should then call Move_to_Position(x=2.5, y=0) to move the robot to the position of the object.

# Available tool 1
Visual_Question_Answering - This tool will answer the question about image in the camera, you don't need to provide the image.

Visual_Question_Answering Tool usage
1. When I ask you to move to the position of an object, you need to use the Visual_Question_Answering tool to find the position of the object.
2. If the answer contains the position of the object, you can use the motion control tools to move the robot to the position of the object.
3. If the answer does not contain the position of the object, you can ask the user to provide the position of the object or change 
the description of the object.

# Available tool 2
Move_to_Position - This tool will control the robot to move to the target position, you need to provide the position in form [x, y].

Move_to_Position Tool usage
1. After you get the position of the object using Visual_Question_Answering tool, 
you can use the Motion_Control tool to move the robot to the position of the object.
2. Be aware that you need to convert the position from the image coordinate to the working space coordinate.
"""
    
    TOOLS = [
        ("tools/core/g1_ros2/vqa_mcp.py", "visual_question_answering_operations")
    ]
    
    MCP_SERVER_NAME = "robot_commander_tool"
    
    TOOL_DESCRIPTION = """Visual_Question_Answering:
    Answers questions about visual content, the image will be automatically extracted by the tool 
    from a camera mounted at the top of the robot working space, the camera looks downward.
    
    Parameters:
        query: str - specific question (e.g., "Where is the chair in image? Return the bounding box in form (cx, cy, w, h)")
        
    Returns:
        str - Visual analysis result or error message

    Move_to_Position:
    This tool moves the robot to a specified position.
    
    Parameters:
        position (str): "[x, y]" Text of a list containing the x and y coordinates of the position.
    Returns:
        bool - True if the robot moves to the position successfully, False otherwise.
    """
    
    @classmethod
    def create_config(cls):
        """Custom configuration for VQA tool"""
        from FractFlow.infra.config import ConfigManager
        from dotenv import load_dotenv
        
        load_dotenv()
        return ConfigManager(
            provider='deepseek',
            deepseek_model='deepseek-chat',
            max_iterations=2,  # Visual analysis usually completes in one iteration
            custom_system_prompt=cls.SYSTEM_PROMPT,
            tool_calling_version='turbo'
        )

if __name__ == "__main__":
    VQATool.main() 