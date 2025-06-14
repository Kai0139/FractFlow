from http import HTTPStatus
from pathlib import PurePosixPath
from urllib.parse import urlparse, unquote
import requests

from tkinter import Image
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from typing import List
import os
from dotenv import load_dotenv
from dashscope import ImageSynthesis

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("qwen_image")


def normalize_path(path: str) -> str:
    """
    Normalize a file path by expanding ~ to user's home directory
    and resolving relative paths.
    
    Args:
        path: The input path to normalize
        
    Returns:
        The normalized absolute path
    """
    # Expand ~ to user's home directory
    expanded_path = os.path.expanduser(path)
    
    # Convert to absolute path if relative
    if not os.path.isabs(expanded_path):
        expanded_path = os.path.abspath(expanded_path)
        
    return expanded_path


@mcp.tool()
async def create_image_with_gpt(
    save_path: str,
    prompt: str,
) -> str:
    """
    Generate a new image from scratch using GPT's image generation capabilities.
    This tool creates images based solely on a text prompt, without requiring any reference images.
    
    Args:
        save_path: Full path where the generated image will be saved (including filename)
        prompt: Detailed text description of the image to generate
        
    Returns:
        Image file path as a string where the generated image is saved
        
    Example:
        To generate a children's book style illustration:
        ```python
        result = await create_image_with_gpt(
            save_path="output/otter.png",
            prompt="A children's book drawing of a veterinarian using a stethoscope to listen to the heartbeat of a baby otter."
        )
        ```
    """
    # Normalize the save path
    save_path = normalize_path(save_path)
    
    # Ensure the save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print("save dir: {}".format(save_dir))
    try:
        # Generate image using GPT
        rsp = ImageSynthesis.call(
            api_key=os.getenv('QWEN_API_KEY'),
            model="wanx2.1-t2i-turbo",
            prompt=prompt,
            n=1,
            size="1024*1024"
        )
        
        if rsp.status_code == HTTPStatus.OK:
            for result in rsp.output.results:
                with open(save_path, "wb") as f:
                    f.write(requests.get(result.url).content)
        else:
            print('sync_call Failed, status_code: %s, code: %s, message: %s' %
                (rsp.status_code, rsp.code, rsp.message))
        return save_path
    except Exception as e:
        raise Exception(f"Failed to generate image: {str(e)}")
    finally:
        # Clean up any resources if needed
        pass

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio') 