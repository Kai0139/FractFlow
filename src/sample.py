import sys
from pathlib import Path

proj_root = Path(__file__).resolve().parent.parent
sys.path.append(str(proj_root))

from FractFlow.tool_template import ToolTemplate

class HelloTool(ToolTemplate):
    """Simple greeting tool"""
    
    SYSTEM_PROMPT = """
You are a friendly greeting assistant.
When users provide names, please give personalized greetings.
Please reply in Chinese, maintaining a friendly and enthusiastic tone.
"""
    
    TOOL_DESCRIPTION = """
Tool for generating personalized greetings.

Parameters:
    query: str - User's name or greeting request

Returns:
    str - Personalized greeting message
"""
    @classmethod
    def create_config(cls):
        """Custom configuration for File I/O agent"""
        from FractFlow.infra.config import ConfigManager
        from dotenv import load_dotenv
        
        load_dotenv()
        return ConfigManager(
            provider='deepseek',
            deepseek_model='deepseek-chat',
            max_iterations=20,  # Higher iterations for complex file operations
            custom_system_prompt=cls.SYSTEM_PROMPT,
            tool_calling_version='stable'
        )

if __name__ == "__main__":
    HelloTool.main()