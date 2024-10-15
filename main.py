import os
import sys
import asyncio
from typing import List, Dict, Optional
import json
import re
import subprocess
import tempfile
import time
import threading
import webbrowser
import shutil

import PyPDF2
import openai
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
from rich.logging import RichHandler
import logging
from aiolimiter import AsyncLimiter
from tqdm import tqdm
import torch
import psutil
import pynvml
from asyncio import Queue

# Initialize Rich Console
console = Console()

# Configure Rich Logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs; change to INFO in production
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger("rich")

# -----------------------------
# PDFExtractor Class
# -----------------------------
class PDFExtractor:
    """
    Extracts text from a PDF file.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.text = ""

    def extract_text_sync(self) -> str:
        """
        Synchronously extracts text from the specified PDF file.

        Returns:
            str: The extracted text with page separators.
        """
        if not os.path.exists(self.pdf_path):
            logger.error(f"Error: The file '{self.pdf_path}' does not exist.")
            sys.exit(1)

        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        self.text += f"--- Page {page_num} ---\n{page_text}\n"
                    else:
                        self.text += f"--- Page {page_num} ---\n[No text found on this page]\n"
            return self.text
        except Exception as e:
            logger.error(f"An error occurred while reading the PDF: {e}")
            sys.exit(1)

    async def extract_text(self) -> str:
        """
        Asynchronously extracts text from the PDF by running the synchronous extraction in a separate thread.

        Returns:
            str: The extracted text with page separators.
        """
        return await asyncio.to_thread(self.extract_text_sync)


# -----------------------------
# ModelLayer Class using OpenAI GPT-4o
# -----------------------------
class ModelLayer:
    """
    Represents a single model layer that interacts with the OpenAI GPT-4o API asynchronously.
    """

    def __init__(
        self,
        prompt_template: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 8000,
        temperature: float = 0.7,
        global_limiter: AsyncLimiter = AsyncLimiter(100, 60),  # 100 requests per 60 seconds
    ):
        """
        Initializes the ModelLayer.

        Args:
            prompt_template (str): The template for the prompt with placeholders.
            model (str): The OpenAI model to use.
            max_tokens (int): The maximum number of tokens for the API response.
            temperature (float): Sampling temperature for the model.
            global_limiter (AsyncLimiter): Global rate limiter to control API request rate based on tokens.
        """
        self.prompt_template = prompt_template
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.outputs: List[str] = []
        self.global_limiter = global_limiter

    def format_prompt(self, variables: Dict[str, str]) -> str:
        """
        Formats the prompt template with the provided variables.

        Args:
            variables (Dict[str, str]): A dictionary containing variables to replace in the prompt.

        Returns:
            str: The formatted prompt.
        """
        return self.prompt_template.format(**variables)

    async def query_openai(self, prompt: str) -> Optional[str]:
        """
        Asynchronously sends a request to the OpenAI API with the given prompt.

        Args:
            prompt (str): The formatted prompt to send to the API.

        Returns:
            Optional[str]: The API response text or None if an error occurs.
        """
        async with self.global_limiter:
            try:
                logger.debug(f"Sending prompt to OpenAI API: {truncate_text(prompt)}")
                response = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                response_content = response.choices[0].message.content.strip()
                logger.debug(f"Received response from OpenAI API: {truncate_text(response_content)}")
                return response_content
            except Exception as e:
                logger.error(f"An error occurred during OpenAI API request: {e}")
                return None

    async def process_inputs(self, prompts: List[str]):
        """
        Processes a list of prompts asynchronously with retries.

        Args:
            prompts (List[str]): The list of formatted prompts to process.
        """
        tasks = [self._retry_query(prompt) for prompt in prompts]
        self.outputs = await asyncio.gather(*tasks)

    async def _retry_query(self, prompt: str, retries: int = 5, backoff_factor: float = 1.5) -> Optional[str]:
        """
        Attempts to query the OpenAI API with exponential backoff on failure.

        Args:
            prompt (str): The formatted prompt to send to the API.
            retries (int): Number of retry attempts.
            backoff_factor (float): Factor by which the wait time increases after each retry.

        Returns:
            Optional[str]: The API response text or None if all retries fail.
        """
        wait_time = 2  # Initial wait time in seconds
        for attempt in range(1, retries + 1):
            response = await self.query_openai(prompt)
            if response:
                logger.debug(f"Model response: {response}")  # Log the raw response
                return response
            else:
                if attempt < retries:
                    logger.warning(f"Retrying request ({attempt}/{retries}) in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    wait_time *= backoff_factor
                else:
                    logger.error(f"All {retries} retry attempts failed for prompt: {truncate_text(prompt)}")
        return None

    def get_outputs(self) -> List[str]:
        """
        Retrieves the outputs after processing.

        Returns:
            List[str]: The list of output texts.
        """
        return self.outputs


# -----------------------------
# ModelPipeline Class
# -----------------------------
class ModelPipeline:
    """
    Manages multiple ModelLayer instances, ensuring data flows sequentially through each layer.
    """

    def __init__(self):
        self.layers: List[ModelLayer] = []

    def add_layer(self, model_layer: ModelLayer):
        """
        Adds a ModelLayer to the pipeline.

        Args:
            model_layer (ModelLayer): The ModelLayer instance to add.
        """
        self.layers.append(model_layer)

    async def run(self, initial_inputs: List[str], transform=None) -> List[str]:
        """
        Runs the pipeline with the initial inputs.

        Args:
            initial_inputs (List[str]): The initial input texts.
            transform (Callable[[List[str]], Any], optional): A function to transform the outputs before passing to the next layer.

        Returns:
            List[str]: The final outputs after all layers have processed the data.
        """
        current_inputs = initial_inputs
        for idx, layer in enumerate(self.layers, start=1):
            layer_type = "Summarization" if idx == 1 else "Manim Code Generation"
            logger.info(f"--- Processing Layer {idx}: {layer_type} ---")
            if transform:
                current_inputs = transform(current_inputs)
            await layer.process_inputs(current_inputs)
            current_outputs = layer.get_outputs()
            logger.info(f"--- Completed Layer {idx} ---\n")
            current_inputs = current_outputs
        return current_inputs


# -----------------------------
# Utility Functions
# -----------------------------
def split_text(text: str, max_length: int) -> List[str]:
    """
    Splits the text into chunks not exceeding max_length characters.

    Args:
        text (str): The text to split.
        max_length (int): The maximum length of each chunk.

    Returns:
        List[str]: The list of text chunks.
    """
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def parse_summary(summary: str) -> Dict[str, str]:
    """
    Parses the summary to extract the title and content.

    Args:
        summary (str): The summary text.

    Returns:
        Dict[str, str]: A dictionary with 'title' and 'content'.
    """
    logger.debug(f"Raw summary: {summary}")  # Debug log for raw summary

    # Attempt to find the JSON part of the response
    json_start = summary.find("{")
    json_end = summary.rfind("}") + 1

    if json_start == -1 or json_end == -1:
        logger.error("JSON braces not found in the summary. Using raw response as content.")
        return {"title": "Untitled", "content": summary.strip()}

    json_str = summary[json_start:json_end]
    logger.debug(f"Extracted JSON string: {json_str}")  # Log extracted JSON

    try:
        summary_dict = json.loads(json_str)
        title = summary_dict.get("title", "Untitled")
        content = summary_dict.get("content", "")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e}. Using raw response as content.")
        return {"title": "Untitled", "content": summary.strip()}

    if not title or not content:
        logger.warning("Summary format incorrect or missing fields. Using raw response as content.")
        return {"title": title or "Untitled", "content": content or summary.strip()}

    return {"title": title, "content": content}


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncates the text to the specified maximum length.

    Args:
        text (str): The text to truncate.
        max_length (int): The maximum length of the truncated text.

    Returns:
        str: The truncated text with an ellipsis if it was longer than max_length.
    """
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def monitor_gpu_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    while True:
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        logger.info(f"GPU Utilization: {utilization.gpu}%")
        time.sleep(5)  # Update every 5 seconds


async def run_manim_async(scene_names, python_path, output_dir):
    """
    Asynchronously renders Manim scenes.

    Args:
        scene_names (List[str]): List of scene class names to render.
        python_path (str): Path to the Python executable.
        output_dir (str): Directory where the final video will be saved.

    Returns:
        List[str]: Successfully rendered scene names.
    """
    successful_scenes = []

    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA is {'available' if cuda_available else 'not available'}")
    if cuda_available:
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    os.environ['CUDA_DEVICE'] = '0'

    cpu_count = psutil.cpu_count(logical=False)
    max_workers = max(1, int(cpu_count * 0.7))
    logger.info(f"Using {max_workers} workers for rendering")

    async def render_scene(queue):
        while True:
            scene = await queue.get()
            try:
                for attempt in range(3):  # Try up to 3 times
                    try:
                        command = (
                            f"{python_path} -m manim -qm --format=mp4 --media_dir ./manim_output/scenes "
                            f"--renderer=opengl --resolution=1280,720 generated_manim_scenes.py {scene}"
                        )
                        logger.debug(f"Executing command: {command}")
                        process = await asyncio.create_subprocess_shell(
                            command,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'MANIM_USE_OPENGL_RENDERER': '1'}
                        )
                        stdout, stderr = await process.communicate()
                        if process.returncode == 0:
                            logger.info(f"Manim command completed successfully for {scene}")
                            successful_scenes.append(scene)
                            break
                        else:
                            stderr_decoded = stderr.decode()
                            if "PermissionError: [WinError 32]" in stderr_decoded:
                                logger.warning(f"Permission error for {scene}. Retrying... (Attempt {attempt + 1})")
                                await asyncio.sleep(0.1)  # Wait for 0.1 seconds before retrying
                            else:
                                logger.error(f"Error running Manim for scene {scene}: {stderr_decoded}")
                                break
                    except Exception as e:
                        logger.error(f"Exception occurred while rendering {scene}: {str(e)}")
                        break
                else:
                    logger.error(f"Failed to render {scene} after 3 attempts")
            finally:
                queue.task_done()

    queue = Queue()
    for scene in scene_names:
        await queue.put(scene)

    workers = [asyncio.create_task(render_scene(queue)) for _ in range(max_workers)]

    with tqdm(total=len(scene_names), desc="Generating Manim scenes", unit="scene") as pbar:
        while not queue.empty():
            await asyncio.sleep(0.1)
            completed = len(scene_names) - queue.qsize()
            pbar.n = completed
            pbar.refresh()

    await queue.join()

    for worker in workers:
        worker.cancel()

    await asyncio.gather(*workers, return_exceptions=True)

    return successful_scenes


def cleanup_temp_files():
    temp_dir = "./temp_videos/Tex"
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Error deleting file {file_path}: {e}")


# -----------------------------
# Main Application
# -----------------------------
async def main():
    # Define python_path
    python_path = sys.executable

    # Check Manim installation
    try:
        subprocess.run([python_path, "-m", "manim", "--version"], check=True, capture_output=True, text=True)
        logger.info("Manim is installed correctly")
    except subprocess.CalledProcessError as e:
        logger.error(f"Manim is not installed or not working correctly: {e}")
        sys.exit(1)

    # -----------------------------
    # Configuration
    # -----------------------------
    # Path to the PDF file
    pdf_path = r"C:\Users\Ricardo\Downloads\sciadv.adm8470.pdf"

    # Maximum chunk size based on API's max input size (in characters)
    max_chunk_size = 8000  # Adjust this value as needed

    # Full script content where the generated Manim code snippets will be inserted
    full_script_content = """
from manim import *

# Existing Scene classes and functions

# Insert generated Scene classes below
"""

    # Layer Definitions
    layer_definitions = [
        # Layer 1: Summarization
        {
            "prompt_template": """
For the following text, extract the title and provide a concise summary.

Text:
{text}

Output Format:
Respond **ONLY** with a single JSON object containing 'title' and 'content' fields. **Do not include any additional text or formatting.**

Example:
{{
    "title": "Understanding Neural Networks",
    "content": "This section delves into the fundamentals of neural networks, exploring their architecture, training processes, and applications in various fields such as image recognition and natural language processing."
}}
""",
            "model": "gpt-4o-mini",
            "max_tokens": 2000,  # Reduced for summarization
            "temperature": 0.0  # Deterministic output
        },
        # Layer 2: Manim Code Generation
        {
            "prompt_template": """
Create a Manim Python code snippet to visualize the following content.

Title: {title}
Content: {content}

Your code will be inserted into the following script structure:

{full_script_content}

Follow these specific instructions:
1. **Manim Version Compatibility:** Use Manim v0.17.2 compatible syntax.
2. **Scene Structure:** The code should define a single Scene class named {scene_class_name}Scene.
3. **Imports:** Do not include from manim import * as it's already present in the full script.
4. **Visual Complexity:**
    - Incorporate advanced Manim features such as custom animations, complex mathematical objects, or interactive elements.
    - Use multiple layers of animations to depict different aspects of the content simultaneously.
    - Integrate transitions that seamlessly move between different visual representations.
5. **Content Integration:**
    - Ensure that each visual element directly relates to and enhances the understanding of the content.
    - Avoid generic or filler animations; focus on conveying key concepts and insights.
6. **Optimization:**
    - Optimize for GPU usage by leveraging vector operations, minimizing object creation/destruction, and grouping similar operations.
    - Ensure that the scene is efficient without compromising visual quality.
7. **Structure and Readability:**
    - Organize the code into well-structured methods within the Scene class for better readability and maintenance.
    - Include comprehensive comments explaining the purpose and functionality of each major code block.
8. **Performance:**
    - Ensure that the scene duration is between 10 to 30 seconds.
    - Optimize animations to prevent unnecessary rendering overhead.
9. **Error Handling:**
    - Include necessary error handling to manage potential runtime issues during scene rendering.
11. Optimize for GPU usage:
    - Use vector operations instead of loops where possible.
    - Minimize creation and destruction of objects.
    - Use GPU-accelerated functions when available (e.g., from mobject.py).
    - Group similar operations together to minimize state changes.

Important:
- DO NOT include any markdown formatting, code block indicators (
), or regex patterns in your response.
- Provide ONLY the Python code for the scene class, without any additional explanation or formatting.
- Begin the code with 'class {scene_class_name}Scene(Scene):' and end with the last line of the scene class.
- Do not include any text before or after the class definition.
""",
            "model": "gpt-4o-mini",
            "max_tokens": 16000,
            "temperature": 0.7
        },
    ]

    # Create directories for storing scenes and output videos
    scenes_dir = "./manim_output/scenes"
    os.makedirs(scenes_dir, exist_ok=True)
    output_dir = "./manim_output"
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Step 1: Extract Text from PDF
    # -----------------------------
    extractor = PDFExtractor(pdf_path)
    logger.info("📄 Starting PDF text extraction...")
    extracted_text = await extractor.extract_text()
    logger.info("✅ PDF text extraction completed.\n")

    # -----------------------------
    # Step 2: Prepare Inputs
    # -----------------------------
    inputs = split_text(extracted_text, max_chunk_size)
    logger.info(f"📝 Text split into {len(inputs)} chunks.\n")

    # -----------------------------
    # Step 3: Initialize OpenAI Client and Model Layers
    # -----------------------------
    # Ensure that the OPENAI_API_KEY environment variable is set
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("❌ Error: OPENAI_API_KEY environment variable not set.")
        logger.info("Please set it using 'export OPENAI_API_KEY=<your-api-key-here>' or set it in your environment.")
        sys.exit(1)

    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Initialize Global Rate Limiter
    global_limiter = AsyncLimiter(100, 60)  # 100 requests per 60 seconds

    # Initialize ModelPipeline
    pipeline = ModelPipeline()

    # Initialize Layer 1: Summarization
    layer1_def = layer_definitions[0]
    layer1 = ModelLayer(
        prompt_template=layer1_def["prompt_template"],
        model=layer1_def["model"],
        max_tokens=layer1_def["max_tokens"],
        temperature=layer1_def["temperature"],
        global_limiter=global_limiter
    )
    pipeline.add_layer(layer1)
    logger.info(f"🔗 Added Layer 1: Summarization")

    # Initialize Layer 2: Manim Code Generation
    layer2_def = layer_definitions[1]
    layer2 = ModelLayer(
        prompt_template=layer2_def["prompt_template"],
        model=layer2_def["model"],
        max_tokens=layer2_def["max_tokens"],
        temperature=layer2_def["temperature"],
        global_limiter=global_limiter
    )
    pipeline.add_layer(layer2)
    logger.info(f"🔗 Added Layer 2: Manim Code Generation")

    logger.info("\n🚀 Starting the model pipeline...\n")

    # -----------------------------
    # Step 4: Run the Pipeline with Progress Bars
    # -----------------------------
    # Create a Rich Progress instance
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        # Add tasks for each layer
        tasks = []
        for idx, layer in enumerate(pipeline.layers, start=1):
            layer_type = "Summarization" if idx == 1 else "Manim Code Generation"
            if idx == 1:
                total_tasks = len(inputs)
            else:
                # For Layer 2, the number of summaries depends on Layer 1's outputs
                total_tasks = len(layer1.get_outputs())
            task_description = f"Processing Layer {idx}: {layer_type}"
            task = progress.add_task(f"{task_description}...", total=total_tasks)
            tasks.append(task)

        # Function to update progress
        async def run_pipeline():
            current_inputs = inputs
            for idx, layer in enumerate(pipeline.layers, start=1):
                task = tasks[idx - 1]
                layer_type = "Summarization" if idx == 1 else "Manim Code Generation"
                logger.info(f"--- Processing Layer {idx}: {layer_type} ---")

                if idx == 1:
                    prompts = [layer.format_prompt({"text": chunk}) for chunk in current_inputs]
                elif idx == 2:
                    summaries = layer1.get_outputs()
                    parsed_summaries = []
                    for summary in summaries:
                        if not summary:
                            logger.warning("Empty summary received. Skipping...")
                            progress.advance(task)
                            continue
                        parsed = parse_summary(summary)
                        if parsed and isinstance(parsed, dict):
                            parsed_summaries.append(parsed)
                        else:
                            logger.warning(f"Failed to parse summary or invalid format: {truncate_text(summary)}. Skipping...")
                            progress.advance(task)

                    prompts = []
                    for summary in parsed_summaries:
                        title = summary.get("title", "Untitled")
                        content = summary.get("content", "")
                        if not content:
                            logger.warning("Empty content found. Skipping this summary.")
                            progress.advance(task)
                            continue
                        # Sanitize the scene class name
                        scene_class_name = re.sub(r'\W+', '_', title)  # Replace non-word characters with underscores
                        scene_class_name = scene_class_name.strip('_')  # Remove leading/trailing underscores
                        if not scene_class_name or scene_class_name[0].isdigit():
                            scene_class_name = "Untitled_Scene"
                        # Ensure CamelCase for class names
                        scene_class_name = ''.join(word.capitalize() for word in scene_class_name.split('_'))
                        prompt = layer.format_prompt({
                            "title": title,
                            "content": content,
                            "full_script_content": full_script_content,
                            "scene_class_name": scene_class_name
                        })
                        prompts.append(prompt)

                else:
                    prompts = current_inputs  # For additional layers

                if not prompts:
                    logger.warning(f"No prompts to process for Layer {idx}. Skipping...")
                    progress.update(task, completed=progress.tasks[task].total)
                    continue

                await layer.process_inputs(prompts)
                current_outputs = layer.get_outputs()

                for output in current_outputs:
                    if output:
                        truncated_output = truncate_text(output, max_length=500)
                        logger.debug(f"🔹 Generated Output: {output}")  # Log the full output for debugging
                        logger.info(f"🔹 Generated Output: {truncated_output}")
                        progress.advance(task)
                    else:
                        logger.warning("Received empty output from the model.")
                        progress.advance(task)

                logger.info(f"--- Completed Layer {idx} ---\n")
                current_inputs = current_outputs

        await run_pipeline()

    logger.info("🎉 Model pipeline processing completed.\n")

    # -----------------------------
    # Step 5: Insert Manim Code Snippets into Full Script
    # -----------------------------
    output_file = "generated_manim_scenes.py"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_script_content)
            f.write("\n\n# Generated Scene Classes\n")
            for i, snippet in enumerate(layer2.get_outputs()):
                if snippet and snippet.strip():
                    try:
                        # Remove regex-like patterns at the start and end of the snippet
                        cleaned_snippet = re.sub(r'^\\w*\\n?|\\s*$', '', snippet.strip())
                        # Try to compile the snippet to check for syntax errors
                        compile(cleaned_snippet, f"<string{i}>", 'exec')
                        f.write(cleaned_snippet + "\n\n")
                    except SyntaxError as e:
                        logger.error(f"Syntax error in scene {i+1}: {e}")
                        logger.debug(f"Problematic snippet:\n{cleaned_snippet}")
        logger.info(f"📄 Manim Scene classes have been generated and inserted into '{output_file}'.")
    except Exception as e:
        logger.error(f"❌ Failed to write to '{output_file}': {e}")

    # After writing the generated_manim_scenes.py file
    logger.debug("Contents of generated_manim_scenes.py:")
    with open(output_file, 'r') as f:
        scene_content = f.read()
        logger.debug(scene_content)
        scene_classes = re.findall(r'class (\w+Scene)\(Scene\):', scene_content)
        logger.info(f"Found {len(scene_classes)} scene classes: {', '.join(scene_classes)}")

    # -----------------------------
    # Step 6: Run Manim to Generate Video
    # -----------------------------
    if os.path.exists(output_file):
        logger.info("Running Manim to generate video...")
        with open(output_file, 'r') as f:
            content = f.read()
            scene_classes = re.findall(r'class (\w+Scene)\(Scene\):', content)

        if scene_classes:
            successful_scenes = await run_manim_async(scene_classes, python_path, output_dir)
            if successful_scenes:
                logger.info(f"Videos generated for the following scenes: {', '.join(successful_scenes)}")
                
                # Check if video files exist and move them to the output directory
                video_files = []
                for root, dirs, files in os.walk("./manim_output/scenes/"):
                    for file in files:
                        if file.endswith(".mp4"):
                            source_path = os.path.join(root, file)
                            dest_path = os.path.join(output_dir, file)
                            try:
                                if os.path.exists(dest_path):
                                    # If file already exists, create a unique name
                                    base, extension = os.path.splitext(file)
                                    counter = 1
                                    while os.path.exists(dest_path):
                                        dest_path = os.path.join(output_dir, f"{base}_{counter}{extension}")
                                        counter += 1
                                shutil.move(source_path, dest_path)
                                video_files.append(dest_path)
                                logger.info(f"Moved video: {dest_path}")
                            except Exception as e:
                                logger.error(f"Error moving file {source_path}: {e}")

                if not video_files:
                    logger.error("No video files found to combine")

                if video_files:
                    final_output_file = os.path.join(os.getcwd(), "final_video.mp4")
                    
                    logger.info("Combining videos...")
                    concat_file = os.path.join(output_dir, "concat_list.txt")
                    with open(concat_file, "w") as f:
                        for video in video_files:
                            f.write(f"file '{os.path.basename(video)}'\n")

                    concat_command = f"ffmpeg -f concat -safe 0 -i {concat_file} -c copy \"{final_output_file}\""
                    try:
                        subprocess.run(concat_command, shell=True, check=True, capture_output=True, text=True)
                        logger.info(f"Final video created: {final_output_file}")

                        # Optionally, play the final video automatically
                        # Uncomment the following lines if you want the video to play after creation
                        # webbrowser.open(final_output_file)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error combining videos: {e}")
                        logger.error(f"ffmpeg output: {e.output}")
                    except Exception as e:
                        logger.error(f"Error during video creation: {e}")

                    # Clean up concat file
                    os.remove(concat_file)
                else:
                    logger.error("No video files found to combine")
            else:
                logger.error("Failed to generate videos for all scenes")
        else:
            logger.warning("No scene classes found in the generated file.")
    else:
        logger.error(f"Generated Manim scenes file '{output_file}' not found.")

    # After processing, log the contents of the output directory
    logger.debug(f"Contents of {output_dir}:")
    for file in os.listdir(output_dir):
        logger.debug(f"- {file}")

    # Add this just before running Manim
    logger.debug("Contents of generated_manim_scenes.py:")
    with open(output_file, 'r') as f:
        logger.debug(f.read())

    # After the run_manim_async function call
    logger.debug("Contents of ./manim_output/scenes/:")
    for root, dirs, files in os.walk("./manim_output/scenes/"):
        for file in files:
            logger.debug(f"- {os.path.join(root, file)}")

    # Start GPU monitoring thread
    gpu_monitor_thread = threading.Thread(target=monitor_gpu_usage, daemon=True)
    gpu_monitor_thread.start()

    # Cleanup temporary files
    cleanup_temp_files()

    # Optionally, play the final video automatically after rendering
    final_output_file = os.path.join(os.getcwd(), "final_video.mp4")
    if os.path.exists(final_output_file):
        logger.info("Opening the final video...")
        webbrowser.open(f"file://{os.path.realpath(final_output_file)}")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✋ Process interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        # Log the full traceback
        logger.exception("❌ An unexpected error occurred:")
        sys.exit(1)