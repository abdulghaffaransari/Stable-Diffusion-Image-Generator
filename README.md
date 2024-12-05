
# **Text-to-Image Generation using Stable Diffusion and Diffusers**

This project demonstrates the use of **Stable Diffusion**, **Diffusers**, and **PyTorch** to generate high-quality and creative images from textual prompts. The repository includes an interactive Python notebook for generating stunning visuals using the Dreamlike Art model.

---

## **Introduction**
Stable Diffusion is a powerful **text-to-image generation** model that creates photorealistic or artistic visuals from textual descriptions. Using the **Hugging Face Diffusers library**, we simplify the process by loading pre-trained pipelines optimized for GPUs and fine-tuning configurations for better performance. This starter project is perfect for those entering the field of **generative AI**, offering practical insights into **image generation**, **prompt engineering**, and **pipeline customization**.

---

## **Features**
1. **Text-to-Image Generation**: Generate images from textual prompts with cinematic and artistic effects.
2. **Customizable Parameters**: Adjust inference steps, image dimensions, and the number of generated images.
3. **FP16 Optimization**: Leverage **FP16 precision** for reduced memory usage on GPUs.
4. **Pretrained Model**: Use the **Dreamlike Diffusion 1.0** model, known for producing artistic and surreal images.
5. **Interactive Visualization**: View and compare multiple images side-by-side using **Matplotlib**.

---

## **Technology Stack**
- **Stable Diffusion**: Model used for text-to-image generation.
- **Hugging Face Diffusers**: Simplifies the use of pre-trained pipelines.
- **PyTorch**: Backend framework for running the model efficiently on GPUs.
- **Matplotlib**: For visualizing and displaying generated images.
- **xFormers**: Memory-efficient attention optimization for better GPU performance.

---

## **Installation**
Ensure you have **Python 3.8+** and **pip** installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/abdulghaffaransari/Stable-Diffusion-Image-Generator.git
   cd Stable-Diffusion-Image-Generator
   ```

2. Install the required packages:
   ```bash
   pip install diffusers accelerate transformers torch matplotlib xformers -q
   ```

3. Verify PyTorch installation:
   ```bash
   pip show torch
   ```

---

## **Usage**
Follow the steps in the **notebook** to run the pipeline and generate images.

### **Steps:**
1. **Load the Pre-trained Model**:
   ```python
   from diffusers import StableDiffusionPipeline
   import torch

   model_id = "dreamlike-art/dreamlike-diffusion-1.0"
   pipe = StableDiffusionPipeline.from_pretrained(
       model_id,
       torch_dtype=torch.float16,  # Memory optimization
       use_safetensors=True
   ).to("cuda")
   ```

2. **Generate Images from Prompts**:
   Define a prompt and run the pipeline to generate an image:
   ```python
   prompt = "A cinematic golden iris scene of a girl with her tiger."
   image = pipe(prompt).images[0]
   image.show()
   ```

3. **Experiment with Parameters**:
   Customize inference steps, image resolution, and the number of generated images:
   ```python
   params = {'num_inference_steps': 100, 'width': 512, 'height': 768, 'num_images_per_prompt': 2}
   images = pipe(prompt, **params).images
   for img in images:
       img.show()
   ```

---

## **Code Overview**
The main code is divided into the following steps:

### 1. **Pipeline Initialization**
   - Loads the **Dreamlike Diffusion 1.0** model with **FP16 optimization** for memory-efficient GPU usage.

### 2. **Prompt Definition**
   - Users input textual descriptions of the desired image. Prompts can range from abstract concepts to detailed scenes.

### 3. **Image Generation**
   - Generates single or multiple images based on the given parameters (e.g., `width`, `height`, `inference steps`).

### 4. **Visualization**
   - Displays generated images using **Matplotlib** for comparison.

---

## **Example Results**
### Prompt 1:
**Input**: `"A girl sitting on a chair with her tiger in a cinematic golden iris scene."`  
**Output**:  
*An elegant image of a girl with a tiger rendered in cinematic style.*

### Prompt 2:
**Input**: `"Beautiful girl playing the festival of colors in traditional Indian attire."`  
**Output**:  
*Vivid imagery of a girl throwing colors, celebrating Holi.*

---

## **Customization**
You can tweak the following parameters:
- `num_inference_steps`: Controls the quality of the image (higher steps = better quality but slower).
- `width` and `height`: Adjusts the image resolution.
- `num_images_per_prompt`: Generates multiple images for a single prompt.

---

## **Model Details**
- **Model Name**: Dreamlike Diffusion 1.0
- **Source**: [Dreamlike Art on Hugging Face](https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0)
- **Description**: Dreamlike Diffusion specializes in artistic, cinematic, and surreal image generation.

---

## **Potential Applications**
- **Creative Projects**: Generate artworks, illustrations, or designs.
- **Marketing**: Visual content for campaigns and storytelling.
- **Education**: Teach students about generative AI and machine learning.
- **Prototyping**: Quickly visualize ideas or concepts.

---

## **Future Improvements**
- Integrate more advanced models like **Stable Diffusion XL**.
- Add a web interface using **Gradio** for a user-friendly experience.
- Include additional customization features like style and theme settings.

---

## **Contributing**
Contributions are welcome! Feel free to fork the repository and submit a pull request.

---

## **License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
Special thanks to the teams behind **Hugging Face**, **Stable Diffusion**, and **Dreamlike Art** for providing such incredible tools and models.

---

This project serves as an excellent starting point for anyone looking to explore **generative AI** and **text-to-image models**. Get creative and share your results! 🚀