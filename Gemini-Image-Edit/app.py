import json
import os
import time
import uuid
import tempfile
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import base64
import mimetypes

from google import genai
from google.genai import types

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)

def generate(text, file_name, api_key, model="gemini-2.5-flash-image"):
    # Initialize client using provided api_key (or fallback to env variable)
    client = genai.Client(api_key=(api_key.strip() if api_key and api_key.strip() != ""
                                   else os.environ.get("GEMINI_API_KEY")))
    
    files = [ client.files.upload(file=file_name) ]
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text=text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_modalities=["TEXT", "IMAGE"],
    )

    text_response = ""
    image_path = None
    # Create a temporary file to potentially store image data.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            candidate = chunk.candidates[0].content.parts[0]
            # Check for inline image data
            if candidate.inline_data:
                save_binary_file(temp_path, candidate.inline_data.data)
                print(f"File of mime type {candidate.inline_data.mime_type} saved to: {temp_path} and prompt input: {text}")
                image_path = temp_path
                # If an image is found, we assume that is the desired output.
                break
            else:
                # Accumulate text response if no inline_data is present.
                text_response += chunk.text + "\n"
    
    del files
    return image_path, text_response

def process_image_and_prompt(composite_pil, prompt, gemini_api_key, model):
    if composite_pil is None:
        raise gr.Error("Silakan unggah gambar terlebih dahulu!", duration=5)
    if not prompt or prompt.strip() == "":
        raise gr.Error("Silakan masukkan prompt pengeditan terlebih dahulu!", duration=5)
    if not gemini_api_key or gemini_api_key.strip() == "":
        if not os.environ.get("GEMINI_API_KEY"):
            raise gr.Error("API Key Gemini tidak ditemukan! Silakan masukkan API Key Anda di kolom yang disediakan.", duration=10)
            
    try:
        # Save the composite image to a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            composite_path = tmp.name
            composite_pil.save(composite_path)
        
        file_name = composite_path  
        input_text = prompt 

        image_path, text_response = generate(text=input_text, file_name=file_name, api_key=gemini_api_key, model=model)
        
        if image_path:
            # Load and convert the image if needed.
            result_img = Image.open(image_path)
            if result_img.mode == "RGBA":
                result_img = result_img.convert("RGB")
            return [result_img], ""  # Return image in gallery and empty text output.
        else:
            # Return no image and the text response.
            return None, text_response
    except Exception as e:
        err_msg = str(e)
        if "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            raise gr.Error(
                "Quota Terbatas / RESOURCE_EXHAUSTED! Model pembuatan gambar Gemini memerlukan kuas/proyek dengan billing aktif (paid tier) di Google AI Studio. Silakan aktifkan penagihan pada akun AI Studio Anda.",
                duration=15
            )
        elif "response modalities" in err_msg.lower() or "modality" in err_msg.lower() or "supports text output" in err_msg.lower():
            raise gr.Error(
                f"Model '{model}' tidak mendukung pembuatan/pengeditan gambar. Silakan pilih model khusus gambar seperti 'gemini-2.5-flash-image' atau 'gemini-3.1-flash-image'.",
                duration=15
            )
        raise gr.Error(f"Error Getting {err_msg}", duration=15)

# --- KONFIGURASI TEMA MAROON & BEIGE ---
my_theme = gr.themes.Default().set(
    body_background_fill="#F5F5DC",          # Warna background keseluruhan: Beige
    button_primary_background_fill="#800000", # Warna tombol utama: Maroon
    button_primary_background_fill_hover="#5A0000", # Warna tombol saat di-hover: Maroon Gelap
    button_primary_text_color="#F5F5DC",     # Warna teks dalam tombol: Beige
    block_title_text_color="#800000",        # Warna teks judul box: Maroon
    block_label_text_color="#800000",        # Warna label inputan: Maroon
    border_color_primary="#800000"           # Warna garis pinggir box: Maroon
)

# Build a Blocks-based interface with custom Theme, HTML header and CSS
with gr.Blocks() as demo:
    
    # Custom HTML header dengan informasi milik 
    gr.HTML(
    """
    <div class="header-container">
      <div>
          <img src="https://www.gstatic.com/lamda/images/gemini_favicon_f069958c85030456e93de685481c559f160ea06b.png" alt="App logo">
      </div>
      <div>
          <h1>AI Image Editor by Dzikri</h1>
          <p>Powered by <a href="https://gradio.app/">Gradio</a>⚡️ & Gemini 2.0 | 
          <a href="https://aistudio.google.com/apikey">Get an API Key</a> <br> 
          Follow me on: 
          <a href="https://www.instagram.com/owi.woi/" target="_blank">Instagram</a> | 
          <a href="https://github.com/dzikri007" target="_blank">GitHub</a></p>
      </div>
    </div>
    """
    )
    
    with gr.Accordion("⚠️ API Configuration ⚠️", open=False, elem_classes="config-accordion"):
        gr.Markdown("""
    - **Issue:** ❗ Sometimes the model returns text instead of an image.  
    ### 🔧 Steps to Address:
    1. **🔑 Use Your Own Gemini API Key** - Get it for free from [Google AI Studio](https://aistudio.google.com/apikey).
       - You **must** configure your own Gemini key for generation!  
    """)

    with gr.Accordion("📌 Usage Instructions", open=False, elem_classes="instructions-accordion"):
        gr.Markdown("""
    ### 📌 Cara Pakai (Made by Owi)
    - Upload gambar yang mau lu edit.
    - Ketik perintah (prompt) pengeditannya (contoh: "jadikan fotonya hitam putih").
    - Jika API error atau gagal bikin gambar, dia bakal balikin respon teks.
    - ❌ **Tolong jangan upload gambar NSFW/Aneh-aneh!**
    """)

    with gr.Row(elem_classes="main-content"):
        with gr.Column(elem_classes="input-column"):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                image_mode="RGBA",
                elem_id="image-input",
                elem_classes="upload-box"
            )
            gemini_api_key = gr.Textbox(
                lines=1,
                placeholder="Enter Gemini API Key (optional)",
                label="Gemini API Key (optional)",
                elem_classes="api-key-input"
            )
            model_input = gr.Dropdown(
                choices=["gemini-2.5-flash-image", "gemini-3.1-flash-image", "gemini-3-pro-image"],
                value="gemini-2.5-flash-image",
                label="Model Gemini (Image Output)",
                elem_classes="model-select"
            )
            prompt_input = gr.Textbox(
                lines=2,
                placeholder="Mau edit apa hari ini? Ketik di sini...",
                label="Prompt Pengeditan",
                elem_classes="prompt-input"
            )
            submit_btn = gr.Button("Generate", elem_classes="generate-btn", variant="primary")
        
        with gr.Column(elem_classes="output-column"):
            output_gallery = gr.Gallery(label="Hasil Render", elem_classes="output-gallery")
            output_text = gr.Textbox(
                label="Pesan dari AI", 
                placeholder="Kalau AI gagal bikin gambar, pesannya muncul di sini ya.",
                elem_classes="output-text"
            )

    # Set up the interaction with two outputs.
    submit_btn.click(
        fn=process_image_and_prompt,
        inputs=[image_input, prompt_input, gemini_api_key, model_input],
        outputs=[output_gallery, output_text],
    )
    
    gr.Markdown("## Coba Contoh Berikut", elem_classes="gr-examples-header")
    
    examples = [
        ["data/1.webp", 'change text to "OWI"'],
        ["data/2.webp", "remove the spoon from hand only"],
        ["data/3.webp", 'change text to "Make it Awesome"'],
        ["data/1.jpg", "add joker style only on face"],
        ["data/1777043.jpg", "add joker style only on face"],
        ["data/2807615.jpg", "add lipstick on lip only"],
        ["data/76860.jpg", "add lipstick on lip only"],
        ["data/2807615.jpg", "make it happy looking face only"],
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[image_input, prompt_input,],
        elem_id="examples-grid"
    )

demo.queue(max_size=50).launch(mcp_server=True, theme=my_theme, css_paths="style.css")