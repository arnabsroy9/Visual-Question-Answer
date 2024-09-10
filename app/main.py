import gradio as gr
from transformers import ViltProcessor, ViltForQuestionAnswering
# from PIL import Image

# Define the function to get the answer for the image and question
def get_image_answer(image, question: str):
    # Load the processor and model
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    
    # Convert PIL image to RGB if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Prepare the inputs
    encoding = processor(image, question, return_tensors="pt")
    
    # Perform the forward pass to get the answer
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    
    # Return the predicted answer
    return model.config.id2label[idx]

# Define Gradio interface
def gradio_interface(image, question):
    return get_image_answer(image, question)

# Create Gradio app using the new components
interface = gr.Interface(
    fn=gradio_interface, 
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Question")], 
    outputs=gr.Textbox(label="Answer"),
    title="Image-based Question Answering",
    description="Upload an image and ask a question about the image."
)

# Launch the app
interface.launch()
