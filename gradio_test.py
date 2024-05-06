import gradio as gr 
with gr.Blocks() as demo:
    radio = gr.Radio([1, 2, 4], label="Set the value of the number") 
    number = gr.Number(value=2, interactive=True)
    radio.change(fn=lambda value: gr.update(value=value), inputs=radio, outputs=number)
demo.launch()