import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "@" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

#demo.launch(server_name="192.168.0.134", server_port=8080, share=False)
# demo.launch(server_name="127.0.0.1", server_port=7777, share=True)
demo.launch(share=True, share_server_address="localhost:8080")
# demo.launch()
