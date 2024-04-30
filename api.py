from gradio_client import Client, file

client = Client("http://localhost:7777/")

client.predict(
  name="adad",
  intensity=36,
  api_name="/predict"
)
