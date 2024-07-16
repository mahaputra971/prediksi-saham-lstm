import base64
from . import exception_handler

@exception_handler
def blob_to_data_url(blob, mime_type="image/png"):
    base64_data = base64.b64encode(blob).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"