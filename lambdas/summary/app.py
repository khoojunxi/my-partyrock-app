"""
Summary Lambda — Flask streaming app for the File Upload Summary widget.

Accepts POST / with JSON body:
  {
    "prompt": "<text>",
    "file_data": "<base64 string>",   # optional
    "file_mime": "<mime type>"         # optional, required when file_data present
  }

Streams Bedrock response token-by-token back to the client.
"""

import json
import os
import boto3
from flask import Flask, Response, request, stream_with_context

app = Flask(__name__)

BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-2")
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def build_messages(prompt: str, file_data: str | None, file_mime: str | None) -> list:
    """Build the Bedrock messages list, prepending a file block when provided."""
    content: list = []

    if file_data:
        mime = (file_mime or "application/octet-stream").lower()
        if mime.startswith("image/"):
            fmt_map = {
                "image/jpeg": "jpeg",
                "image/jpg":  "jpeg",
                "image/png":  "png",
                "image/gif":  "gif",
                "image/webp": "webp",
            }
            img_fmt = fmt_map.get(mime, "jpeg")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": file_data,
                },
            })
        else:
            doc_fmt_map = {
                "application/pdf": "pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
                "text/csv":        "csv",
                "application/json": "json",
                "text/html":       "html",
                "text/plain":      "txt",
            }
            doc_fmt = doc_fmt_map.get(mime, "txt")
            content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": file_data,
                },
                "format": doc_fmt,
            })

    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


def generate(prompt: str, file_data: str | None, file_mime: str | None):
    """Generator that yields text chunks from Bedrock streaming response."""
    messages = build_messages(prompt, file_data, file_mime)

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 1,
        "top_p": 1,
        "messages": messages,
    }

    response = bedrock.invoke_model_with_response_stream(
        modelId=MODEL_ID,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )

    stream = response.get("body")
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_data = json.loads(chunk["bytes"].decode("utf-8"))
                if chunk_data.get("type") == "content_block_delta":
                    delta = chunk_data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def invoke():
    data = request.get_json(force=True, silent=True) or {}
    prompt    = data.get("prompt", "write the conclusion of the generated menu")
    file_data = data.get("file_data")  # may be None
    file_mime = data.get("file_mime")

    def stream():
        for chunk in generate(prompt, file_data, file_mime):
            yield chunk

    resp = Response(
        stream_with_context(stream()),
        content_type="text/plain; charset=utf-8",
        headers=CORS_HEADERS,
    )
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
