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
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


@app.after_request
def add_cors(response):
    """Stamp CORS headers on every response so the browser never blocks."""
    for key, value in CORS_HEADERS.items():
        response.headers[key] = value
    return response


def build_messages(prompt, file_data, file_mime):
    content = []
    if file_data:
        mime = (file_mime or "application/octet-stream").lower()
        if mime.startswith("image/"):
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": file_data},
            })
        else:
            fmt_map = {
                "application/pdf": "pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
                "text/csv": "csv",
                "application/json": "json",
                "text/html": "html",
                "text/plain": "txt",
            }
            content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": mime, "data": file_data},
                "format": fmt_map.get(mime, "txt"),
            })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def generate(prompt, file_data, file_mime):
    messages = build_messages(prompt, file_data, file_mime)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 1,
        "top_p": 1,
        "messages": messages,
    })
    response = bedrock.invoke_model_with_response_stream(
        modelId=MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    stream = response.get("body")
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                data = json.loads(chunk["bytes"].decode("utf-8"))
                if data.get("type") == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200)


@app.route("/", methods=["POST"])
def invoke():
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt", "write the conclusion of the generated menu")
    file_data = data.get("file_data")
    file_mime = data.get("file_mime")

    return Response(
        stream_with_context(generate(prompt, file_data, file_mime)),
        content_type="text/plain; charset=utf-8",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
