# Handwritten Notes Microservice 🖋️

This service accepts a structured Roadmap JSON and returns a Base64 encoded PDF of "handwritten" notes.

## 🏗️ Expected Input Structure (Data Contract)
The microservice expects a POST request with a JSON body. The `roadmap` object must follow the Pydantic schema defined in `models.py`.

### Full JSON Example
```json
{
  "roadmap": {
    "id": "string",
    "title": "string",
    "author": "string",
    "description": "string",
    "date": "YYYY-MM-DD",
    "modules": [
      {
        "name": "string",
        "subtitle": "string",
        "content": "string",
        "sections": [
          {
            "heading": "string",
            "title": "string",
            "content": "string",
            "bullets": [
              { "text": "string" }
            ]
          }
        ]
      }
    ]
  }
}
handwritten_notes/
├── handler.py
├── requirements.txt
├── Dockerfile,buildspec
└── src/
    ├── __init__.py
    ├── models.py
    ├── pdf_generator.py
    └── notebook.html