# Serefine Chatbot

## Features

- Patient data management and retrieval
- Chat functionality with patient data
- Integration with AWS Bedrock for LLM capabilities
- Streaming responses for chat interactions

## Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- AWS account with Bedrock access (for LLM functionality)

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/MuzammilMeh/serefine_bot.git
   cd <project-directory>
   ```

2. Set up the environment:
   ```
   pip install -r requirements.txt
   
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=your_aws_region
   PHOENIX_API_KEY=your_phoenix_api_key
   ```

4. Run the development server:
   ```
   python main.py
   ```

## API Endpoints

### Chat Endpoint

To interact with the chat functionality:

```
curl --location 'localhost:8000/api/chat' \
--header 'Content-Type: application/json' \
--data '{ "messages": [{ "role": "user", "content": "Hello" }] }'
```

You can start editing the API by modifying `app/api/routers/chat.py`. The endpoint auto-updates as you save the file.

Open [http://localhost:8000/docs](http://localhost:8000/docs) with your browser to see the Swagger UI of the API.

The API allows CORS for all origins to simplify development. You can change this behavior by setting the `ENVIRONMENT` environment variable to `prod`:

```
ENVIRONMENT=prod uvicorn main:app
```
