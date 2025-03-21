# OpenAI Compatible API Server with OpenVINO

This is an OpenAI compatible API server that uses OpenVINO for inference.

## Endpoints

*   `/v1/chat/completions`: This endpoint is used to create chat completions.
*   `/v1/models`: This endpoint is used to list the available models.
*   `/health`: This endpoint is used to check the health of the API.

## Usage

To use the API, you need to send a request to the `/v1/chat/completions` endpoint with a JSON payload that contains the following parameters:

*   `model` (string, optional): The name of the model to use. If not specified, the default model is used.
*   `messages` (array, required): A list of messages comprising the conversation so far.
    *   `role` (string, required): The role of the message sender (e.g., "user" or "assistant").
    *   `content` (string, required): The content of the message.
*   `temperature` (number, optional, default: 0.7): What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
*   `top_p` (number, optional, default: 1.0): An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top\_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
*   `n` (integer, optional, default: 1): How many chat completion choices to generate for each input message.
*   `stream` (boolean, optional, default: false): Whether to stream back partial progress. If set, tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).
*   `max_tokens` (integer, optional, default: 900): The maximum number of [tokens](/tokenizer#how-to-count-tokens) to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length.
*   `presence_penalty` (number, optional, default: 0.0): Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
*   `frequency_penalty` (number, optional, default: 0.0): Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same lines verbatim.
*   `stop` (string or array, optional, default: null): Up to 4 sequences where the API will stop generating further tokens.

You can find more information about the API in the `api_documentation.md` file.

## Model Directory

The model should be present in the `./models` directory. The directory structure should be as follows:

```
./models/
└── <model_name>/
    ├── openvino_model.bin
    ├── openvino_tokenizer.bin
    ├── openvino_detokenizer.bin
    └── tokenizer_config.json
```

*   `openvino_model.bin`: The OpenVINO model file.
*   `openvino_tokenizer.bin`: The OpenVINO tokenizer file.
*   `openvino_detokenizer.bin`: The OpenVINO detokenizer file.
*   `tokenizer_config.json`: The tokenizer configuration file.
