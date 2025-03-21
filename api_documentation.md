# API Documentation

## /v1/chat/completions

This endpoint is used to create chat completions.

### Request

```json
{
  "model": "string",
  "messages": [
    {
      "role": "string",
      "content": "string"
    }
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "n": 1,
  "stream": false,
  "max_tokens": 900,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stop": null
}
```

#### Request Parameters

*   `model` (string, required): The name of the model to use.
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

### Response

```json
{
  "id": "string",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "string",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "string"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

#### Response Parameters

*   `id` (string): A unique identifier for the chat completion.
*   `object` (string): The object type, which is always "chat.completion".
*   `created` (integer): The timestamp of when the chat completion was created.
*   `model` (string): The model used for the chat completion.
*   `choices` (array): A list of chat completion choices.
    *   `index` (integer): The index of the choice in the list.
    *   `message` (object): The chat completion message.
        *   `role` (string): The role of the message sender (e.g., "assistant").
        *   `content` (string): The content of the message.
    *   `finish_reason` (string): The reason the chat completion finished (e.g., "stop").
*   `usage` (object): Usage statistics for the chat completion.
    *   `prompt_tokens` (integer): The number of tokens in the prompt.
    *   `completion_tokens` (integer): The number of tokens in the completion.
    *   `total_tokens` (integer): The total number of tokens used.

### Example Request

```json
{
  "model": "SmolLM2-360M-Instruct-openvino-8bit",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ]
}
```

### Example Response

```json
{
  "id": "chatcmpl-1677652288",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "SmolLM2-360M-Instruct-openvino-8bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}

## /v1/models

This endpoint is used to list the available models.

### Response

```json
[
  {
    "id": "string",
    "object": "model",
    "created": 1677652288,
    "owned_by": "string",
    "permission": [
      {
        "id": "string",
        "object": "model_permission",
        "allow_create_engine": true,
        "allow_sampling": true,
        "allow_logprobs": true,
        "allow_search_indices": false,
        "allow_view": false,
        "allow_fine_tuning": false,
        "organization": "*",
        "group": null,
        "is_blocking": false
      }
    ]
  }
]
```

#### Response Parameters

*   `id` (string): The ID of the model.
*   `object` (string): The object type, which is always "model".
*   `created` (integer): The timestamp of when the model was created.
*   `owned_by` (string): The owner of the model.
*   `permission` (array): The permissions for the model.
    *   `id` (string): The ID of the permission.
    *   `object` (string): The object type, which is always "model_permission".
    *   `allow_create_engine` (boolean): Whether the model can be used to create engines.
    *   `allow_sampling` (boolean): Whether the model can be used for sampling.
    *   `allow_logprobs` (boolean): Whether the model can be used for log probabilities.
    *   `allow_search_indices` (boolean): Whether the model can be used for search indices.
    *   `allow_view` (boolean): Whether the model can be viewed.
    *   `allow_fine_tuning` (boolean): Whether the model can be used for fine-tuning.
    *   `organization` (string): The organization that owns the model.
    *   `group` (string): The group that owns the model.
    *   `is_blocking` (boolean): Whether the model is blocking.

## /health

This endpoint is used to check the health of the API.

### Response

```json
{
  "status": "ok"
}
```

#### Response Parameters

*   `status` (string): The status of the API.