const DEFAULT_MAX_REQUEST_ATTEMPTS = 3;
const DEFAULT_REQUEST_TIMEOUT_MS = 90000;
const BASE_RETRY_DELAY_MS = 750;
const MAX_RETRY_DELAY_MS = 8000;

export function createOpenAIClient(config) {
  const endpoint = chatCompletionsUrl(config.baseUrl);

  return {
    async ask({ messages }) {
      return retryRequest(() => askOnce({ endpoint, config, messages }), config);
    }
  };
}

async function askOnce({ endpoint, config, messages }) {
  let parsed;
  try {
    parsed = await sendChatCompletion({ endpoint, config, messages, useResponseFormat: true });
  } catch (error) {
    if (!isResponseFormatUnsupported(error)) {
      throw error;
    }
    parsed = await sendChatCompletion({ endpoint, config, messages, useResponseFormat: false });
  }
  let content = extractContent(parsed);

  if (!content) {
    parsed = await sendChatCompletion({ endpoint, config, messages, useResponseFormat: false });
    content = extractContent(parsed);
  }

  if (!content) {
    const message = parsed.choices?.[0]?.message;
    const available = message && typeof message === "object"
      ? Object.keys(message).join(", ")
      : "none";
    throw new ApiRequestError(`API response did not include assistant content. Message fields: ${available}`, {
      retryable: true
    });
  }

  return {
    content,
    usage: parsed.usage ?? null
  };
}

async function sendChatCompletion({ endpoint, config, messages, useResponseFormat }) {
  const requestBody = {
    model: config.model,
    messages,
    temperature: config.temperature,
    max_tokens: config.maxTokens
  };

  if (config.reasoningEffort) {
    requestBody.reasoning_effort = config.reasoningEffort;
  }

  const enableThinking = resolveEnableThinking(config);
  if (enableThinking !== undefined) {
    requestBody.chat_template_kwargs = {
      enable_thinking: enableThinking
    };
  }

  if (useResponseFormat) {
    requestBody.response_format = { type: "json_object" };
  }

  const timeoutMs = requestTimeoutMs(config);
  const controller = new AbortController();
  const timeout = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  let response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${config.apiKey}`
      },
      body: JSON.stringify(requestBody),
      signal: controller.signal
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new ApiRequestError(`API request timed out after ${timeoutMs}ms`, {
        retryable: true
      });
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }

  const responseBody = await response.text();
  let parsed;
  try {
    parsed = JSON.parse(responseBody);
  } catch {
    parsed = responseBody;
  }

  if (!response.ok) {
    const message = extractApiErrorMessage(parsed) ?? responseBody;
    throw new ApiRequestError(`API request failed (${response.status}): ${message}`, {
      retryable: isRetryableStatus(response.status),
      retryAfterMs: parseRetryAfterMs(response.headers.get("retry-after")),
      status: response.status
    });
  }

  return parsed;
}

async function retryRequest(operation, config) {
  const configuredMaxAttempts = Number(config.maxRequestAttempts ?? DEFAULT_MAX_REQUEST_ATTEMPTS);
  const maxAttempts = Number.isFinite(configuredMaxAttempts)
    ? Math.max(1, configuredMaxAttempts)
    : DEFAULT_MAX_REQUEST_ATTEMPTS;
  let lastError = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt >= maxAttempts || !isRetryableError(error)) {
        throw annotateFinalError(error, attempt);
      }
      await sleep(retryDelayMs(error, attempt));
    }
  }

  throw lastError;
}

class ApiRequestError extends Error {
  constructor(message, { retryable = false, retryAfterMs = null, status = null } = {}) {
    super(message);
    this.name = "ApiRequestError";
    this.retryable = retryable;
    this.retryAfterMs = retryAfterMs;
    this.status = status;
  }
}

function isRetryableError(error) {
  if (error?.retryable === true) {
    return true;
  }
  return ["AbortError", "TimeoutError", "TypeError"].includes(error?.name);
}

function isRetryableStatus(status) {
  return status === 408 || status === 409 || status === 425 || status === 429 || status >= 500;
}

function retryDelayMs(error, attempt) {
  if (Number.isFinite(error?.retryAfterMs)) {
    return Math.min(error.retryAfterMs, MAX_RETRY_DELAY_MS);
  }
  return Math.min(BASE_RETRY_DELAY_MS * (2 ** (attempt - 1)), MAX_RETRY_DELAY_MS);
}

function requestTimeoutMs(config) {
  const configuredTimeoutMs = Number(config.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS);
  return Number.isFinite(configuredTimeoutMs) && configuredTimeoutMs > 0
    ? configuredTimeoutMs
    : DEFAULT_REQUEST_TIMEOUT_MS;
}

function annotateFinalError(error, attempts) {
  if (!isRetryableError(error) || attempts <= 1) {
    return error;
  }
  return new Error(`${error.message} (after ${attempts} attempts)`, { cause: error });
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function isResponseFormatUnsupported(error) {
  if (error?.status !== 400) {
    return false;
  }
  return /response_format|json_object|json mode|extra arguments/iu.test(error.message);
}

function extractApiErrorMessage(parsed) {
  if (!parsed || typeof parsed !== "object") {
    return null;
  }
  if (typeof parsed.error?.message === "string") {
    return parsed.error.message;
  }
  if (typeof parsed.error === "string") {
    return parsed.error;
  }
  if (typeof parsed.message === "string") {
    return parsed.message;
  }
  return null;
}

function parseRetryAfterMs(value) {
  if (!value) {
    return null;
  }
  const seconds = Number(value);
  if (Number.isFinite(seconds)) {
    return Math.max(0, seconds * 1000);
  }
  const timestamp = Date.parse(value);
  if (Number.isFinite(timestamp)) {
    return Math.max(0, timestamp - Date.now());
  }
  return null;
}

function extractContent(parsed) {
  const choice = parsed.choices?.[0];
  const content = choice?.message?.content
    ?? choice?.text
    ?? choice?.message?.reasoning_content
    ?? choice?.message?.reasoning;

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        return part?.text ?? "";
      })
      .join("")
      .trim();
  }

  return typeof content === "string" ? content.trim() : "";
}

function resolveEnableThinking(config) {
  if (config.enableThinking !== undefined) {
    return config.enableThinking;
  }

  if (shouldDisableThinkingByDefault(config.model)) {
    return false;
  }

  return undefined;
}

function shouldDisableThinkingByDefault(model = "") {
  const normalized = model.toLowerCase();
  const isQwenThinkingModel =
    normalized.includes("qwen/qwen3.5-") ||
    normalized.includes("qwen/qwen3.6-");

  if (!isQwenThinkingModel) {
    return false;
  }

  const provider = normalized.includes(":") ? normalized.split(":").at(-1) : "";
  return ["", "together", "deepinfra", "fastest", "cheapest"].includes(provider);
}

function chatCompletionsUrl(baseUrl) {
  const trimmed = baseUrl.replace(/\/+$/u, "");
  if (trimmed.endsWith("/chat/completions")) {
    return trimmed;
  }
  return `${trimmed}/chat/completions`;
}
