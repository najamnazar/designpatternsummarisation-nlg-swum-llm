package dps_llm.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * HTTP client for communicating with LLM APIs via the OpenRouter platform.
 * <p>
 * This client provides a simplified interface for making chat completion requests
 * to Large Language Models through OpenRouter's unified API. It handles request
 * construction, HTTP communication, and response parsing.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Construct properly formatted JSON payloads for chat completion requests</li>
 *   <li>Manage HTTP connections with timeout and retry logic</li>
 *   <li>Parse and extract text responses from LLM API responses</li>
 *   <li>Handle authentication and custom headers (referer, title)</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class LlmClient {

    private final String apiUrl;
    private final String apiKey;
    private final String model;
    private final int maxOutputTokens;
    private final double temperature;
    private final String httpReferer;
    private final String title;

    private final HttpClient httpClient;
    private final ObjectMapper mapper = new ObjectMapper();

    // CHANGE: retry settings. The class Javadoc promised retry logic but none existed, so a single
    // rate-limit (429) or transient server error (5xx) lost the summary. Requests are now retried
    // with exponential backoff (2s, 4s, 8s), honouring a numeric Retry-After header when present.
    private static final int MAX_ATTEMPTS = 4;
    private static final long INITIAL_BACKOFF_MILLIS = 2_000L;
    private static final long MAX_BACKOFF_MILLIS = 60_000L;

    /**
     * Constructs a new LLM client with the specified configuration.
     * 
     * @param apiUrl the base URL for the LLM API endpoint
     * @param apiKey the API key for authentication
     * @param model the model identifier (e.g., "mistralai/mixtral-8x22b-instruct")
     * @param maxOutputTokens maximum number of tokens in the response
     * @param temperature sampling temperature (0.0-1.0) for response generation
     * @param httpReferer optional HTTP referer header value
     * @param title optional title header for request identification
     */
    public LlmClient(String apiUrl,
                     String apiKey,
                     String model,
                     int maxOutputTokens,
                     double temperature,
                     String httpReferer,
                     String title) {
        if (apiUrl == null || apiUrl.isBlank()) {
            throw new IllegalArgumentException("apiUrl must not be null or blank");
        }
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalArgumentException("apiKey must not be null or blank");
        }
        if (model == null || model.isBlank()) {
            throw new IllegalArgumentException("model must not be null or blank");
        }
        if (maxOutputTokens <= 0) {
            throw new IllegalArgumentException("maxOutputTokens must be positive");
        }
        if (temperature < 0.0 || temperature > 2.0) {
            throw new IllegalArgumentException("temperature must be between 0.0 and 2.0");
        }
        
        this.apiUrl = apiUrl;
        this.apiKey = apiKey;
        this.model = model;
        this.maxOutputTokens = maxOutputTokens;
        this.temperature = temperature;
        this.httpReferer = httpReferer;
        this.title = title;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();
    }

    /**
     * Generates a summary by sending prompts to the LLM API.
     * <p>
     * Constructs a chat completion request with system and user messages,
     * sends it to the configured API endpoint, and extracts the generated text.
     * </p>
     * 
     * @param systemPrompt the system-level instruction that sets the behavior/role of the LLM
     * @param userPrompt the user message containing the content to summarize
     * @return an Optional containing the generated summary, or empty if no content returned
     * @throws LlmClientException if the API request fails or returns an error
     */
    public Optional<String> createSummary(String systemPrompt, String userPrompt) throws LlmClientException {
        if (systemPrompt == null) {
            throw new IllegalArgumentException("systemPrompt must not be null");
        }
        if (userPrompt == null) {
            throw new IllegalArgumentException("userPrompt must not be null");
        }

        // CHANGE: wrap the single request in a retry loop for 429, 5xx and network (IO) errors.
        // Other non-2xx statuses (e.g. 400 bad request, 401 bad key) still fail immediately.
        // Original implementation (single attempt):
        // try {
        //     String payload = buildPayload(systemPrompt, userPrompt);
        //     HttpRequest.Builder requestBuilder = ...;  // same request as built below
        //     HttpResponse<String> response = httpClient.send(requestBuilder.build(), HttpResponse.BodyHandlers.ofString());
        //     if (response.statusCode() < 200 || response.statusCode() >= 300) {
        //         throw new LlmClientException(String.format("LLM request failed (%d): %s", response.statusCode(), response.body()));
        //     }
        //     return extractMessage(response.body());
        // } catch (InterruptedException e) {
        //     Thread.currentThread().interrupt();
        //     throw new LlmClientException("LLM request interrupted", e);
        // } catch (IOException e) {
        //     throw new LlmClientException("LLM request error: " + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName()), e);
        // }
        HttpRequest request;
        try {
            request = buildRequest(buildPayload(systemPrompt, userPrompt));
        } catch (IOException e) {
            throw new LlmClientException("Failed to build LLM request: " + e.getMessage(), e);
        }

        long backoffMillis = INITIAL_BACKOFF_MILLIS;
        for (int attempt = 1; ; attempt++) {
            boolean lastAttempt = attempt >= MAX_ATTEMPTS;
            try {
                HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
                int status = response.statusCode();
                if (status >= 200 && status < 300) {
                    return extractMessage(response.body());
                }
                String failure = String.format("LLM request failed (%d): %s", status, response.body());
                if (!isRetryableStatus(status) || lastAttempt) {
                    throw new LlmClientException(failure);
                }
                long waitMillis = retryAfterMillis(response).orElse(backoffMillis);
                System.err.printf("  %s - retrying in %d ms (attempt %d/%d)%n", failure, waitMillis, attempt + 1, MAX_ATTEMPTS);
                sleep(waitMillis);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new LlmClientException("LLM request interrupted", e);
            } catch (IOException e) {
                String message = "LLM request error: " + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
                if (lastAttempt) {
                    throw new LlmClientException(message, e);
                }
                System.err.printf("  %s - retrying in %d ms (attempt %d/%d)%n", message, backoffMillis, attempt + 1, MAX_ATTEMPTS);
                sleep(backoffMillis);
            }
            backoffMillis = Math.min(backoffMillis * 2, MAX_BACKOFF_MILLIS);
        }
    }

    /**
     * Builds the HTTP request once so retries resend the identical payload.
     * CHANGE: extracted from createSummary when the retry loop was added.
     */
    private HttpRequest buildRequest(String payload) {
        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(URI.create(apiUrl))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(payload));

        if (httpReferer != null && !httpReferer.isBlank()) {
            requestBuilder.header("HTTP-Referer", httpReferer);
        }
        if (title != null && !title.isBlank()) {
            requestBuilder.header("X-Title", title);
        }
        return requestBuilder.build();
    }

    /** CHANGE: rate limits (429) and server-side errors (5xx) are worth retrying; client errors are not. */
    private static boolean isRetryableStatus(int status) {
        return status == 429 || status >= 500;
    }

    /** CHANGE: honour a numeric Retry-After header (seconds), capped at MAX_BACKOFF_MILLIS. */
    private static Optional<Long> retryAfterMillis(HttpResponse<?> response) {
        Optional<String> header = response.headers().firstValue("Retry-After");
        if (header.isEmpty()) {
            return Optional.empty();
        }
        try {
            long seconds = Long.parseLong(header.get().trim());
            return Optional.of(Math.min(Math.max(seconds, 0L) * 1000L, MAX_BACKOFF_MILLIS));
        } catch (NumberFormatException e) {
            return Optional.empty(); // HTTP-date form is not used by OpenRouter; fall back to backoff
        }
    }

    private static void sleep(long millis) throws LlmClientException {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new LlmClientException("LLM request interrupted while waiting to retry", e);
        }
    }

    /**
     * Constructs the JSON payload for an LLM API request.
     * 
     * @param systemPrompt the system message
     * @param userPrompt the user message
     * @return JSON string representation of the request payload
     * @throws IOException if JSON serialization fails
     */
    private String buildPayload(String systemPrompt, String userPrompt) throws IOException {
        Map<String, Object> json = new HashMap<>();
        json.put("model", model);
        json.put("max_tokens", maxOutputTokens);
        json.put("temperature", temperature);

        List<Map<String, String>> messages = new ArrayList<>();
        messages.add(createMessage("system", systemPrompt));
        messages.add(createMessage("user", userPrompt));
        json.put("messages", messages);

        return mapper.writeValueAsString(json);
    }

    /**
     * Creates a chat message map with role and content.
     * 
     * @param role the message role ("system", "user", or "assistant")
     * @param content the message content
     * @return a map representing the message
     */
    private Map<String, String> createMessage(String role, String content) {
        Map<String, String> message = new HashMap<>();
        message.put("role", role);
        message.put("content", content);
        return message;
    }

    /**
     * Extracts the text content from an LLM API response.
     * 
     * @param body the raw JSON response body
     * @return an Optional containing the extracted text, or empty if not found
     * @throws IOException if JSON parsing fails
     */
    private Optional<String> extractMessage(String body) throws IOException {
        JsonNode root = mapper.readTree(body);
        JsonNode choices = root.path("choices");
        if (!choices.isArray() || choices.isEmpty()) {
            return Optional.empty();
        }

        JsonNode messageNode = choices.get(0).path("message");
        if (!messageNode.has("content")) {
            return Optional.empty();
        }
        String content = messageNode.path("content").asText().trim();
        return content.isEmpty() ? Optional.empty() : Optional.of(content);
    }
}
