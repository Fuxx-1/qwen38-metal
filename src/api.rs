use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    error::Error,
    fmt,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use subtle::ConstantTimeEq;
use tokio::{net::TcpListener, sync::Semaphore, task};

const DEFAULT_MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescriptor {
    pub id: String,
    pub context_tokens: u32,
    pub execution: ExecutionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionKind {
    Fixture,
    Native,
}

impl ExecutionKind {
    fn label(self) -> &'static str {
        match self {
            Self::Fixture => "fixture",
            Self::Native => "native",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptMessage {
    pub role: PromptRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub messages: Vec<PromptMessage>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    StopSequence,
}

impl FinishReason {
    fn openai_label(self) -> &'static str {
        match self {
            Self::Stop | Self::StopSequence => "stop",
            Self::Length => "length",
        }
    }

    fn anthropic_label(self) -> &'static str {
        match self {
            Self::Stop => "end_turn",
            Self::Length => "max_tokens",
            Self::StopSequence => "stop_sequence",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    pub text: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub finish_reason: FinishReason,
}

pub trait InferenceEngine: Send + Sync + 'static {
    fn descriptor(&self) -> ModelDescriptor;

    fn estimate_prompt_tokens(&self, messages: &[PromptMessage]) -> Result<u32, EngineError>;

    fn generate(&self, request: GenerationRequest) -> Result<Generation, EngineError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    ContextLimit { requested: u32, maximum: u32 },
    Unavailable(String),
    Failure(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextLimit { requested, maximum } => write!(
                formatter,
                "requested {requested} tokens but the model supports at most {maximum}"
            ),
            Self::Unavailable(message) | Self::Failure(message) => formatter.write_str(message),
        }
    }
}

impl Error for EngineError {}

/// A deterministic engine used exclusively to exercise protocol clients before
/// the native Qwen execution path is available.
#[derive(Debug, Clone)]
pub struct FixtureEngine {
    descriptor: ModelDescriptor,
    response: String,
}

impl FixtureEngine {
    pub fn new(
        model_id: impl Into<String>,
        context_tokens: u32,
        response: impl Into<String>,
    ) -> Self {
        Self {
            descriptor: ModelDescriptor {
                id: model_id.into(),
                context_tokens,
                execution: ExecutionKind::Fixture,
            },
            response: response.into(),
        }
    }
}

impl InferenceEngine for FixtureEngine {
    fn descriptor(&self) -> ModelDescriptor {
        self.descriptor.clone()
    }

    fn estimate_prompt_tokens(&self, messages: &[PromptMessage]) -> Result<u32, EngineError> {
        let token_count = messages.iter().try_fold(0_u32, |total, message| {
            total
                .checked_add(4)
                .and_then(|value| value.checked_add(estimated_text_tokens(&message.content)))
        });
        token_count
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| EngineError::Failure("prompt token count overflowed u32".to_owned()))
    }

    fn generate(&self, request: GenerationRequest) -> Result<Generation, EngineError> {
        let input_tokens = self.estimate_prompt_tokens(&request.messages)?;
        let requested = input_tokens
            .checked_add(request.max_tokens)
            .ok_or_else(|| {
                EngineError::Failure("requested token count overflowed u32".to_owned())
            })?;
        if requested > self.descriptor.context_tokens {
            return Err(EngineError::ContextLimit {
                requested,
                maximum: self.descriptor.context_tokens,
            });
        }

        let (mut text, truncated) =
            truncate_to_estimated_tokens(&self.response, request.max_tokens);
        let mut finish_reason = if truncated {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        if let Some(stop_index) = first_stop_index(&text, &request.stop) {
            text.truncate(stop_index);
            finish_reason = FinishReason::StopSequence;
        }

        Ok(Generation {
            output_tokens: estimated_text_tokens(&text),
            text,
            input_tokens,
            finish_reason,
        })
    }
}

fn estimated_text_tokens(text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }

    let words = text.split_whitespace().count() as u32;
    let character_floor = (text.chars().count() as u32).div_ceil(4);
    words.max(character_floor).max(1)
}

fn truncate_to_estimated_tokens(text: &str, max_tokens: u32) -> (String, bool) {
    if estimated_text_tokens(text) <= max_tokens {
        return (text.to_owned(), false);
    }

    let max_characters = (max_tokens as usize).saturating_mul(4).max(1);
    let truncated: String = text.chars().take(max_characters).collect();
    (truncated, true)
}

fn first_stop_index(text: &str, stop: &[String]) -> Option<usize> {
    stop.iter()
        .filter(|sequence| !sequence.is_empty())
        .filter_map(|sequence| text.find(sequence))
        .min()
}

#[derive(Clone)]
pub struct ServerConfig {
    pub max_output_tokens: u32,
    pub api_key: Option<String>,
    pub max_request_bytes: usize,
}

impl ServerConfig {
    pub fn local() -> Self {
        Self {
            max_output_tokens: 4_096,
            api_key: None,
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
        }
    }
}

#[derive(Clone)]
struct ApiState {
    engine: Arc<dyn InferenceEngine>,
    config: ServerConfig,
    generation_lane: Arc<Semaphore>,
    identifiers: Arc<IdentifierSource>,
}

impl ApiState {
    fn descriptor(&self) -> ModelDescriptor {
        self.engine.descriptor()
    }

    async fn generate(&self, request: GenerationRequest) -> Result<Generation, ApiFailure> {
        let permit = self
            .generation_lane
            .clone()
            .try_acquire_owned()
            .map_err(|_| ApiFailure::busy())?;
        let engine = self.engine.clone();

        task::spawn_blocking(move || {
            let _permit = permit;
            engine.generate(request)
        })
        .await
        .map_err(|error| ApiFailure::internal(format!("inference worker failed: {error}")))?
        .map_err(ApiFailure::from_engine)
    }
}

struct IdentifierSource {
    sequence: AtomicU64,
}

impl IdentifierSource {
    fn next(&self, prefix: &str) -> String {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or_default();
        format!("{prefix}_{timestamp:x}{sequence:08x}")
    }
}

pub fn router(engine: Arc<dyn InferenceEngine>, config: ServerConfig) -> Router {
    let max_request_bytes = config.max_request_bytes;
    let state = ApiState {
        engine,
        config,
        generation_lane: Arc::new(Semaphore::new(1)),
        identifiers: Arc::new(IdentifierSource {
            sequence: AtomicU64::new(1),
        }),
    };

    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(openai_models))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .layer(DefaultBodyLimit::max(max_request_bytes))
        .with_state(state)
}

pub async fn serve(
    bind_address: SocketAddr,
    engine: Arc<dyn InferenceEngine>,
    config: ServerConfig,
) -> Result<(), ApiServerError> {
    let listener = TcpListener::bind(bind_address)
        .await
        .map_err(ApiServerError::Bind)?;
    let local_address = listener.local_addr().map_err(ApiServerError::Bind)?;
    let descriptor = engine.descriptor();
    println!(
        "listening on http://{local_address} for model {} ({})",
        descriptor.id,
        descriptor.execution.label()
    );
    axum::serve(listener, router(engine, config))
        .await
        .map_err(ApiServerError::Serve)
}

#[derive(Debug)]
pub enum ApiServerError {
    Bind(std::io::Error),
    Serve(std::io::Error),
}

impl fmt::Display for ApiServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bind(error) => write!(formatter, "cannot bind HTTP listener: {error}"),
            Self::Serve(error) => write!(formatter, "HTTP server stopped unexpectedly: {error}"),
        }
    }
}

impl Error for ApiServerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Bind(error) | Self::Serve(error) => Some(error),
        }
    }
}

async fn health(State(state): State<ApiState>) -> Json<HealthResponse> {
    let descriptor = state.descriptor();
    Json(HealthResponse {
        status: "ready",
        model: descriptor.id,
        context_tokens: descriptor.context_tokens,
        execution: descriptor.execution.label(),
    })
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    model: String,
    context_tokens: u32,
    execution: &'static str,
}

async fn openai_models(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    if let Err(error) = authorize(&state, &headers, Protocol::OpenAi) {
        return openai_error(error);
    }

    let descriptor = state.descriptor();
    Json(json!({
        "object": "list",
        "data": [{
            "id": descriptor.id,
            "object": "model",
            "created": 0,
            "owned_by": "qwen38-metal"
        }]
    }))
    .into_response()
}

async fn openai_chat_completions(
    State(state): State<ApiState>,
    headers: HeaderMap,
    payload: Result<Json<OpenAiChatRequest>, JsonRejection>,
) -> Response {
    if let Err(error) = authorize(&state, &headers, Protocol::OpenAi) {
        return openai_error(error);
    }

    let payload = match payload {
        Ok(Json(payload)) => payload,
        Err(error) => return openai_error(ApiFailure::invalid_json(error.body_text())),
    };
    let request = match convert_openai_request(&state, payload) {
        Ok(request) => request,
        Err(error) => return openai_error(error),
    };

    let generation = match state.generate(request.generation).await {
        Ok(generation) => generation,
        Err(error) => return openai_error(error),
    };
    let identifier = state.identifiers.next("chatcmpl");
    let created = unix_seconds();

    if request.stream {
        openai_stream(
            identifier,
            created,
            state.descriptor().id,
            generation,
            request.include_usage,
        )
    } else {
        let finish_reason = generation.finish_reason.openai_label();
        let usage = OpenAiUsage::from_generation(&generation);
        Json(OpenAiChatResponse {
            id: identifier,
            object: "chat.completion",
            created,
            model: state.descriptor().id,
            choices: vec![OpenAiChoice {
                index: 0,
                message: OpenAiAssistantMessage {
                    role: "assistant",
                    content: generation.text,
                },
                finish_reason,
            }],
            usage,
        })
        .into_response()
    }
}

fn openai_stream(
    identifier: String,
    created: u64,
    model: String,
    generation: Generation,
    include_usage: bool,
) -> Response {
    let mut events = Vec::new();
    events.push(
        Event::default().data(
            json!({
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": generation.text},
                    "finish_reason": Value::Null
                }]
            })
            .to_string(),
        ),
    );
    events.push(
        Event::default().data(
            json!({
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": generation.finish_reason.openai_label()
                }]
            })
            .to_string(),
        ),
    );
    if include_usage {
        events.push(
            Event::default().data(
                json!({
                    "id": identifier,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": OpenAiUsage::from_generation(&generation)
                })
                .to_string(),
            ),
        );
    }
    events.push(Event::default().data("[DONE]"));

    Sse::new(stream::iter(
        events.into_iter().map(Ok::<Event, Infallible>),
    ))
    .keep_alive(KeepAlive::default())
    .into_response()
}

#[derive(Debug, Deserialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<OpenAiStreamOptions>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stop: Option<StopSequences>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    tools: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    role: String,
    #[serde(default)]
    content: Option<TextContent>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(sequence) => vec![sequence],
            Self::Many(sequences) => sequences,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TextContent {
    String(String),
    Parts(Vec<TextContentPart>),
}

impl TextContent {
    fn into_text(self, parameter: &str, allow_input_text: bool) -> Result<String, ApiFailure> {
        match self {
            Self::String(text) => Ok(text),
            Self::Parts(parts) => {
                let mut text = String::new();
                for (index, part) in parts.into_iter().enumerate() {
                    let allowed =
                        part.kind == "text" || (allow_input_text && part.kind == "input_text");
                    if !allowed {
                        return Err(ApiFailure::bad_request(
                            format!(
                                "unsupported content block type {:?}; text-only serving is enabled",
                                part.kind
                            ),
                            Some(format!("{parameter}[{index}].type")),
                        ));
                    }
                    let part_text = part.text.ok_or_else(|| {
                        ApiFailure::bad_request(
                            "a text content block requires its text field".to_owned(),
                            Some(format!("{parameter}[{index}].text")),
                        )
                    })?;
                    text.push_str(&part_text);
                }
                Ok(text)
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct TextContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

struct ConvertedRequest {
    generation: GenerationRequest,
    stream: bool,
    include_usage: bool,
}

fn convert_openai_request(
    state: &ApiState,
    request: OpenAiChatRequest,
) -> Result<ConvertedRequest, ApiFailure> {
    ensure_model(&state.descriptor(), &request.model)?;
    if request.n.unwrap_or(1) != 1 {
        return Err(ApiFailure::bad_request(
            "only n=1 is supported by the single-generation runtime".to_owned(),
            Some("n".to_owned()),
        ));
    }
    if request.tools.is_some() {
        return Err(ApiFailure::bad_request(
            "tool execution is not implemented for this text-only runtime".to_owned(),
            Some("tools".to_owned()),
        ));
    }

    let max_tokens = resolve_max_tokens(
        request.max_tokens,
        request.max_completion_tokens,
        state.config.max_output_tokens,
        "max_tokens",
    )?;
    validate_sampling(request.temperature, request.top_p)?;
    let messages = convert_openai_messages(request.messages)?;
    let stop = validate_stop_sequences(request.stop.map(StopSequences::into_vec), "stop")?;
    ensure_context_fits(state, &messages, max_tokens)?;

    Ok(ConvertedRequest {
        generation: GenerationRequest {
            messages,
            max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop,
        },
        stream: request.stream,
        include_usage: request
            .stream_options
            .map(|options| options.include_usage)
            .unwrap_or(false),
    })
}

fn convert_openai_messages(messages: Vec<OpenAiMessage>) -> Result<Vec<PromptMessage>, ApiFailure> {
    if messages.is_empty() {
        return Err(ApiFailure::bad_request(
            "at least one message is required".to_owned(),
            Some("messages".to_owned()),
        ));
    }

    let mut result = Vec::with_capacity(messages.len());
    let mut saw_non_system = false;
    let mut saw_user = false;
    for (index, message) in messages.into_iter().enumerate() {
        let role = match message.role.as_str() {
            "system" | "developer" => PromptRole::System,
            "user" => {
                saw_non_system = true;
                saw_user = true;
                PromptRole::User
            }
            "assistant" => {
                saw_non_system = true;
                PromptRole::Assistant
            }
            unsupported => {
                return Err(ApiFailure::bad_request(
                    format!(
                        "unsupported message role {unsupported:?}; supported roles are system, developer, user, and assistant"
                    ),
                    Some(format!("messages[{index}].role")),
                ));
            }
        };
        if role == PromptRole::System && saw_non_system {
            return Err(ApiFailure::bad_request(
                "system or developer messages must precede user and assistant messages".to_owned(),
                Some(format!("messages[{index}].role")),
            ));
        }
        let content = message.content.ok_or_else(|| {
            ApiFailure::bad_request(
                "a text message requires content".to_owned(),
                Some(format!("messages[{index}].content")),
            )
        })?;
        result.push(PromptMessage {
            role,
            content: content.into_text(&format!("messages[{index}].content"), true)?,
        });
    }

    if !saw_user {
        return Err(ApiFailure::bad_request(
            "at least one user message is required".to_owned(),
            Some("messages".to_owned()),
        ));
    }
    Ok(result)
}

#[derive(Serialize)]
struct OpenAiChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChoice>,
    usage: OpenAiUsage,
}

#[derive(Serialize)]
struct OpenAiChoice {
    index: u32,
    message: OpenAiAssistantMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct OpenAiAssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl OpenAiUsage {
    fn from_generation(generation: &Generation) -> Self {
        Self {
            prompt_tokens: generation.input_tokens,
            completion_tokens: generation.output_tokens,
            total_tokens: generation
                .input_tokens
                .saturating_add(generation.output_tokens),
        }
    }
}

async fn anthropic_messages(
    State(state): State<ApiState>,
    headers: HeaderMap,
    payload: Result<Json<AnthropicMessageRequest>, JsonRejection>,
) -> Response {
    if let Err(error) = authorize(&state, &headers, Protocol::Anthropic) {
        return anthropic_error(error);
    }
    if !headers.contains_key("anthropic-version") {
        return anthropic_error(ApiFailure::bad_request(
            "anthropic-version header is required".to_owned(),
            None,
        ));
    }

    let payload = match payload {
        Ok(Json(payload)) => payload,
        Err(error) => return anthropic_error(ApiFailure::invalid_json(error.body_text())),
    };
    let request = match convert_anthropic_request(&state, payload) {
        Ok(request) => request,
        Err(error) => return anthropic_error(error),
    };

    let generation = match state.generate(request.generation).await {
        Ok(generation) => generation,
        Err(error) => return anthropic_error(error),
    };
    let identifier = state.identifiers.next("msg");

    if request.stream {
        anthropic_stream(identifier, state.descriptor().id, generation)
    } else {
        Json(AnthropicResponse::from_generation(
            identifier,
            state.descriptor().id,
            &generation,
        ))
        .into_response()
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    system: Option<TextContent>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    thinking: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: TextContent,
}

fn convert_anthropic_request(
    state: &ApiState,
    request: AnthropicMessageRequest,
) -> Result<ConvertedRequest, ApiFailure> {
    ensure_model(&state.descriptor(), &request.model)?;
    if request.tools.is_some() {
        return Err(ApiFailure::bad_request(
            "tool execution is not implemented for this text-only runtime".to_owned(),
            Some("tools".to_owned()),
        ));
    }
    if request.thinking.is_some() {
        return Err(ApiFailure::bad_request(
            "extended thinking is not implemented for this runtime".to_owned(),
            Some("thinking".to_owned()),
        ));
    }
    if request.max_tokens == 0 || request.max_tokens > state.config.max_output_tokens {
        return Err(ApiFailure::bad_request(
            format!(
                "max_tokens must be between 1 and {}",
                state.config.max_output_tokens
            ),
            Some("max_tokens".to_owned()),
        ));
    }
    validate_sampling(request.temperature, request.top_p)?;

    let mut messages =
        Vec::with_capacity(request.messages.len() + usize::from(request.system.is_some()));
    if let Some(system) = request.system {
        messages.push(PromptMessage {
            role: PromptRole::System,
            content: system.into_text("system", false)?,
        });
    }
    if request.messages.is_empty() {
        return Err(ApiFailure::bad_request(
            "at least one message is required".to_owned(),
            Some("messages".to_owned()),
        ));
    }
    for (index, message) in request.messages.into_iter().enumerate() {
        let role = match message.role.as_str() {
            "user" => PromptRole::User,
            "assistant" => PromptRole::Assistant,
            unsupported => {
                return Err(ApiFailure::bad_request(
                    format!(
                        "unsupported message role {unsupported:?}; supported roles are user and assistant"
                    ),
                    Some(format!("messages[{index}].role")),
                ));
            }
        };
        messages.push(PromptMessage {
            role,
            content: message
                .content
                .into_text(&format!("messages[{index}].content"), false)?,
        });
    }
    if !messages
        .iter()
        .any(|message| message.role == PromptRole::User)
    {
        return Err(ApiFailure::bad_request(
            "at least one user message is required".to_owned(),
            Some("messages".to_owned()),
        ));
    }

    let stop = validate_stop_sequences(request.stop_sequences, "stop_sequences")?;
    ensure_context_fits(state, &messages, request.max_tokens)?;
    Ok(ConvertedRequest {
        generation: GenerationRequest {
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop,
        },
        stream: request.stream,
        include_usage: false,
    })
}

fn anthropic_stream(identifier: String, model: String, generation: Generation) -> Response {
    let start = json!({
        "type": "message_start",
        "message": {
            "id": identifier,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": Value::Null,
            "stop_sequence": Value::Null,
            "usage": {"input_tokens": generation.input_tokens, "output_tokens": 0}
        }
    });
    let content_start = json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""}
    });
    let content_delta = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": generation.text}
    });
    let content_stop = json!({"type": "content_block_stop", "index": 0});
    let message_delta = json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": generation.finish_reason.anthropic_label(),
            "stop_sequence": Value::Null
        },
        "usage": {"output_tokens": generation.output_tokens}
    });
    let message_stop = json!({"type": "message_stop"});
    let events = vec![
        Event::default()
            .event("message_start")
            .data(start.to_string()),
        Event::default()
            .event("content_block_start")
            .data(content_start.to_string()),
        Event::default()
            .event("content_block_delta")
            .data(content_delta.to_string()),
        Event::default()
            .event("content_block_stop")
            .data(content_stop.to_string()),
        Event::default()
            .event("message_delta")
            .data(message_delta.to_string()),
        Event::default()
            .event("message_stop")
            .data(message_stop.to_string()),
    ];

    Sse::new(stream::iter(
        events.into_iter().map(Ok::<Event, Infallible>),
    ))
    .keep_alive(KeepAlive::default())
    .into_response()
}

#[derive(Serialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: &'static str,
    role: &'static str,
    model: String,
    content: Vec<AnthropicTextBlock>,
    stop_reason: &'static str,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

impl AnthropicResponse {
    fn from_generation(identifier: String, model: String, generation: &Generation) -> Self {
        Self {
            id: identifier,
            response_type: "message",
            role: "assistant",
            model,
            content: vec![AnthropicTextBlock {
                block_type: "text",
                text: generation.text.clone(),
            }],
            stop_reason: generation.finish_reason.anthropic_label(),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: generation.input_tokens,
                output_tokens: generation.output_tokens,
            },
        }
    }
}

#[derive(Serialize)]
struct AnthropicTextBlock {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: String,
}

#[derive(Serialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

fn ensure_model(descriptor: &ModelDescriptor, requested: &str) -> Result<(), ApiFailure> {
    if requested == descriptor.id {
        Ok(())
    } else {
        Err(ApiFailure::bad_request(
            format!("model {requested:?} is not loaded"),
            Some("model".to_owned()),
        ))
    }
}

fn resolve_max_tokens(
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    configured_maximum: u32,
    parameter: &str,
) -> Result<u32, ApiFailure> {
    if max_tokens.is_some() && max_completion_tokens.is_some() {
        return Err(ApiFailure::bad_request(
            "max_tokens and max_completion_tokens cannot both be set".to_owned(),
            Some("max_tokens".to_owned()),
        ));
    }
    let value = max_tokens
        .or(max_completion_tokens)
        .unwrap_or(configured_maximum);
    if value == 0 || value > configured_maximum {
        return Err(ApiFailure::bad_request(
            format!("{parameter} must be between 1 and {configured_maximum}"),
            Some(parameter.to_owned()),
        ));
    }
    Ok(value)
}

fn validate_sampling(temperature: Option<f32>, top_p: Option<f32>) -> Result<(), ApiFailure> {
    if let Some(value) = temperature {
        if !(0.0..=2.0).contains(&value) {
            return Err(ApiFailure::bad_request(
                "temperature must be between 0 and 2".to_owned(),
                Some("temperature".to_owned()),
            ));
        }
    }
    if let Some(value) = top_p {
        if !(0.0..=1.0).contains(&value) || value == 0.0 {
            return Err(ApiFailure::bad_request(
                "top_p must be greater than 0 and at most 1".to_owned(),
                Some("top_p".to_owned()),
            ));
        }
    }
    Ok(())
}

fn validate_stop_sequences(
    sequences: Option<Vec<String>>,
    parameter: &str,
) -> Result<Vec<String>, ApiFailure> {
    let sequences = sequences.unwrap_or_default();
    if sequences.len() > 4 {
        return Err(ApiFailure::bad_request(
            "at most four stop sequences are supported".to_owned(),
            Some(parameter.to_owned()),
        ));
    }
    if sequences.iter().any(|sequence| sequence.is_empty()) {
        return Err(ApiFailure::bad_request(
            "stop sequences cannot be empty".to_owned(),
            Some(parameter.to_owned()),
        ));
    }
    Ok(sequences)
}

fn ensure_context_fits(
    state: &ApiState,
    messages: &[PromptMessage],
    max_tokens: u32,
) -> Result<(), ApiFailure> {
    let input_tokens = state
        .engine
        .estimate_prompt_tokens(messages)
        .map_err(ApiFailure::from_engine)?;
    let requested = input_tokens.checked_add(max_tokens).ok_or_else(|| {
        ApiFailure::bad_request("requested token count overflowed u32".to_owned(), None)
    })?;
    let descriptor = state.descriptor();
    if requested > descriptor.context_tokens {
        return Err(ApiFailure::bad_request(
            format!(
                "prompt uses {input_tokens} tokens and max_tokens requests {max_tokens}, exceeding the {}-token context window",
                descriptor.context_tokens
            ),
            Some("max_tokens".to_owned()),
        ));
    }
    Ok(())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[derive(Clone, Copy)]
enum Protocol {
    OpenAi,
    Anthropic,
}

fn authorize(state: &ApiState, headers: &HeaderMap, protocol: Protocol) -> Result<(), ApiFailure> {
    let Some(expected) = state.config.api_key.as_deref() else {
        return Ok(());
    };
    let candidate = match protocol {
        Protocol::OpenAi => bearer_key(headers).or_else(|| header_key(headers, "x-api-key")),
        Protocol::Anthropic => header_key(headers, "x-api-key").or_else(|| bearer_key(headers)),
    };
    let valid = candidate
        .map(|candidate| bool::from(expected.as_bytes().ct_eq(candidate.as_bytes())))
        .unwrap_or(false);
    if valid {
        Ok(())
    } else {
        Err(ApiFailure::unauthorized())
    }
}

fn bearer_key(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .filter(|value| !value.is_empty())
}

fn header_key<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
}

#[derive(Debug)]
struct ApiFailure {
    status: StatusCode,
    message: String,
    parameter: Option<String>,
    kind: FailureKind,
}

#[derive(Debug, Clone, Copy)]
enum FailureKind {
    BadRequest,
    Authentication,
    Busy,
    Unavailable,
    Internal,
}

impl ApiFailure {
    fn bad_request(message: String, parameter: Option<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message,
            parameter,
            kind: FailureKind::BadRequest,
        }
    }

    fn invalid_json(message: String) -> Self {
        Self::bad_request(format!("invalid JSON request body: {message}"), None)
    }

    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: "invalid or missing API key".to_owned(),
            parameter: None,
            kind: FailureKind::Authentication,
        }
    }

    fn busy() -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "the single native generation lane is busy; retry after the active response completes"
                .to_owned(),
            parameter: None,
            kind: FailureKind::Busy,
        }
    }

    fn internal(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message,
            parameter: None,
            kind: FailureKind::Internal,
        }
    }

    fn from_engine(error: EngineError) -> Self {
        match error {
            EngineError::ContextLimit { requested, maximum } => Self::bad_request(
                format!("requested {requested} tokens but the model supports at most {maximum}"),
                Some("max_tokens".to_owned()),
            ),
            EngineError::Unavailable(message) => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message,
                parameter: None,
                kind: FailureKind::Unavailable,
            },
            EngineError::Failure(message) => Self::internal(message),
        }
    }
}

fn openai_error(error: ApiFailure) -> Response {
    let error_type = match error.kind {
        FailureKind::BadRequest => "invalid_request_error",
        FailureKind::Authentication => "invalid_api_key",
        FailureKind::Busy => "rate_limit_error",
        FailureKind::Unavailable | FailureKind::Internal => "server_error",
    };
    let code = match error.kind {
        FailureKind::Authentication => Some("invalid_api_key"),
        FailureKind::Busy => Some("generation_busy"),
        FailureKind::Unavailable => Some("model_unavailable"),
        FailureKind::BadRequest | FailureKind::Internal => None,
    };
    let mut response = (
        error.status,
        Json(json!({
            "error": {
                "message": error.message,
                "type": error_type,
                "param": error.parameter,
                "code": code
            }
        })),
    )
        .into_response();
    add_failure_headers(&mut response, error.kind);
    response
}

fn anthropic_error(error: ApiFailure) -> Response {
    let error_type = match error.kind {
        FailureKind::BadRequest => "invalid_request_error",
        FailureKind::Authentication => "authentication_error",
        FailureKind::Busy => "rate_limit_error",
        FailureKind::Unavailable => "api_error",
        FailureKind::Internal => "api_error",
    };
    let mut response = (
        error.status,
        Json(json!({
            "type": "error",
            "error": {"type": error_type, "message": error.message}
        })),
    )
        .into_response();
    add_failure_headers(&mut response, error.kind);
    response
}

fn add_failure_headers(response: &mut Response, kind: FailureKind) {
    match kind {
        FailureKind::Authentication => {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        FailureKind::Busy => {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        FailureKind::BadRequest | FailureKind::Unavailable | FailureKind::Internal => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    fn test_router(api_key: Option<&str>) -> Router {
        let engine = Arc::new(FixtureEngine::new(
            "qwen3.8-27b",
            64,
            "native fixture answer",
        ));
        let mut config = ServerConfig::local();
        config.max_output_tokens = 32;
        config.api_key = api_key.map(str::to_owned);
        router(engine, config)
    }

    async fn response_text(response: Response) -> String {
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        String::from_utf8(body.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn lists_the_loaded_model() {
        let response = test_router(None)
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_text(response).await;
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["data"][0]["id"],
            "qwen3.8-27b"
        );
    }

    #[tokio::test]
    async fn openai_sync_response_uses_the_chat_contract() {
        let body = json!({
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 16
        })
        .to_string();
        let response = test_router(None)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let response: Value = serde_json::from_str(&response_text(response).await).unwrap();
        assert_eq!(response["object"], "chat.completion");
        assert_eq!(
            response["choices"][0]["message"]["content"],
            "native fixture answer"
        );
        assert!(response["usage"]["total_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn openai_stream_response_finishes_with_done() {
        let body = json!({
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 16
        })
        .to_string();
        let response = test_router(None)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = response_text(response).await;
        assert!(body.contains("chat.completion.chunk"));
        assert!(body.contains("data: [DONE]"));
        assert!(body.contains("\"usage\""));
    }

    #[tokio::test]
    async fn anthropic_sync_response_uses_the_messages_contract() {
        let body = json!({
            "model": "qwen3.8-27b",
            "system": "You are concise.",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 16
        })
        .to_string();
        let response = test_router(None)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let response: Value = serde_json::from_str(&response_text(response).await).unwrap();
        assert_eq!(response["type"], "message");
        assert_eq!(response["content"][0]["text"], "native fixture answer");
        assert_eq!(response["stop_reason"], "end_turn");
    }

    #[tokio::test]
    async fn anthropic_stream_uses_named_sse_events() {
        let body = json!({
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 16,
            "stream": true
        })
        .to_string();
        let response = test_router(None)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("event: message_start"));
        assert!(body.contains("event: content_block_delta"));
        assert!(body.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn configured_key_protects_openai_and_anthropic_routes() {
        let openai = test_router(Some("test-key"))
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(openai.status(), StatusCode::UNAUTHORIZED);

        let anthropic = test_router(Some("test-key"))
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .header("x-api-key", "test-key")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_tokens": 8
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(anthropic.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rejects_vision_and_context_overflow_without_calling_the_engine() {
        let vision = test_router(None)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [{"role": "user", "content": [{"type":"image_url", "image_url": {"url":"x"}}]}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(vision.status(), StatusCode::BAD_REQUEST);

        let context = test_router(None)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [{"role": "user", "content": "x".repeat(260)}],
                            "max_tokens": 32
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(context.status(), StatusCode::BAD_REQUEST);
    }
}
