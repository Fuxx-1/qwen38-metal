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
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    error::Error,
    fmt,
    io::{BufReader, Cursor},
    net::{IpAddr, SocketAddr},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use subtle::ConstantTimeEq;
use tokio::{net::TcpListener, sync::Semaphore, task};

const DEFAULT_MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_REMOTE_IMAGE_BYTES: usize = 16 * 1024 * 1024;

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
    Tool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptMessage {
    pub role: PromptRole,
    pub content: Vec<PromptPart>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl PromptMessage {
    pub fn text(role: PromptRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![PromptPart::Text(content.into())],
            reasoning_content: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                PromptPart::Text(text) => Some(text.as_str()),
                PromptPart::Image(_) => None,
            })
            .collect()
    }

    pub fn image_count(&self) -> usize {
        self.content
            .iter()
            .filter(|part| matches!(part, PromptPart::Image(_)))
            .count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptPart {
    Text(String),
    Image(InputImage),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputImage {
    pub media_type: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Specific(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingConfig {
    pub enabled: bool,
    pub budget_tokens: Option<u32>,
}

impl ThinkingConfig {
    pub const DISABLED: Self = Self {
        enabled: false,
        budget_tokens: None,
    };

    pub const ENABLED: Self = Self {
        enabled: true,
        budget_tokens: None,
    };
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub messages: Vec<PromptMessage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub thinking: ThinkingConfig,
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
    ToolCalls,
}

impl FinishReason {
    fn openai_label(self) -> &'static str {
        match self {
            Self::Stop | Self::StopSequence => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
        }
    }

    fn anthropic_label(self) -> &'static str {
        match self {
            Self::Stop => "end_turn",
            Self::Length => "max_tokens",
            Self::StopSequence => "stop_sequence",
            Self::ToolCalls => "tool_use",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub finish_reason: FinishReason,
}

pub trait InferenceEngine: Send + Sync + 'static {
    fn descriptor(&self) -> ModelDescriptor;

    fn estimate_prompt_tokens(&self, request: &GenerationRequest) -> Result<u32, EngineError>;

    fn generate(&self, request: GenerationRequest) -> Result<Generation, EngineError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    ContextLimit { requested: u32, maximum: u32 },
    InvalidRequest(String),
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
            Self::InvalidRequest(message) | Self::Unavailable(message) | Self::Failure(message) => {
                formatter.write_str(message)
            }
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

    fn estimate_prompt_tokens(&self, request: &GenerationRequest) -> Result<u32, EngineError> {
        let token_count = request.messages.iter().try_fold(0_u32, |total, message| {
            let content_tokens = message.content.iter().try_fold(0_u32, |sum, part| {
                let tokens = match part {
                    PromptPart::Text(text) => estimated_text_tokens(text),
                    // Fixture mode has no image encoder. Reserve a conservative visual span so
                    // protocol validation still exercises context accounting.
                    PromptPart::Image(_) => 256,
                };
                sum.checked_add(tokens)
            });
            total
                .checked_add(4)
                .and_then(|value| content_tokens.and_then(|tokens| value.checked_add(tokens)))
        });
        let tool_tokens = request.tools.iter().try_fold(0_u32, |total, tool| {
            let encoded = serde_json::to_string(&tool.input_schema).map_err(|error| {
                EngineError::Failure(format!("cannot encode tool schema: {error}"))
            })?;
            total
                .checked_add(estimated_text_tokens(&tool.name))
                .and_then(|value| {
                    value.checked_add(estimated_text_tokens(
                        tool.description.as_deref().unwrap_or_default(),
                    ))
                })
                .and_then(|value| value.checked_add(estimated_text_tokens(&encoded)))
                .ok_or_else(|| EngineError::Failure("tool token count overflowed u32".to_owned()))
        });
        let token_count = token_count
            .ok_or_else(|| EngineError::Failure("prompt token count overflowed u32".to_owned()))?;
        let tool_tokens = tool_tokens?;
        token_count
            .checked_add(tool_tokens)
            .and_then(|value| {
                if request.thinking.enabled {
                    value.checked_add(8)
                } else {
                    Some(value)
                }
            })
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| EngineError::Failure("prompt token count overflowed u32".to_owned()))
    }

    fn generate(&self, request: GenerationRequest) -> Result<Generation, EngineError> {
        let input_tokens = self.estimate_prompt_tokens(&request)?;
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

        let parts = parse_model_output(&text, request.thinking, &request.tools)?;
        if !parts.tool_calls.is_empty() {
            finish_reason = FinishReason::ToolCalls;
        }

        Ok(Generation {
            output_tokens: estimated_text_tokens(&text),
            text: parts.text,
            reasoning: parts.reasoning,
            tool_calls: parts.tool_calls,
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

#[derive(Debug)]
pub(crate) struct ParsedModelOutput {
    pub(crate) text: String,
    pub(crate) reasoning: Option<String>,
    pub(crate) tool_calls: Vec<ToolCall>,
}

/// Decode the textual protocol emitted by Qwen's bundled chat template. Tool
/// execution deliberately remains outside the server: this only describes a
/// requested call to the API client.
pub(crate) fn parse_model_output(
    raw: &str,
    thinking: ThinkingConfig,
    tools: &[ToolDefinition],
) -> Result<ParsedModelOutput, EngineError> {
    let (reasoning, answer) = if thinking.enabled {
        split_reasoning(raw)
    } else {
        (None, raw.to_owned())
    };
    let (text, tool_calls) = parse_tool_calls(&answer, tools)?;
    Ok(ParsedModelOutput {
        text,
        reasoning,
        tool_calls,
    })
}

fn split_reasoning(raw: &str) -> (Option<String>, String) {
    let raw = raw.strip_prefix("<think>\n").unwrap_or(raw);
    let raw = raw.strip_prefix("<think>").unwrap_or(raw);
    let Some(end) = raw.find("</think>") else {
        let reasoning = raw.trim();
        return (
            (!reasoning.is_empty()).then(|| reasoning.to_owned()),
            String::new(),
        );
    };
    let reasoning = raw[..end].trim();
    let answer = raw[end + "</think>".len()..]
        .strip_prefix("\n\n")
        .or_else(|| raw[end + "</think>".len()..].strip_prefix('\n'))
        .unwrap_or(&raw[end + "</think>".len()..])
        .to_owned();
    (
        (!reasoning.is_empty()).then(|| reasoning.to_owned()),
        answer,
    )
}

fn parse_tool_calls(
    answer: &str,
    tools: &[ToolDefinition],
) -> Result<(String, Vec<ToolCall>), EngineError> {
    if tools.is_empty() {
        return Ok((answer.to_owned(), Vec::new()));
    }

    let mut remaining = answer;
    let mut text = String::new();
    let mut calls = Vec::new();
    while let Some(start) = remaining.find("<tool_call>") {
        text.push_str(&remaining[..start]);
        let body_start = start + "<tool_call>".len();
        let Some(end_relative) = remaining[body_start..].find("</tool_call>") else {
            text.push_str(&remaining[start..]);
            break;
        };
        let body_end = body_start + end_relative;
        let body = &remaining[body_start..body_end];
        match parse_tool_call(body, tools)? {
            Some(call) => calls.push(call),
            None => text.push_str(&remaining[start..body_end + "</tool_call>".len()]),
        }
        remaining = &remaining[body_end + "</tool_call>".len()..];
    }
    text.push_str(remaining);
    Ok((text.trim().to_owned(), calls))
}

fn parse_tool_call(body: &str, tools: &[ToolDefinition]) -> Result<Option<ToolCall>, EngineError> {
    let body = body.trim();
    let Some(function_start) = body.find("<function=") else {
        return Ok(None);
    };
    let name_start = function_start + "<function=".len();
    let Some(name_end_relative) = body[name_start..].find('>') else {
        return Ok(None);
    };
    let name_end = name_start + name_end_relative;
    let name = body[name_start..name_end].trim();
    if name.is_empty() || !tools.iter().any(|tool| tool.name == name) {
        return Ok(None);
    }
    let parameters = &body[name_end + 1..];
    let Some(function_end) = parameters.find("</function>") else {
        return Ok(None);
    };
    let parameters = &parameters[..function_end];
    let mut values = serde_json::Map::new();
    let mut remaining = parameters;
    while let Some(start) = remaining.find("<parameter=") {
        let name_start = start + "<parameter=".len();
        let Some(name_end_relative) = remaining[name_start..].find('>') else {
            return Ok(None);
        };
        let name_end = name_start + name_end_relative;
        let parameter_name = remaining[name_start..name_end].trim();
        if parameter_name.is_empty() {
            return Ok(None);
        }
        let value_start = name_end + 1;
        let Some(value_end_relative) = remaining[value_start..].find("</parameter>") else {
            return Ok(None);
        };
        let value_end = value_start + value_end_relative;
        let value = remaining[value_start..value_end].trim();
        let value = serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()));
        values.insert(parameter_name.to_owned(), value);
        remaining = &remaining[value_end + "</parameter>".len()..];
    }

    Ok(Some(ToolCall {
        id: String::new(),
        name: name.to_owned(),
        arguments: Value::Object(values),
    }))
}

#[derive(Clone)]
pub struct ServerConfig {
    pub max_output_tokens: u32,
    pub api_key: Option<String>,
    pub max_request_bytes: usize,
    /// The number of independent model executions allowed at once. Native
    /// Qwen defaults to one until a batched scheduler is available.
    pub generation_concurrency: usize,
    /// Includes currently executing requests. Requests within this bound wait
    /// fairly for a generation lane instead of receiving a transient 429.
    pub max_queued_requests: usize,
}

impl ServerConfig {
    pub fn local() -> Self {
        Self {
            max_output_tokens: 4_096,
            api_key: None,
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
            generation_concurrency: 1,
            max_queued_requests: 64,
        }
    }
}

#[derive(Clone)]
struct ApiState {
    engine: Arc<dyn InferenceEngine>,
    config: ServerConfig,
    image_client: reqwest::Client,
    generation_lanes: Arc<Semaphore>,
    queue_slots: Arc<Semaphore>,
    identifiers: Arc<IdentifierSource>,
}

/// The image downloader resolves every hostname itself and drops local,
/// link-local, multicast, and private answers before Hyper opens a socket.
/// Keeping the validation inside Reqwest prevents a domain from passing a
/// one-time lookup and then rebinding to localhost for the actual request.
#[derive(Clone)]
struct SafeImageResolver;

impl reqwest::dns::Resolve for SafeImageResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_owned();
        Box::pin(async move {
            let addresses = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .map_err(|error| -> Box<dyn Error + Send + Sync> { Box::new(error) })?
                .filter(|address| !disallowed_image_address(address.ip()))
                .collect::<Vec<_>>();
            if addresses.is_empty() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "image host resolved only to blocked network addresses",
                )) as Box<dyn Error + Send + Sync>);
            }
            Ok(Box::new(addresses.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

impl ApiState {
    fn descriptor(&self) -> ModelDescriptor {
        self.engine.descriptor()
    }

    async fn generate(&self, request: GenerationRequest) -> Result<Generation, ApiFailure> {
        let queue_slot = self
            .queue_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| ApiFailure::busy())?;
        let permit = self
            .generation_lanes
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ApiFailure::unavailable("generation scheduler stopped".to_owned()))?;
        let engine = self.engine.clone();

        task::spawn_blocking(move || {
            let _queue_slot = queue_slot;
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

fn assign_tool_call_ids(generation: &mut Generation, response_id: &str) {
    if generation.tool_calls.is_empty() {
        return;
    }
    for (index, call) in generation.tool_calls.iter_mut().enumerate() {
        if call.id.is_empty() {
            call.id = format!("call_{response_id}_{index}");
        }
    }
    generation.finish_reason = FinishReason::ToolCalls;
}

pub fn router(engine: Arc<dyn InferenceEngine>, config: ServerConfig) -> Router {
    let max_request_bytes = config.max_request_bytes;
    let generation_concurrency = config.generation_concurrency.max(1);
    let max_queued_requests = config.max_queued_requests.max(generation_concurrency);
    let state = ApiState {
        engine,
        config,
        image_client: reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(3))
            .dns_resolver(Arc::new(SafeImageResolver))
            .build()
            .expect("the built-in HTTP client configuration is valid"),
        generation_lanes: Arc::new(Semaphore::new(generation_concurrency)),
        queue_slots: Arc::new(Semaphore::new(max_queued_requests)),
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
    let request = match convert_openai_request(&state, payload).await {
        Ok(request) => request,
        Err(error) => return openai_error(error),
    };

    let mut generation = match state.generate(request.generation).await {
        Ok(generation) => generation,
        Err(error) => return openai_error(error),
    };
    let identifier = state.identifiers.next("chatcmpl");
    let created = unix_seconds();
    assign_tool_call_ids(&mut generation, &identifier);

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
                message: OpenAiAssistantMessage::from_generation(&generation),
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
                "id": &identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": &model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": Value::Null
                }]
            })
            .to_string(),
        ),
    );
    if let Some(reasoning) = generation
        .reasoning
        .as_deref()
        .filter(|text| !text.is_empty())
    {
        events.push(
            Event::default().data(
                json!({
                    "id": &identifier,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": &model,
                    "choices": [{
                        "index": 0,
                        "delta": {"reasoning_content": reasoning},
                        "finish_reason": Value::Null
                    }]
                })
                .to_string(),
            ),
        );
    }
    if !generation.text.is_empty() {
        events.push(
            Event::default().data(
                json!({
                    "id": &identifier,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": &model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": &generation.text},
                        "finish_reason": Value::Null
                    }]
                })
                .to_string(),
            ),
        );
    }
    for (index, call) in generation.tool_calls.iter().enumerate() {
        let arguments = serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_owned());
        events.push(
            Event::default().data(
                json!({
                    "id": &identifier,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": &model,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": [{
                            "index": index,
                            "id": &call.id,
                            "type": "function",
                            "function": {"name": &call.name, "arguments": arguments}
                        }]},
                        "finish_reason": Value::Null
                    }]
                })
                .to_string(),
            ),
        );
    }
    events.push(
        Event::default().data(
            json!({
                "id": &identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": &model,
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
    tools: Option<Vec<OpenAiTool>>,
    #[serde(default)]
    tool_choice: Option<Value>,
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
    content: Option<OpenAiContent>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: String,
    function: OpenAiFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAiFunction {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type", default)]
    kind: Option<String>,
    function: OpenAiToolFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolFunction {
    name: String,
    arguments: String,
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
enum OpenAiContent {
    String(String),
    Parts(Vec<OpenAiContentPart>),
}

impl OpenAiContent {
    async fn into_parts(
        self,
        client: &reqwest::Client,
        parameter: &str,
        allow_images: bool,
    ) -> Result<Vec<PromptPart>, ApiFailure> {
        match self {
            Self::String(text) => Ok(vec![PromptPart::Text(text)]),
            Self::Parts(parts) => {
                let mut content = Vec::with_capacity(parts.len());
                for (index, part) in parts.into_iter().enumerate() {
                    match part.kind.as_str() {
                        "text" | "input_text" => {
                            let text = part.text.ok_or_else(|| {
                                ApiFailure::bad_request(
                                    "a text content block requires its text field".to_owned(),
                                    Some(format!("{parameter}[{index}].text")),
                                )
                            })?;
                            content.push(PromptPart::Text(text));
                        }
                        "image_url" if allow_images => {
                            let image_url = part.image_url.ok_or_else(|| {
                                ApiFailure::bad_request(
                                    "an image_url content block requires its image_url field"
                                        .to_owned(),
                                    Some(format!("{parameter}[{index}].image_url")),
                                )
                            })?;
                            content.push(PromptPart::Image(
                                load_openai_image(
                                    client,
                                    image_url.url(),
                                    &format!("{parameter}[{index}].image_url"),
                                )
                                .await?,
                            ));
                        }
                        "image_url" => {
                            return Err(ApiFailure::bad_request(
                                "images are allowed only in user messages".to_owned(),
                                Some(format!("{parameter}[{index}].type")),
                            ));
                        }
                        unsupported => {
                            return Err(ApiFailure::bad_request(
                                format!("unsupported OpenAI content block type {unsupported:?}"),
                                Some(format!("{parameter}[{index}].type")),
                            ));
                        }
                    }
                }
                Ok(content)
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image_url: Option<OpenAiImageUrl>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiImageUrl {
    Object { url: String },
    String(String),
}

impl OpenAiImageUrl {
    fn url(self) -> String {
        match self {
            Self::Object { url } | Self::String(url) => url,
        }
    }
}

struct ConvertedRequest {
    generation: GenerationRequest,
    stream: bool,
    include_usage: bool,
}

async fn convert_openai_request(
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
    let max_tokens = resolve_max_tokens(
        request.max_tokens,
        request.max_completion_tokens,
        state.config.max_output_tokens,
        "max_tokens",
    )?;
    validate_sampling(request.temperature, request.top_p)?;
    let mut tools = convert_openai_tools(request.tools)?;
    let tool_choice = parse_openai_tool_choice(request.tool_choice, &tools)?;
    if tool_choice == ToolChoice::None {
        tools.clear();
    } else if let ToolChoice::Specific(name) = &tool_choice {
        tools.retain(|tool| tool.name == *name);
    }
    let messages = convert_openai_messages(&state.image_client, request.messages).await?;
    let stop = validate_stop_sequences(request.stop.map(StopSequences::into_vec), "stop")?;
    let generation = GenerationRequest {
        messages,
        tools,
        tool_choice,
        thinking: ThinkingConfig::DISABLED,
        max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        stop,
    };
    ensure_context_fits(state, &generation)?;

    Ok(ConvertedRequest {
        generation,
        stream: request.stream,
        include_usage: request
            .stream_options
            .map(|options| options.include_usage)
            .unwrap_or(false),
    })
}

async fn convert_openai_messages(
    client: &reqwest::Client,
    messages: Vec<OpenAiMessage>,
) -> Result<Vec<PromptMessage>, ApiFailure> {
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
            "tool" => {
                saw_non_system = true;
                PromptRole::Tool
            }
            unsupported => {
                return Err(ApiFailure::bad_request(
                    format!(
                        "unsupported message role {unsupported:?}; supported roles are system, developer, user, assistant, and tool"
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
        if role == PromptRole::Tool
            && message
                .tool_call_id
                .as_deref()
                .unwrap_or_default()
                .is_empty()
        {
            return Err(ApiFailure::bad_request(
                "a tool message requires tool_call_id".to_owned(),
                Some(format!("messages[{index}].tool_call_id")),
            ));
        }
        let content = match message.content {
            Some(content) => {
                content
                    .into_parts(
                        client,
                        &format!("messages[{index}].content"),
                        role == PromptRole::User,
                    )
                    .await?
            }
            None if role == PromptRole::Assistant && message.tool_calls.is_some() => Vec::new(),
            None => {
                return Err(ApiFailure::bad_request(
                    "a message requires content unless it is an assistant tool-call message"
                        .to_owned(),
                    Some(format!("messages[{index}].content")),
                ));
            }
        };
        let tool_calls = convert_openai_tool_calls(
            message.tool_calls.unwrap_or_default(),
            &format!("messages[{index}].tool_calls"),
        )?;
        result.push(PromptMessage {
            role,
            content,
            reasoning_content: message.reasoning_content,
            tool_calls,
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

fn convert_openai_tools(tools: Option<Vec<OpenAiTool>>) -> Result<Vec<ToolDefinition>, ApiFailure> {
    let tools = tools.unwrap_or_default();
    let mut converted = Vec::with_capacity(tools.len());
    for (index, tool) in tools.into_iter().enumerate() {
        if tool.kind != "function" {
            return Err(ApiFailure::bad_request(
                "only function tools are supported".to_owned(),
                Some(format!("tools[{index}].type")),
            ));
        }
        validate_tool_name(
            &tool.function.name,
            &format!("tools[{index}].function.name"),
        )?;
        let input_schema = tool
            .function
            .parameters
            .unwrap_or_else(|| json!({"type": "object"}));
        if !input_schema.is_object() {
            return Err(ApiFailure::bad_request(
                "function parameters must be a JSON Schema object".to_owned(),
                Some(format!("tools[{index}].function.parameters")),
            ));
        }
        if converted
            .iter()
            .any(|existing: &ToolDefinition| existing.name == tool.function.name)
        {
            return Err(ApiFailure::bad_request(
                "tool function names must be unique".to_owned(),
                Some(format!("tools[{index}].function.name")),
            ));
        }
        converted.push(ToolDefinition {
            name: tool.function.name,
            description: tool.function.description,
            input_schema,
        });
    }
    Ok(converted)
}

fn parse_openai_tool_choice(
    tool_choice: Option<Value>,
    tools: &[ToolDefinition],
) -> Result<ToolChoice, ApiFailure> {
    let Some(choice) = tool_choice else {
        return Ok(ToolChoice::Auto);
    };
    let choice = match choice {
        Value::String(value) if value == "auto" => ToolChoice::Auto,
        Value::String(value) if value == "none" => ToolChoice::None,
        Value::String(value) if value == "required" => ToolChoice::Required,
        Value::Object(value) => {
            let name = value
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ApiFailure::bad_request(
                        "a function tool_choice requires function.name".to_owned(),
                        Some("tool_choice".to_owned()),
                    )
                })?;
            ToolChoice::Specific(name.to_owned())
        }
        _ => {
            return Err(ApiFailure::bad_request(
                "tool_choice must be auto, none, required, or a function selection".to_owned(),
                Some("tool_choice".to_owned()),
            ));
        }
    };
    if let ToolChoice::Specific(name) = &choice {
        if !tools.iter().any(|tool| tool.name == *name) {
            return Err(ApiFailure::bad_request(
                format!("tool_choice selects undeclared function {name:?}"),
                Some("tool_choice".to_owned()),
            ));
        }
    }
    if !matches!(choice, ToolChoice::None) && tools.is_empty() {
        return Err(ApiFailure::bad_request(
            "tool_choice requires at least one declared tool".to_owned(),
            Some("tool_choice".to_owned()),
        ));
    }
    Ok(choice)
}

fn convert_openai_tool_calls(
    calls: Vec<OpenAiToolCall>,
    parameter: &str,
) -> Result<Vec<ToolCall>, ApiFailure> {
    calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| {
            if call.kind.as_deref().is_some_and(|kind| kind != "function") {
                return Err(ApiFailure::bad_request(
                    "only function tool calls are supported".to_owned(),
                    Some(format!("{parameter}[{index}].type")),
                ));
            }
            validate_tool_name(
                &call.function.name,
                &format!("{parameter}[{index}].function.name"),
            )?;
            let arguments: Value =
                serde_json::from_str(&call.function.arguments).map_err(|_| {
                    ApiFailure::bad_request(
                        "tool call function.arguments must be valid JSON".to_owned(),
                        Some(format!("{parameter}[{index}].function.arguments")),
                    )
                })?;
            if !arguments.is_object() {
                return Err(ApiFailure::bad_request(
                    "tool call function.arguments must be a JSON object".to_owned(),
                    Some(format!("{parameter}[{index}].function.arguments")),
                ));
            }
            Ok(ToolCall {
                id: call.id.unwrap_or_default(),
                name: call.function.name,
                arguments,
            })
        })
        .collect()
}

fn validate_tool_name(name: &str, parameter: &str) -> Result<(), ApiFailure> {
    let valid = !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'));
    if valid {
        Ok(())
    } else {
        Err(ApiFailure::bad_request(
            "tool names must be 1-64 ASCII letters, digits, _, -, or .".to_owned(),
            Some(parameter.to_owned()),
        ))
    }
}

async fn load_openai_image(
    client: &reqwest::Client,
    url: String,
    parameter: &str,
) -> Result<InputImage, ApiFailure> {
    if let Some(data) = url.strip_prefix("data:") {
        return decode_data_image(data, parameter);
    }
    let url = reqwest::Url::parse(&url).map_err(|_| {
        ApiFailure::bad_request(
            "image_url.url must be a data URI or an absolute http(s) URL".to_owned(),
            Some(parameter.to_owned()),
        )
    })?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(ApiFailure::bad_request(
            "image_url.url must be a data URI or an absolute http(s) URL".to_owned(),
            Some(parameter.to_owned()),
        ));
    }
    if url.username() != "" || url.password().is_some() || url.host_str() == Some("localhost") {
        return Err(ApiFailure::bad_request(
            "image_url.url cannot target a local or credentialed URL".to_owned(),
            Some(parameter.to_owned()),
        ));
    }
    if let Some(host) = url.host_str() {
        if let Ok(address) = host.parse::<IpAddr>() {
            if disallowed_image_address(address) {
                return Err(ApiFailure::bad_request(
                    "image_url.url cannot target a local network address".to_owned(),
                    Some(parameter.to_owned()),
                ));
            }
        }
    }

    let mut response = client.get(url).send().await.map_err(|_| {
        ApiFailure::bad_request(
            "could not download image_url.url".to_owned(),
            Some(parameter.to_owned()),
        )
    })?;
    if !response.status().is_success() {
        return Err(ApiFailure::bad_request(
            "image_url.url returned a non-success response".to_owned(),
            Some(parameter.to_owned()),
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_REMOTE_IMAGE_BYTES as u64)
    {
        return Err(ApiFailure::bad_request(
            format!("image_url.url exceeds the {MAX_REMOTE_IMAGE_BYTES}-byte image limit"),
            Some(parameter.to_owned()),
        ));
    }
    let media_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .filter(|value| value.starts_with("image/"))
        .unwrap_or("application/octet-stream")
        .to_owned();
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| {
        ApiFailure::bad_request(
            "could not download image_url.url".to_owned(),
            Some(parameter.to_owned()),
        )
    })? {
        if bytes.len().saturating_add(chunk.len()) > MAX_REMOTE_IMAGE_BYTES {
            return Err(ApiFailure::bad_request(
                format!("image_url.url exceeds the {MAX_REMOTE_IMAGE_BYTES}-byte image limit"),
                Some(parameter.to_owned()),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.is_empty() {
        return Err(ApiFailure::bad_request(
            "image_url.url returned an empty image".to_owned(),
            Some(parameter.to_owned()),
        ));
    }
    validate_input_image(InputImage { media_type, bytes }, parameter)
}

fn decode_data_image(data: &str, parameter: &str) -> Result<InputImage, ApiFailure> {
    let (header, payload) = data.split_once(',').ok_or_else(|| {
        ApiFailure::bad_request(
            "image data URI must contain a comma before the payload".to_owned(),
            Some(parameter.to_owned()),
        )
    })?;
    let media_type = header.split(';').next().unwrap_or_default();
    if !media_type.starts_with("image/") || !header.split(';').any(|part| part == "base64") {
        return Err(ApiFailure::bad_request(
            "image data URI must use an image/* media type and base64 encoding".to_owned(),
            Some(parameter.to_owned()),
        ));
    }
    let bytes = BASE64.decode(payload).map_err(|_| {
        ApiFailure::bad_request(
            "image data URI contains invalid base64".to_owned(),
            Some(parameter.to_owned()),
        )
    })?;
    if bytes.is_empty() || bytes.len() > MAX_REMOTE_IMAGE_BYTES {
        return Err(ApiFailure::bad_request(
            format!("image data must be between 1 and {MAX_REMOTE_IMAGE_BYTES} bytes"),
            Some(parameter.to_owned()),
        ));
    }
    validate_input_image(
        InputImage {
            media_type: media_type.to_owned(),
            bytes,
        },
        parameter,
    )
}

fn validate_input_image(input: InputImage, parameter: &str) -> Result<InputImage, ApiFailure> {
    let reader = BufReader::new(Cursor::new(input.bytes.as_slice()));
    let mut reader = image::ImageReader::new(reader)
        .with_guessed_format()
        .map_err(|_| {
            ApiFailure::bad_request(
                "image data cannot be read".to_owned(),
                Some(parameter.to_owned()),
            )
        })?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16_384);
    limits.max_image_height = Some(16_384);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    reader.decode().map_err(|_| {
        ApiFailure::bad_request(
            "image data is not a supported, decodable image".to_owned(),
            Some(parameter.to_owned()),
        )
    })?;
    Ok(input)
}

fn disallowed_image_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => {
            let octets = address.octets();
            address.is_private()
                || address.is_loopback()
                || address.is_link_local()
                || address.is_broadcast()
                || address.is_unspecified()
                || octets[0] == 0
                || (octets[0] == 100 && (64..=127).contains(&octets[1]))
                || (octets[0] >= 224)
        }
        IpAddr::V6(address) => {
            address.is_loopback()
                || address.is_unspecified()
                || address.is_unique_local()
                || address.is_unicast_link_local()
                || address.is_multicast()
        }
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAiResponseToolCall>,
}

impl OpenAiAssistantMessage {
    fn from_generation(generation: &Generation) -> Self {
        Self {
            role: "assistant",
            content: (!generation.text.is_empty()).then(|| generation.text.clone()),
            reasoning_content: generation.reasoning.clone(),
            tool_calls: generation
                .tool_calls
                .iter()
                .map(OpenAiResponseToolCall::from_tool_call)
                .collect(),
        }
    }
}

#[derive(Serialize)]
struct OpenAiResponseToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiResponseToolFunction,
}

impl OpenAiResponseToolCall {
    fn from_tool_call(call: &ToolCall) -> Self {
        Self {
            id: call.id.clone(),
            kind: "function",
            function: OpenAiResponseToolFunction {
                name: call.name.clone(),
                arguments: serde_json::to_string(&call.arguments)
                    .unwrap_or_else(|_| "{}".to_owned()),
            },
        }
    }
}

#[derive(Serialize)]
struct OpenAiResponseToolFunction {
    name: String,
    arguments: String,
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
    let request = match convert_anthropic_request(&state, payload).await {
        Ok(request) => request,
        Err(error) => return anthropic_error(error),
    };

    let mut generation = match state.generate(request.generation).await {
        Ok(generation) => generation,
        Err(error) => return anthropic_error(error),
    };
    let identifier = state.identifiers.next("msg");
    assign_tool_call_ids(&mut generation, &identifier);

    if request.stream {
        anthropic_stream(identifier, state.descriptor().id, generation)
    } else {
        Json(anthropic_response(
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
    system: Option<AnthropicContent>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    thinking: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    String(String),
    Parts(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<Value>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    tool_use_id: Option<String>,
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    source: Option<AnthropicImageSource>,
}

#[derive(Debug, Deserialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    media_type: Option<String>,
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_schema: Option<Value>,
}

async fn convert_anthropic_request(
    state: &ApiState,
    request: AnthropicMessageRequest,
) -> Result<ConvertedRequest, ApiFailure> {
    ensure_model(&state.descriptor(), &request.model)?;
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

    let mut tools = convert_anthropic_tools(request.tools)?;
    let tool_choice = parse_anthropic_tool_choice(request.tool_choice, &tools)?;
    if tool_choice == ToolChoice::None {
        tools.clear();
    } else if let ToolChoice::Specific(name) = &tool_choice {
        tools.retain(|tool| tool.name == *name);
    }
    let thinking = parse_anthropic_thinking(request.thinking, request.max_tokens)?;

    let mut messages =
        Vec::with_capacity(request.messages.len() + usize::from(request.system.is_some()));
    if let Some(system) = request.system {
        messages.push(PromptMessage {
            role: PromptRole::System,
            content: convert_anthropic_system_content(system, "system")?,
            reasoning_content: None,
            tool_calls: Vec::new(),
        });
    }
    if request.messages.is_empty() {
        return Err(ApiFailure::bad_request(
            "at least one message is required".to_owned(),
            Some("messages".to_owned()),
        ));
    }
    messages.extend(convert_anthropic_messages(&state.image_client, request.messages).await?);
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
    let generation = GenerationRequest {
        messages,
        tools,
        tool_choice,
        thinking,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        stop,
    };
    ensure_context_fits(state, &generation)?;
    Ok(ConvertedRequest {
        generation,
        stream: request.stream,
        include_usage: false,
    })
}

fn convert_anthropic_system_content(
    content: AnthropicContent,
    parameter: &str,
) -> Result<Vec<PromptPart>, ApiFailure> {
    match content {
        AnthropicContent::String(text) => Ok(vec![PromptPart::Text(text)]),
        AnthropicContent::Parts(parts) => parts
            .into_iter()
            .enumerate()
            .map(|(index, part)| match part.kind.as_str() {
                "text" => part.text.map(PromptPart::Text).ok_or_else(|| {
                    ApiFailure::bad_request(
                        "a text block requires text".to_owned(),
                        Some(format!("{parameter}[{index}].text")),
                    )
                }),
                unsupported => Err(ApiFailure::bad_request(
                    format!("unsupported system content block type {unsupported:?}"),
                    Some(format!("{parameter}[{index}].type")),
                )),
            })
            .collect(),
    }
}

async fn convert_anthropic_messages(
    client: &reqwest::Client,
    messages: Vec<AnthropicMessage>,
) -> Result<Vec<PromptMessage>, ApiFailure> {
    let mut converted = Vec::new();
    for (message_index, message) in messages.into_iter().enumerate() {
        match message.role.as_str() {
            "user" => {
                let parts = anthropic_content_parts(client, message.content, message_index).await?;
                let mut user_parts = Vec::new();
                for part in parts {
                    match part {
                        AnthropicInputPart::Content(part) => user_parts.push(part),
                        AnthropicInputPart::ToolResult { _id: _, content } => {
                            if !user_parts.is_empty() {
                                converted.push(PromptMessage {
                                    role: PromptRole::User,
                                    content: std::mem::take(&mut user_parts),
                                    reasoning_content: None,
                                    tool_calls: Vec::new(),
                                });
                            }
                            converted.push(PromptMessage::text(PromptRole::Tool, content));
                        }
                        AnthropicInputPart::ToolUse(_) | AnthropicInputPart::Thinking(_) => {
                            return Err(ApiFailure::bad_request(
                                "tool_use and thinking blocks are allowed only in assistant messages".to_owned(),
                                Some(format!("messages[{message_index}].content")),
                            ));
                        }
                    }
                }
                if !user_parts.is_empty() {
                    converted.push(PromptMessage {
                        role: PromptRole::User,
                        content: user_parts,
                        reasoning_content: None,
                        tool_calls: Vec::new(),
                    });
                }
            }
            "assistant" => {
                let parts = anthropic_content_parts(client, message.content, message_index).await?;
                let mut content = Vec::new();
                let mut reasoning = Vec::new();
                let mut tool_calls = Vec::new();
                for part in parts {
                    match part {
                        AnthropicInputPart::Content(part) => content.push(part),
                        AnthropicInputPart::Thinking(thinking) => reasoning.push(thinking),
                        AnthropicInputPart::ToolUse(call) => tool_calls.push(call),
                        AnthropicInputPart::ToolResult { .. } => {
                            return Err(ApiFailure::bad_request(
                                "tool_result blocks are allowed only in user messages".to_owned(),
                                Some(format!("messages[{message_index}].content")),
                            ));
                        }
                    }
                }
                converted.push(PromptMessage {
                    role: PromptRole::Assistant,
                    content,
                    reasoning_content: (!reasoning.is_empty()).then(|| reasoning.join("\n")),
                    tool_calls,
                });
            }
            unsupported => {
                return Err(ApiFailure::bad_request(
                    format!("unsupported message role {unsupported:?}; supported roles are user and assistant"),
                    Some(format!("messages[{message_index}].role")),
                ));
            }
        }
    }
    Ok(converted)
}

enum AnthropicInputPart {
    Content(PromptPart),
    Thinking(String),
    ToolUse(ToolCall),
    ToolResult { _id: String, content: String },
}

async fn anthropic_content_parts(
    client: &reqwest::Client,
    content: AnthropicContent,
    message_index: usize,
) -> Result<Vec<AnthropicInputPart>, ApiFailure> {
    let parts = match content {
        AnthropicContent::String(text) => {
            return Ok(vec![AnthropicInputPart::Content(PromptPart::Text(text))])
        }
        AnthropicContent::Parts(parts) => parts,
    };
    let mut result = Vec::with_capacity(parts.len());
    for (part_index, part) in parts.into_iter().enumerate() {
        let parameter = format!("messages[{message_index}].content[{part_index}]");
        match part.kind.as_str() {
            "text" => {
                let text = part.text.ok_or_else(|| {
                    ApiFailure::bad_request(
                        "a text block requires text".to_owned(),
                        Some(format!("{parameter}.text")),
                    )
                })?;
                result.push(AnthropicInputPart::Content(PromptPart::Text(text)));
            }
            "image" => {
                let source = part.source.ok_or_else(|| {
                    ApiFailure::bad_request(
                        "an image block requires source".to_owned(),
                        Some(format!("{parameter}.source")),
                    )
                })?;
                result.push(AnthropicInputPart::Content(PromptPart::Image(
                    decode_anthropic_image(client, source, &format!("{parameter}.source")).await?,
                )));
            }
            "thinking" => {
                let thinking = part.thinking.ok_or_else(|| {
                    ApiFailure::bad_request(
                        "a thinking block requires thinking".to_owned(),
                        Some(format!("{parameter}.thinking")),
                    )
                })?;
                result.push(AnthropicInputPart::Thinking(thinking));
            }
            "tool_use" => {
                let name = part.name.ok_or_else(|| {
                    ApiFailure::bad_request(
                        "a tool_use block requires name".to_owned(),
                        Some(format!("{parameter}.name")),
                    )
                })?;
                validate_tool_name(&name, &format!("{parameter}.name"))?;
                let input = part.input.unwrap_or_else(|| json!({}));
                if !input.is_object() {
                    return Err(ApiFailure::bad_request(
                        "tool_use input must be a JSON object".to_owned(),
                        Some(format!("{parameter}.input")),
                    ));
                }
                result.push(AnthropicInputPart::ToolUse(ToolCall {
                    id: part.id.unwrap_or_default(),
                    name,
                    arguments: input,
                }));
            }
            "tool_result" => {
                let id = part.tool_use_id.ok_or_else(|| {
                    ApiFailure::bad_request(
                        "a tool_result block requires tool_use_id".to_owned(),
                        Some(format!("{parameter}.tool_use_id")),
                    )
                })?;
                result.push(AnthropicInputPart::ToolResult {
                    _id: id,
                    content: anthropic_tool_result_text(part.content, &parameter)?,
                });
            }
            unsupported => {
                return Err(ApiFailure::bad_request(
                    format!("unsupported Anthropic content block type {unsupported:?}"),
                    Some(format!("{parameter}.type")),
                ));
            }
        }
    }
    Ok(result)
}

fn anthropic_tool_result_text(
    content: Option<Value>,
    parameter: &str,
) -> Result<String, ApiFailure> {
    let Some(content) = content else {
        return Ok(String::new());
    };
    match content {
        Value::String(text) => Ok(text),
        Value::Array(parts) => {
            let mut text = String::new();
            for (index, part) in parts.into_iter().enumerate() {
                let part = part.as_object().ok_or_else(|| {
                    ApiFailure::bad_request(
                        "tool_result content blocks must be objects".to_owned(),
                        Some(format!("{parameter}.content[{index}]")),
                    )
                })?;
                if part.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(ApiFailure::bad_request(
                        "only text tool_result content blocks are supported".to_owned(),
                        Some(format!("{parameter}.content[{index}].type")),
                    ));
                }
                text.push_str(part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ApiFailure::bad_request(
                        "a text tool_result block requires text".to_owned(),
                        Some(format!("{parameter}.content[{index}].text")),
                    )
                })?);
            }
            Ok(text)
        }
        _ => Err(ApiFailure::bad_request(
            "tool_result content must be a string or text blocks".to_owned(),
            Some(format!("{parameter}.content")),
        )),
    }
}

async fn decode_anthropic_image(
    client: &reqwest::Client,
    source: AnthropicImageSource,
    parameter: &str,
) -> Result<InputImage, ApiFailure> {
    match source.kind.as_str() {
        "base64" => {
            let media_type = source
                .media_type
                .filter(|value| value.starts_with("image/"))
                .ok_or_else(|| {
                    ApiFailure::bad_request(
                        "image source media_type must be image/*".to_owned(),
                        Some(format!("{parameter}.media_type")),
                    )
                })?;
            let bytes = BASE64
                .decode(source.data.unwrap_or_default())
                .map_err(|_| {
                    ApiFailure::bad_request(
                        "image source data contains invalid base64".to_owned(),
                        Some(format!("{parameter}.data")),
                    )
                })?;
            if bytes.is_empty() || bytes.len() > MAX_REMOTE_IMAGE_BYTES {
                return Err(ApiFailure::bad_request(
                    format!("image data must be between 1 and {MAX_REMOTE_IMAGE_BYTES} bytes"),
                    Some(format!("{parameter}.data")),
                ));
            }
            validate_input_image(InputImage { media_type, bytes }, parameter)
        }
        "url" => {
            load_openai_image(
                client,
                source.url.ok_or_else(|| {
                    ApiFailure::bad_request(
                        "a URL image source requires url".to_owned(),
                        Some(format!("{parameter}.url")),
                    )
                })?,
                parameter,
            )
            .await
        }
        unsupported => Err(ApiFailure::bad_request(
            format!("unsupported image source type {unsupported:?}"),
            Some(format!("{parameter}.type")),
        )),
    }
}

fn convert_anthropic_tools(
    tools: Option<Vec<AnthropicTool>>,
) -> Result<Vec<ToolDefinition>, ApiFailure> {
    let mut converted = Vec::new();
    for (index, tool) in tools.unwrap_or_default().into_iter().enumerate() {
        validate_tool_name(&tool.name, &format!("tools[{index}].name"))?;
        let input_schema = tool
            .input_schema
            .unwrap_or_else(|| json!({"type": "object"}));
        if !input_schema.is_object() {
            return Err(ApiFailure::bad_request(
                "tool input_schema must be a JSON Schema object".to_owned(),
                Some(format!("tools[{index}].input_schema")),
            ));
        }
        if converted
            .iter()
            .any(|existing: &ToolDefinition| existing.name == tool.name)
        {
            return Err(ApiFailure::bad_request(
                "tool names must be unique".to_owned(),
                Some(format!("tools[{index}].name")),
            ));
        }
        converted.push(ToolDefinition {
            name: tool.name,
            description: tool.description,
            input_schema,
        });
    }
    Ok(converted)
}

fn parse_anthropic_tool_choice(
    choice: Option<Value>,
    tools: &[ToolDefinition],
) -> Result<ToolChoice, ApiFailure> {
    let Some(choice) = choice else {
        return Ok(ToolChoice::Auto);
    };
    let object = choice.as_object().ok_or_else(|| {
        ApiFailure::bad_request(
            "tool_choice must be an object".to_owned(),
            Some("tool_choice".to_owned()),
        )
    })?;
    let choice = match object.get("type").and_then(Value::as_str) {
        Some("auto") => ToolChoice::Auto,
        Some("any") => ToolChoice::Required,
        Some("tool") => ToolChoice::Specific(
            object
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ApiFailure::bad_request(
                        "tool_choice type=tool requires name".to_owned(),
                        Some("tool_choice.name".to_owned()),
                    )
                })?
                .to_owned(),
        ),
        _ => {
            return Err(ApiFailure::bad_request(
                "tool_choice.type must be auto, any, or tool".to_owned(),
                Some("tool_choice.type".to_owned()),
            ));
        }
    };
    if let ToolChoice::Specific(name) = &choice {
        if !tools.iter().any(|tool| tool.name == *name) {
            return Err(ApiFailure::bad_request(
                format!("tool_choice selects undeclared tool {name:?}"),
                Some("tool_choice.name".to_owned()),
            ));
        }
    }
    if tools.is_empty() {
        return Err(ApiFailure::bad_request(
            "tool_choice requires at least one declared tool".to_owned(),
            Some("tool_choice".to_owned()),
        ));
    }
    Ok(choice)
}

fn parse_anthropic_thinking(
    thinking: Option<Value>,
    max_tokens: u32,
) -> Result<ThinkingConfig, ApiFailure> {
    let Some(thinking) = thinking else {
        return Ok(ThinkingConfig::DISABLED);
    };
    let object = thinking.as_object().ok_or_else(|| {
        ApiFailure::bad_request(
            "thinking must be an object".to_owned(),
            Some("thinking".to_owned()),
        )
    })?;
    match object.get("type").and_then(Value::as_str) {
        Some("disabled") => Ok(ThinkingConfig::DISABLED),
        Some("enabled") => {
            let budget = object
                .get("budget_tokens")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    ApiFailure::bad_request(
                        "enabled thinking requires budget_tokens".to_owned(),
                        Some("thinking.budget_tokens".to_owned()),
                    )
                })?;
            let budget = u32::try_from(budget).map_err(|_| {
                ApiFailure::bad_request(
                    "thinking.budget_tokens is too large".to_owned(),
                    Some("thinking.budget_tokens".to_owned()),
                )
            })?;
            if budget == 0 || budget > max_tokens {
                return Err(ApiFailure::bad_request(
                    "thinking.budget_tokens must be between 1 and max_tokens".to_owned(),
                    Some("thinking.budget_tokens".to_owned()),
                ));
            }
            Ok(ThinkingConfig {
                enabled: true,
                budget_tokens: Some(budget),
            })
        }
        _ => Err(ApiFailure::bad_request(
            "thinking.type must be enabled or disabled".to_owned(),
            Some("thinking.type".to_owned()),
        )),
    }
}

fn anthropic_stream(identifier: String, model: String, generation: Generation) -> Response {
    let start = json!({
        "type": "message_start",
        "message": {
            "id": &identifier,
            "type": "message",
            "role": "assistant",
            "model": &model,
            "content": [],
            "stop_reason": Value::Null,
            "stop_sequence": Value::Null,
            "usage": {"input_tokens": generation.input_tokens, "output_tokens": 0}
        }
    });
    let mut events = vec![Event::default()
        .event("message_start")
        .data(start.to_string())];
    let mut index = 0_u32;
    if let Some(thinking) = generation
        .reasoning
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        events.push(
            Event::default().event("content_block_start").data(
                json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": ""}
                })
                .to_string(),
            ),
        );
        events.push(
            Event::default().event("content_block_delta").data(
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": thinking}
                })
                .to_string(),
            ),
        );
        events.push(Event::default().event("content_block_delta").data(
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": thinking_signature(&identifier)}
            })
            .to_string(),
        ));
        events.push(
            Event::default()
                .event("content_block_stop")
                .data(json!({"type": "content_block_stop", "index": index}).to_string()),
        );
        index += 1;
    }
    if !generation.text.is_empty() {
        events.push(
            Event::default().event("content_block_start").data(
                json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""}
                })
                .to_string(),
            ),
        );
        events.push(
            Event::default().event("content_block_delta").data(
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": &generation.text}
                })
                .to_string(),
            ),
        );
        events.push(
            Event::default()
                .event("content_block_stop")
                .data(json!({"type": "content_block_stop", "index": index}).to_string()),
        );
        index += 1;
    }
    for call in &generation.tool_calls {
        events.push(Event::default().event("content_block_start").data(
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": &call.id, "name": &call.name, "input": {}}
            })
            .to_string(),
        ));
        events.push(Event::default().event("content_block_delta").data(
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_owned())
                }
            })
            .to_string(),
        ));
        events.push(
            Event::default()
                .event("content_block_stop")
                .data(json!({"type": "content_block_stop", "index": index}).to_string()),
        );
        index += 1;
    }
    events.push(
        Event::default().event("message_delta").data(
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": generation.finish_reason.anthropic_label(),
                    "stop_sequence": Value::Null
                },
                "usage": {"output_tokens": generation.output_tokens}
            })
            .to_string(),
        ),
    );
    events.push(
        Event::default()
            .event("message_stop")
            .data(json!({"type": "message_stop"}).to_string()),
    );

    Sse::new(stream::iter(
        events.into_iter().map(Ok::<Event, Infallible>),
    ))
    .keep_alive(KeepAlive::default())
    .into_response()
}

fn anthropic_response(identifier: String, model: String, generation: &Generation) -> Value {
    json!({
        "id": identifier,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": anthropic_content_blocks(&identifier, generation),
        "stop_reason": generation.finish_reason.anthropic_label(),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens
        }
    })
}

fn anthropic_content_blocks(identifier: &str, generation: &Generation) -> Vec<Value> {
    let mut blocks = Vec::new();
    if let Some(thinking) = generation
        .reasoning
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        blocks.push(json!({
            "type": "thinking",
            "thinking": thinking,
            "signature": thinking_signature(identifier)
        }));
    }
    if !generation.text.is_empty() {
        blocks.push(json!({"type": "text", "text": &generation.text}));
    }
    for call in &generation.tool_calls {
        blocks.push(json!({
            "type": "tool_use",
            "id": &call.id,
            "name": &call.name,
            "input": &call.arguments
        }));
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }
    blocks
}

fn thinking_signature(identifier: &str) -> String {
    format!("qwen38-metal:{identifier}")
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

fn ensure_context_fits(state: &ApiState, request: &GenerationRequest) -> Result<(), ApiFailure> {
    let input_tokens = state
        .engine
        .estimate_prompt_tokens(request)
        .map_err(ApiFailure::from_engine)?;
    let requested = input_tokens
        .checked_add(request.max_tokens)
        .ok_or_else(|| {
            ApiFailure::bad_request("requested token count overflowed u32".to_owned(), None)
        })?;
    let descriptor = state.descriptor();
    if requested > descriptor.context_tokens {
        return Err(ApiFailure::bad_request(
            format!(
                "prompt uses {input_tokens} tokens and max_tokens requests {max_tokens}, exceeding the {}-token context window",
                descriptor.context_tokens,
                max_tokens = request.max_tokens,
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
            message:
                "the native generation queue is full; retry after an active response completes"
                    .to_owned(),
            parameter: None,
            kind: FailureKind::Busy,
        }
    }

    fn unavailable(message: String) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message,
            parameter: None,
            kind: FailureKind::Unavailable,
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
            EngineError::InvalidRequest(message) => Self::bad_request(message, None),
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
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::Duration,
    };
    use tower::ServiceExt;

    fn test_router(api_key: Option<&str>) -> Router {
        test_router_with_response("native fixture answer", 64, 32, api_key)
    }

    fn test_router_with_response(
        response: &str,
        context_tokens: u32,
        max_output_tokens: u32,
        api_key: Option<&str>,
    ) -> Router {
        let engine = Arc::new(FixtureEngine::new("qwen3.8-27b", context_tokens, response));
        let mut config = ServerConfig::local();
        config.max_output_tokens = max_output_tokens;
        config.api_key = api_key.map(str::to_owned);
        router(engine, config)
    }

    #[derive(Debug, Clone)]
    struct SlowFixtureEngine {
        fixture: FixtureEngine,
        starts: Arc<AtomicUsize>,
        delay: Duration,
    }

    impl InferenceEngine for SlowFixtureEngine {
        fn descriptor(&self) -> ModelDescriptor {
            self.fixture.descriptor()
        }

        fn estimate_prompt_tokens(&self, request: &GenerationRequest) -> Result<u32, EngineError> {
            self.fixture.estimate_prompt_tokens(request)
        }

        fn generate(&self, request: GenerationRequest) -> Result<Generation, EngineError> {
            self.starts.fetch_add(1, Ordering::Release);
            std::thread::sleep(self.delay);
            self.fixture.generate(request)
        }
    }

    fn slow_router(
        generation_concurrency: usize,
        max_queued_requests: usize,
        starts: Arc<AtomicUsize>,
    ) -> Router {
        let engine = Arc::new(SlowFixtureEngine {
            fixture: FixtureEngine::new("qwen3.8-27b", 128, "slow fixture answer"),
            starts,
            delay: Duration::from_millis(250),
        });
        let mut config = ServerConfig::local();
        config.max_output_tokens = 32;
        config.generation_concurrency = generation_concurrency;
        config.max_queued_requests = max_queued_requests;
        router(engine, config)
    }

    fn openai_completion_request() -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "qwen3.8-27b",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 16
                })
                .to_string(),
            ))
            .unwrap()
    }

    async fn wait_for_starts(starts: &AtomicUsize, wanted: usize) {
        for _ in 0..100 {
            if starts.load(Ordering::Acquire) >= wanted {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!(
            "timed out waiting for {wanted} generation starts; observed {}",
            starts.load(Ordering::Acquire)
        );
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
    async fn openai_supports_tool_calls_and_tool_result_history() {
        let router = test_router_with_response(
            "<tool_call>\n<function=get_weather>\n<parameter=location>\n\"Shanghai\"\n</parameter>\n</function>\n</tool_call>",
            1_024,
            128,
            None,
        );
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Gets the current weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }]);
        let first = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [{"role": "user", "content": "What is the weather in Shanghai?"}],
                            "tools": tools,
                            "tool_choice": "required",
                            "max_tokens": 64
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let first: Value = serde_json::from_str(&response_text(first).await).unwrap();
        assert_eq!(first["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            first["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            serde_json::from_str::<Value>(
                first["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap()["location"],
            "Shanghai"
        );

        let second = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [
                                {"role": "user", "content": "What is the weather in Shanghai?"},
                                {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": [{
                                        "id": "call_weather",
                                        "type": "function",
                                        "function": {"name": "get_weather", "arguments": "{\"location\":\"Shanghai\"}"}
                                    }]
                                },
                                {"role": "tool", "tool_call_id": "call_weather", "content": "22 C and clear"}
                            ],
                            "tools": tools,
                            "max_tokens": 64
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn openai_accepts_data_uri_images() {
        let response = test_router_with_response("image fixture answer", 1_024, 64, None)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "What color is this?"},
                                    {"type": "image_url", "image_url": {"url": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="}}
                                ]
                            }],
                            "max_tokens": 16
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response: Value = serde_json::from_str(&response_text(response).await).unwrap();
        assert!(response["usage"]["prompt_tokens"].as_u64().unwrap() >= 256);
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
    async fn anthropic_supports_thinking_tools_and_image_blocks() {
        let thinking_router = test_router_with_response(
            "<think>Use the available evidence.</think>\n\nThe answer is 4.",
            1_024,
            128,
            None,
        );
        let thinking = thinking_router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                            "max_tokens": 64,
                            "thinking": {"type": "enabled", "budget_tokens": 16}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(thinking.status(), StatusCode::OK);
        let thinking: Value = serde_json::from_str(&response_text(thinking).await).unwrap();
        assert_eq!(thinking["content"][0]["type"], "thinking");
        assert_eq!(
            thinking["content"][0]["thinking"],
            "Use the available evidence."
        );
        assert_eq!(thinking["content"][1]["type"], "text");
        assert_eq!(thinking["content"][1]["text"], "The answer is 4.");

        let stream = thinking_router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                            "max_tokens": 64,
                            "stream": true,
                            "thinking": {"type": "enabled", "budget_tokens": 16}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let stream = response_text(stream).await;
        assert!(stream.contains("\"type\":\"thinking_delta\""));
        assert!(stream.contains("\"type\":\"signature_delta\""));

        let tool_router = test_router_with_response(
            "<tool_call>\n<function=get_weather>\n<parameter=location>\n\"Shanghai\"\n</parameter>\n</function>\n</tool_call>",
            1_024,
            128,
            None,
        );
        let tool = tool_router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [{"role": "user", "content": "Weather in Shanghai?"}],
                            "max_tokens": 64,
                            "tools": [{
                                "name": "get_weather",
                                "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
                            }],
                            "tool_choice": {"type": "any"}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(tool.status(), StatusCode::OK);
        let tool: Value = serde_json::from_str(&response_text(tool).await).unwrap();
        assert_eq!(tool["stop_reason"], "tool_use");
        assert_eq!(tool["content"][0]["type"], "tool_use");
        assert_eq!(tool["content"][0]["name"], "get_weather");

        let image = test_router_with_response("image fixture answer", 1_024, 64, None)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(
                        json!({
                            "model": "qwen3.8-27b",
                            "messages": [{
                                "role": "user",
                                "content": [{
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/gif",
                                        "data": "R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
                                    }
                                }]
                            }],
                            "max_tokens": 16
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(image.status(), StatusCode::OK);
        let image: Value = serde_json::from_str(&response_text(image).await).unwrap();
        assert!(image["usage"]["input_tokens"].as_u64().unwrap() >= 256);
    }

    #[tokio::test]
    async fn scheduler_runs_multiple_configured_generation_lanes() {
        let starts = Arc::new(AtomicUsize::new(0));
        let router = slow_router(2, 2, starts.clone());
        let first = tokio::spawn(router.clone().oneshot(openai_completion_request()));
        wait_for_starts(&starts, 1).await;
        let second = tokio::spawn(router.clone().oneshot(openai_completion_request()));
        wait_for_starts(&starts, 2).await;

        assert_eq!(first.await.unwrap().unwrap().status(), StatusCode::OK);
        assert_eq!(second.await.unwrap().unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn scheduler_queues_within_its_bound_and_rejects_overflow() {
        let starts = Arc::new(AtomicUsize::new(0));
        let queued_router = slow_router(1, 2, starts.clone());
        let first = tokio::spawn(queued_router.clone().oneshot(openai_completion_request()));
        wait_for_starts(&starts, 1).await;
        let second = tokio::spawn(queued_router.clone().oneshot(openai_completion_request()));
        assert_eq!(first.await.unwrap().unwrap().status(), StatusCode::OK);
        assert_eq!(second.await.unwrap().unwrap().status(), StatusCode::OK);

        let starts = Arc::new(AtomicUsize::new(0));
        let full_router = slow_router(1, 1, starts.clone());
        let first = tokio::spawn(full_router.clone().oneshot(openai_completion_request()));
        wait_for_starts(&starts, 1).await;
        let overflow = full_router
            .oneshot(openai_completion_request())
            .await
            .unwrap();
        assert_eq!(overflow.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(overflow.headers()[header::RETRY_AFTER], "1");
        assert_eq!(first.await.unwrap().unwrap().status(), StatusCode::OK);
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
    async fn rejects_invalid_images_and_context_overflow_without_calling_the_engine() {
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
