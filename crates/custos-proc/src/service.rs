use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use custos_core::{ParseError, ScannerConfig, inspect_payload_with_report};
use envoy_types::pb::envoy::r#type::v3::{HttpStatus, StatusCode};
use envoy_types::pb::envoy::service::ext_proc::v3::common_response::ResponseStatus;
use envoy_types::pb::envoy::service::ext_proc::v3::external_processor_server::ExternalProcessor;
use envoy_types::pb::envoy::service::ext_proc::v3::processing_request::Request as EnvoyRequest;
use envoy_types::pb::envoy::service::ext_proc::v3::processing_response::Response as EnvoyResponse;
use envoy_types::pb::envoy::service::ext_proc::v3::{
    BodyResponse, CommonResponse, GrpcStatus, HeadersResponse, HttpHeaders, ImmediateResponse,
    ProcessingRequest, ProcessingResponse, TrailersResponse,
};
use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use tracing::{ debug, info, info_span, warn, Instrument };

/// gRPC status code 3 = `INVALID_ARGUMENT`.
const GRPC_STATUS_INVALID_ARGUMENT: u32 = 3;

/// Maps a `ParseError` to a static details string for Envoy's `%RESPONSE_CODE_DETAILS%` log field
/// without allocating dynamic strings on the hot path.
#[must_use]
pub const fn static_error_details(err: &ParseError) -> &'static str {
    match err {
        ParseError::EmptyFrame => "custos_denied:empty_frame",
        ParseError::UnsupportedCompression(_) => "custos_denied:unsupported_compression",
        ParseError::FrameLengthMismatch { .. } => "custos_denied:frame_length_mismatch",
        ParseError::UnexpectedEof => "custos_denied:unexpected_eof",
        ParseError::MalformedVarint => "custos_denied:malformed_varint",
        ParseError::InvalidWireType(_) => "custos_denied:invalid_wire_type",
        ParseError::NestingLimitExceeded(_) => "custos_denied:nesting_limit_exceeded",
        ParseError::SubmessageOverrun => "custos_denied:submessage_overrun",
        ParseError::NegativeDimension(_) => "custos_denied:negative_tensor_dimension",
        ParseError::ShapeOverflow => "custos_denied:tensor_shape_overflow",
        ParseError::BufferCeilingExceeded { .. } => "custos_denied:tensor_buffer_ceiling_exceeded",
    }
}

/// Lock-free atomic telemetry counters for cluster observability, capacity planning,
/// and threshold tuning across iterations.
#[derive(Debug, Default)]
pub struct ServiceMetrics {
    pub streams_opened: AtomicU64,
    pub streams_active: AtomicU64,
    pub stream_errors: AtomicU64,
    pub headers_evaluated: AtomicU64,
    pub bodies_evaluated: AtomicU64,
    pub bodies_admitted: AtomicU64,
    pub bodies_rejected: AtomicU64,
    pub bytes_inspected_total: AtomicU64,
    pub bytes_skipped_o1_total: AtomicU64,
    pub max_depth_observed: AtomicU64,
    pub max_tensor_elements_observed: AtomicU64,
    pub rejected_malformed_varint: AtomicU64,
    pub rejected_nesting_limit: AtomicU64,
    pub rejected_shape_overflow: AtomicU64,
    pub rejected_buffer_ceiling: AtomicU64,
    pub rejected_negative_dimension: AtomicU64,
    pub rejected_eof_or_overrun: AtomicU64,
    pub rejected_frame_or_wire: AtomicU64,
}

impl ServiceMetrics {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn record_rejection(&self, err: &ParseError) {
        match err {
            ParseError::MalformedVarint => {
                self.rejected_malformed_varint
                    .fetch_add(1, Ordering::Relaxed);
            }
            ParseError::NestingLimitExceeded(_) => {
                self.rejected_nesting_limit.fetch_add(1, Ordering::Relaxed);
            }
            ParseError::ShapeOverflow => {
                self.rejected_shape_overflow.fetch_add(1, Ordering::Relaxed);
            }
            ParseError::BufferCeilingExceeded { .. } => {
                self.rejected_buffer_ceiling.fetch_add(1, Ordering::Relaxed);
            }
            ParseError::NegativeDimension(_) => {
                self.rejected_negative_dimension
                    .fetch_add(1, Ordering::Relaxed);
            }
            ParseError::UnexpectedEof | ParseError::SubmessageOverrun => {
                self.rejected_eof_or_overrun.fetch_add(1, Ordering::Relaxed);
            }
            ParseError::EmptyFrame
            | ParseError::UnsupportedCompression(_)
            | ParseError::FrameLengthMismatch { .. }
            | ParseError::InvalidWireType(_) => {
                self.rejected_frame_or_wire.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Emits a structured periodic or shutdown summary log entry with all cumulative counters.
    pub fn emit_summary_log(&self, reason: &'static str, uptime_secs: u64) {
        info!(
            event = "telemetry_summary",
            reason = reason,
            uptime_secs = uptime_secs,
            streams_opened = self.streams_opened.load(Ordering::Relaxed),
            streams_active = self.streams_active.load(Ordering::Relaxed),
            stream_errors = self.stream_errors.load(Ordering::Relaxed),
            headers_evaluated = self.headers_evaluated.load(Ordering::Relaxed),
            bodies_evaluated = self.bodies_evaluated.load(Ordering::Relaxed),
            bodies_admitted = self.bodies_admitted.load(Ordering::Relaxed),
            bodies_rejected = self.bodies_rejected.load(Ordering::Relaxed),
            bytes_inspected_total = self.bytes_inspected_total.load(Ordering::Relaxed),
            bytes_skipped_o1_total = self.bytes_skipped_o1_total.load(Ordering::Relaxed),
            peak_nesting_depth_observed = self.max_depth_observed.load(Ordering::Relaxed),
            peak_tensor_elements_observed =
                self.max_tensor_elements_observed.load(Ordering::Relaxed),
            rejected_malformed_varint = self.rejected_malformed_varint.load(Ordering::Relaxed),
            rejected_nesting_limit = self.rejected_nesting_limit.load(Ordering::Relaxed),
            rejected_shape_overflow = self.rejected_shape_overflow.load(Ordering::Relaxed),
            rejected_buffer_ceiling = self.rejected_buffer_ceiling.load(Ordering::Relaxed),
            rejected_negative_dimension =
                self.rejected_negative_dimension.load(Ordering::Relaxed),
            rejected_eof_or_overrun = self.rejected_eof_or_overrun.load(Ordering::Relaxed),
            rejected_frame_or_wire = self.rejected_frame_or_wire.load(Ordering::Relaxed),
            "custos-proc cumulative telemetry snapshot"
        );
    }
}

/// Request-scoped HTTP/2 and gRPC correlation metadata extracted from Envoy's `RequestHeaders` frame.
#[derive(Debug, Clone)]
pub struct StreamRequestContext {
    pub stream_id: u64,
    pub request_id: String,
    pub grpc_path: String,
    pub authority: String,
    pub user_agent: String,
    pub content_length: String,
    pub grpc_timeout: String,
}

impl Default for StreamRequestContext {
    fn default() -> Self {
        Self {
            stream_id: 0,
            request_id: "-".to_owned(),
            grpc_path: "-".to_owned(),
            authority: "-".to_owned(),
            user_agent: "-".to_owned(),
            content_length: "-".to_owned(),
            grpc_timeout: "-".to_owned(),
        }
    }
}

impl StreamRequestContext {
    fn new(stream_id: u64) -> Self {
        Self {
            stream_id,
            ..Default::default()
        }
    }

    /// Populates correlation fields from Envoy's `HttpHeaders` message.
    /// Supports both string `value` and byte-array `raw_value` encodings used by Envoy `ext_proc`.
    fn update_from_envoy_headers(&mut self, http_headers: &HttpHeaders) {
        let Some(header_map) = http_headers.headers.as_ref() else {
            return;
        };
        for hv in &header_map.headers {
            let val_cow = if !hv.value.is_empty() {
                std::borrow::Cow::Borrowed(hv.value.as_str())
            } else if !hv.raw_value.is_empty() {
                String::from_utf8_lossy(&hv.raw_value)
            } else {
                continue;
            };

            match hv.key.as_str() {
                "x-request-id" => self.request_id = val_cow.into_owned(),
                ":path" => self.grpc_path = val_cow.into_owned(),
                ":authority" | "host" => self.authority = val_cow.into_owned(),
                "user-agent" => self.user_agent = val_cow.into_owned(),
                "content-length" => self.content_length = val_cow.into_owned(),
                "grpc-timeout" => self.grpc_timeout = val_cow.into_owned(),
                _ => {}
            }
        }
    }
}

/// Stateless Envoy `ExternalProcessor` service implementation backed by `custos-core`.
#[derive(Debug, Clone)]
pub struct CustosExtProcService {
    config: ScannerConfig,
    log_admitted: bool,
    metrics: Arc<ServiceMetrics>,
}

impl CustosExtProcService {
    #[must_use]
    pub fn new(config: ScannerConfig) -> Self {
        Self {
            config,
            log_admitted: true,
            metrics: Arc::new(ServiceMetrics::new()),
        }
    }

    #[must_use]
    pub fn with_telemetry(
        config: ScannerConfig,
        log_admitted: bool,
        metrics: Arc<ServiceMetrics>,
    ) -> Self {
        Self {
            config,
            log_admitted,
            metrics,
        }
    }

    #[must_use]
    pub fn metrics(&self) -> &Arc<ServiceMetrics> {
        &self.metrics
    }

    /// Evaluates a single `ProcessingRequest` frame without stream context.
    #[must_use]
    pub fn evaluate_request(&self, req: &ProcessingRequest) -> Option<ProcessingResponse> {
        let mut ctx = StreamRequestContext::default();
        self.evaluate_request_with_context(req, &mut ctx)
    }

    /// Evaluates a single `ProcessingRequest` frame while maintaining per-stream correlation headers
    /// and emitting structured telemetry logs.
    #[must_use]
    pub fn evaluate_request_with_context(
        &self,
        req: &ProcessingRequest,
        ctx: &mut StreamRequestContext,
    ) -> Option<ProcessingResponse> {
        let request_kind = req.request.as_ref()?;
        let response_oneof = match request_kind {
            EnvoyRequest::RequestHeaders(headers) => {
                self.metrics
                    .headers_evaluated
                    .fetch_add(1, Ordering::Relaxed);
                ctx.update_from_envoy_headers(headers);
                debug!(
                    event = "request_headers_received",
                    stream_id = ctx.stream_id,
                    request_id = %ctx.request_id,
                    grpc_path = %ctx.grpc_path,
                    authority = %ctx.authority,
                    user_agent = %ctx.user_agent,
                    content_length = %ctx.content_length,
                    grpc_timeout = %ctx.grpc_timeout,
                    end_of_stream = headers.end_of_stream,
                    observability_mode = req.observability_mode,
                    "admitted request headers; awaiting buffered gRPC inference body"
                );
                EnvoyResponse::RequestHeaders(HeadersResponse {
                    response: Some(CommonResponse {
                        status: ResponseStatus::Continue as i32,
                        ..Default::default()
                    }),
                })
            }
            EnvoyRequest::ResponseHeaders(_) => EnvoyResponse::ResponseHeaders(HeadersResponse {
                response: Some(CommonResponse {
                    status: ResponseStatus::Continue as i32,
                    ..Default::default()
                }),
            }),
            EnvoyRequest::RequestBody(http_body) => {
                self.metrics
                    .bodies_evaluated
                    .fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .bytes_inspected_total
                    .fetch_add(http_body.body.len() as u64, Ordering::Relaxed);

                let start = Instant::now();
                let scan_result = inspect_payload_with_report(&http_body.body, &self.config);
                let elapsed = start.elapsed();
                let scan_duration_ns = elapsed.as_nanos() as u64;
                let scan_duration_us = elapsed.as_micros() as u64;

                match scan_result {
                    Ok(report) => {
                        let admitted_total = self
                            .metrics
                            .bodies_admitted
                            .fetch_add(1, Ordering::Relaxed)
                            + 1;
                        let rejected_total = self.metrics.bodies_rejected.load(Ordering::Relaxed);
                        self.metrics
                            .bytes_skipped_o1_total
                            .fetch_add(report.skipped_bulk_bytes as u64, Ordering::Relaxed);
                        self.metrics
                            .max_depth_observed
                            .fetch_max(report.max_depth_reached as u64, Ordering::Relaxed);
                        if report.max_tensor_elements_seen > 0 {
                            self.metrics.max_tensor_elements_observed.fetch_max(
                                report.max_tensor_elements_seen as u64,
                                Ordering::Relaxed,
                            );
                        }

                        if self.log_admitted {
                            info!(
                                event = "request_admitted",
                                decision = "ALLOW",
                                stream_id = ctx.stream_id,
                                request_id = %ctx.request_id,
                                grpc_path = %ctx.grpc_path,
                                authority = %ctx.authority,
                                user_agent = %ctx.user_agent,
                                scan_duration_us = scan_duration_us,
                                scan_duration_ns = scan_duration_ns,
                                payload_bytes = report.payload_bytes,
                                protobuf_bytes = report.protobuf_bytes,
                                grpc_framed = report.grpc_framed,
                                tags_scanned = report.tags_scanned,
                                skipped_bulk_bytes = report.skipped_bulk_bytes,
                                max_depth_reached = report.max_depth_reached,
                                max_nesting_depth_limit = self.config.max_nesting_depth,
                                input_tensor_count = report.input_tensor_count,
                                max_tensor_elements_seen = report.max_tensor_elements_seen,
                                total_tensor_elements = report.total_tensor_elements,
                                max_tensor_elements_limit = self.config.max_tensor_elements,
                                end_of_stream = http_body.end_of_stream,
                                admitted_total = admitted_total,
                                rejected_total = rejected_total,
                                "admitted KServe v2 inference payload"
                            );
                        } else {
                            debug!(
                                event = "request_admitted",
                                decision = "ALLOW",
                                stream_id = ctx.stream_id,
                                request_id = %ctx.request_id,
                                grpc_path = %ctx.grpc_path,
                                scan_duration_us = scan_duration_us,
                                payload_bytes = report.payload_bytes,
                                max_depth_reached = report.max_depth_reached,
                                input_tensor_count = report.input_tensor_count,
                                max_tensor_elements_seen = report.max_tensor_elements_seen,
                                "admitted KServe v2 inference payload"
                            );
                        }

                        EnvoyResponse::RequestBody(BodyResponse {
                            response: Some(CommonResponse {
                                status: ResponseStatus::Continue as i32,
                                ..Default::default()
                            }),
                        })
                    }
                    Err(failure) => {
                        let rejected_total = self
                            .metrics
                            .bodies_rejected
                            .fetch_add(1, Ordering::Relaxed)
                            + 1;
                        let admitted_total = self.metrics.bodies_admitted.load(Ordering::Relaxed);
                        self.metrics.record_rejection(&failure.error);
                        self.metrics.bytes_skipped_o1_total.fetch_add(
                            failure.partial_report.skipped_bulk_bytes as u64,
                            Ordering::Relaxed,
                        );
                        self.metrics.max_depth_observed.fetch_max(
                            failure.partial_report.max_depth_reached as u64,
                            Ordering::Relaxed,
                        );

                        let details = static_error_details(&failure.error);
                        warn!(
                            event = "request_rejected",
                            decision = "DENY",
                            stream_id = ctx.stream_id,
                            request_id = %ctx.request_id,
                            grpc_path = %ctx.grpc_path,
                            authority = %ctx.authority,
                            user_agent = %ctx.user_agent,
                            scan_duration_us = scan_duration_us,
                            scan_duration_ns = scan_duration_ns,
                            error_code = details,
                            error_reason = %failure.error,
                            failure_byte_offset = failure.byte_offset,
                            failure_depth = failure.depth_at_failure,
                            last_field_number = failure.last_field_number,
                            last_wire_type = failure.last_wire_type,
                            payload_bytes = failure.partial_report.payload_bytes,
                            protobuf_bytes = failure.partial_report.protobuf_bytes,
                            grpc_framed = failure.partial_report.grpc_framed,
                            tags_scanned_before_failure = failure.partial_report.tags_scanned,
                            skipped_bulk_bytes_before_failure =
                                failure.partial_report.skipped_bulk_bytes,
                            max_depth_reached = failure.partial_report.max_depth_reached,
                            max_nesting_depth_limit = self.config.max_nesting_depth,
                            input_tensor_count = failure.partial_report.input_tensor_count,
                            max_tensor_elements_seen =
                                failure.partial_report.max_tensor_elements_seen,
                            max_tensor_elements_limit = self.config.max_tensor_elements,
                            http_status = StatusCode::BadRequest as i32,
                            grpc_status = GRPC_STATUS_INVALID_ARGUMENT,
                            admitted_total = admitted_total,
                            rejected_total = rejected_total,
                            "rejected adversarial KServe v2 inference payload"
                        );

                        EnvoyResponse::ImmediateResponse(ImmediateResponse {
                            status: Some(HttpStatus {
                                code: StatusCode::BadRequest as i32,
                            }),
                            headers: None,
                            body: details.as_bytes().to_vec(),
                            grpc_status: Some(GrpcStatus {
                                status: GRPC_STATUS_INVALID_ARGUMENT,
                            }),
                            details: details.to_owned(),
                        })
                    }
                }
            }
            EnvoyRequest::ResponseBody(_) => EnvoyResponse::ResponseBody(BodyResponse {
                response: Some(CommonResponse {
                    status: ResponseStatus::Continue as i32,
                    ..Default::default()
                }),
            }),
            EnvoyRequest::RequestTrailers(_) => {
                EnvoyResponse::RequestTrailers(TrailersResponse::default())
            }
            EnvoyRequest::ResponseTrailers(_) => {
                EnvoyResponse::ResponseTrailers(TrailersResponse::default())
            }
        };

        Some(ProcessingResponse {
            response: Some(response_oneof),
            ..Default::default()
        })
    }
}

#[tonic::async_trait]
impl ExternalProcessor for CustosExtProcService {
    type ProcessStream =
        Pin<Box<dyn Stream<Item = Result<ProcessingResponse, Status>> + Send + 'static>>;

    async fn process(
        &self,
        request: Request<Streaming<ProcessingRequest>>,
    ) -> Result<Response<Self::ProcessStream>, Status> {
        let mut in_stream = request.into_inner();
        let (tx, rx) = mpsc::channel(16);
        let svc = self.clone();

        let stream_id = svc.metrics.streams_opened.fetch_add(1, Ordering::Relaxed) + 1;
        svc.metrics.streams_active.fetch_add(1, Ordering::Relaxed);

        let span = info_span!("ext_proc_stream", stream_id = stream_id);
        tokio::spawn(
            async move {
                let mut ctx = StreamRequestContext::new(stream_id);
                loop {
                    match in_stream.message().await {
                        Ok(Some(req)) => {
                            if let Some(resp) = svc.evaluate_request_with_context(&req, &mut ctx) {
                                let is_immediate = matches!(
                                    resp.response,
                                    Some(EnvoyResponse::ImmediateResponse(_))
                                );
                                if tx.send(Ok(resp)).await.is_err() {
                                    debug!(
                                        event = "stream_send_closed",
                                        stream_id = ctx.stream_id,
                                        request_id = %ctx.request_id,
                                        "downstream Envoy closed ext_proc response channel early"
                                    );
                                    break;
                                }
                                if is_immediate {
                                    break;
                                }
                            }
                        }
                        Ok(None) => {
                            debug!(
                                event = "stream_completed",
                                stream_id = ctx.stream_id,
                                request_id = %ctx.request_id,
                                "ext_proc stream completed cleanly"
                            );
                            break;
                        }
                        Err(status) => {
                            svc.metrics.stream_errors.fetch_add(1, Ordering::Relaxed);
                            warn!(
                                event = "stream_transport_error",
                                stream_id = ctx.stream_id,
                                request_id = %ctx.request_id,
                                grpc_path = %ctx.grpc_path,
                                tonic_code = ?status.code(),
                                error = %status.message(),
                                "Envoy ext_proc stream terminated with gRPC transport error"
                            );
                            break;
                        }
                    }
                }
                svc.metrics.streams_active.fetch_sub(1, Ordering::Relaxed);
            }
            .instrument(span),
        );

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}
