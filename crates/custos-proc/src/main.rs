pub mod service;

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use custos_core::{DEFAULT_MAX_TENSOR_ELEMENTS, MAX_DEPTH, ScannerConfig};
use envoy_types::pb::envoy::service::ext_proc::v3::external_processor_server::ExternalProcessorServer;
use service::{CustosExtProcService, ServiceMetrics};
use tokio::net::UnixListener;
use tokio::signal::unix::{SignalKind, signal};
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Server;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

/// Structured output format for cluster log aggregation pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    /// Newline-delimited JSON objects (ideal for Kubernetes FluentBit, Loki, Datadog, or `jq`).
    Json,
    /// Single-line key=value human-readable logs.
    Compact,
    /// Multi-line pretty logs for local interactive debugging.
    Pretty,
}

/// Unprivileged Layer 7 Envoy `ext_proc` admission control daemon for KServe v2 gRPC inference.
#[derive(Debug, Parser)]
#[command(name = "custos-proc", version, about)]
pub struct CliArgs {
    /// Path to the Unix Domain Socket shared with Envoy Proxy.
    #[arg(
        long,
        env = "CUSTOS_SOCKET_PATH",
        default_value = "/var/run/custos/ext_proc.sock"
    )]
    pub socket_path: PathBuf,

    /// Maximum allowed Protobuf submessage nesting depth (1..=16).
    #[arg(long, env = "CUSTOS_MAX_NESTING_DEPTH", default_value_t = MAX_DEPTH)]
    pub max_nesting_depth: usize,

    /// Maximum allowed tensor shape product (elements) per input tensor.
    #[arg(
        long,
        env = "CUSTOS_MAX_TENSOR_ELEMENTS",
        default_value_t = DEFAULT_MAX_TENSOR_ELEMENTS
    )]
    pub max_tensor_elements: i64,

    /// Maximum gRPC message decoding size in MiB.
    #[arg(long, env = "CUSTOS_MAX_MESSAGE_SIZE_MB", default_value_t = 32)]
    pub max_message_size_mb: usize,

    /// Log output format (`json`, `compact`, or `pretty`).
    #[arg(long, env = "CUSTOS_LOG_FORMAT", value_enum, default_value_t = LogFormat::Json)]
    pub log_format: LogFormat,

    /// Default log filter directive when `RUST_LOG` is not set (e.g. `info`, `debug`, `custos_proc=debug`).
    #[arg(long, env = "CUSTOS_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Emit `INFO`-level structured telemetry logs for admitted requests (in addition to `WARN` on rejections).
    #[arg(
        long,
        env = "CUSTOS_LOG_ADMITTED",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    pub log_admitted: bool,

    /// Interval in seconds between periodic `telemetry_summary` heartbeat logs (`0` disables periodic logs).
    #[arg(long, env = "CUSTOS_STATS_INTERVAL_SECS", default_value_t = 60)]
    pub stats_interval_secs: u64,
}

fn init_logging(args: &CliArgs) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(args.log_level.as_str()));

    match args.log_format {
        LogFormat::Json => {
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(env_filter)
                .with_current_span(true)
                .with_target(true)
                .with_thread_ids(true)
                .flatten_event(true)
                .init();
        }
        LogFormat::Compact => {
            tracing_subscriber::fmt()
                .compact()
                .with_env_filter(env_filter)
                .with_target(true)
                .init();
        }
        LogFormat::Pretty => {
            tracing_subscriber::fmt()
                .pretty()
                .with_env_filter(env_filter)
                .with_target(true)
                .init();
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();
    init_logging(&args);

    let start_instant = Instant::now();
    let max_message_bytes = args.max_message_size_mb.saturating_mul(1024 * 1024);

    if args.max_nesting_depth > MAX_DEPTH {
        warn!(
            event = "config_clamped",
            requested_max_nesting_depth = args.max_nesting_depth,
            compile_time_max_depth = MAX_DEPTH,
            "requested max_nesting_depth exceeds stack frame capacity; clamping to MAX_DEPTH"
        );
    }

    let scanner_config = ScannerConfig {
        max_nesting_depth: args.max_nesting_depth.min(MAX_DEPTH),
        max_tensor_elements: args.max_tensor_elements,
    };

    if let Some(parent) = args.socket_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    // Remove stale Unix Domain Socket file if present before binding.
    if args.socket_path.exists() {
        info!(
            event = "stale_socket_removed",
            socket_path = %args.socket_path.display(),
            "removing preexisting Unix Domain Socket file prior to bind"
        );
        std::fs::remove_file(&args.socket_path)?;
    }

    let uds = UnixListener::bind(&args.socket_path)?;
    std::fs::set_permissions(&args.socket_path, std::fs::Permissions::from_mode(0o660))?;

    info!(
        event = "daemon_started",
        version = env!("CARGO_PKG_VERSION"),
        pid = std::process::id(),
        socket_path = %args.socket_path.display(),
        socket_mode = "0660",
        max_nesting_depth = scanner_config.max_nesting_depth,
        max_tensor_elements = scanner_config.max_tensor_elements,
        max_message_size_mb = args.max_message_size_mb,
        max_message_bytes = max_message_bytes,
        log_format = ?args.log_format,
        log_level = %args.log_level,
        log_admitted = args.log_admitted,
        stats_interval_secs = args.stats_interval_secs,
        "custos-proc ext_proc daemon listening on Unix Domain Socket"
    );

    let metrics = Arc::new(ServiceMetrics::new());
    if args.stats_interval_secs > 0 {
        let periodic_metrics = Arc::clone(&metrics);
        let interval_secs = args.stats_interval_secs;
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
            // Skip immediate first tick at t=0
            ticker.tick().await;
            loop {
                ticker.tick().await;
                periodic_metrics.emit_summary_log("periodic_heartbeat", start_instant.elapsed().as_secs());
            }
        });
    }

    let uds_stream = UnixListenerStream::new(uds);
    let service = CustosExtProcService::with_telemetry(
        scanner_config,
        args.log_admitted,
        Arc::clone(&metrics),
    );
    let ext_proc_svc = ExternalProcessorServer::new(service)
        .max_decoding_message_size(max_message_bytes)
        .max_encoding_message_size(max_message_bytes);

    let socket_path_cleanup = args.socket_path.clone();
    Server::builder()
        .add_service(ext_proc_svc)
        .serve_with_incoming_shutdown(uds_stream, shutdown_signal())
        .await?;

    metrics.emit_summary_log("graceful_shutdown", start_instant.elapsed().as_secs());

    if socket_path_cleanup.exists() {
        let _ = std::fs::remove_file(&socket_path_cleanup);
    }
    info!(
        event = "daemon_stopped",
        uptime_secs = start_instant.elapsed().as_secs(),
        socket_path = %socket_path_cleanup.display(),
        "custos-proc daemon shut down cleanly"
    );
    Ok(())
}

async fn shutdown_signal() {
    let mut sigterm = signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!(event = "shutdown_signal_received", signal = "SIGINT", "received SIGINT, initiating graceful shutdown");
        }
        _ = sigterm.recv() => {
            info!(event = "shutdown_signal_received", signal = "SIGTERM", "received SIGTERM, initiating graceful shutdown");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    use envoy_types::pb::envoy::config::core::v3::{HeaderMap, HeaderValue};
    use envoy_types::pb::envoy::r#type::v3::StatusCode;
    use envoy_types::pb::envoy::service::ext_proc::v3::common_response::ResponseStatus;
    use envoy_types::pb::envoy::service::ext_proc::v3::external_processor_client::ExternalProcessorClient;
    use envoy_types::pb::envoy::service::ext_proc::v3::processing_request::Request as EnvoyRequest;
    use envoy_types::pb::envoy::service::ext_proc::v3::processing_response::Response as EnvoyResponse;
    use envoy_types::pb::envoy::service::ext_proc::v3::{HttpBody, HttpHeaders, ProcessingRequest};
    use hyper_util::rt::TokioIo;
    use tokio::net::UnixStream;
    use tonic::transport::{Endpoint, Uri};
    use tower::service_fn;

    #[tokio::test]
    async fn test_uds_ext_proc_stream_admit_and_reject_with_telemetry() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let sock_path = tmp_dir.path().join("ext_proc.sock");

        let uds = UnixListener::bind(&sock_path).expect("bind UDS");
        std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o660))
            .expect("chmod 0660");

        let mode = std::fs::metadata(&sock_path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o660);

        let metrics = Arc::new(ServiceMetrics::new());
        let custos_svc = CustosExtProcService::with_telemetry(
            ScannerConfig::default(),
            true,
            Arc::clone(&metrics),
        );
        let svc = ExternalProcessorServer::new(custos_svc)
            .max_decoding_message_size(32 * 1024 * 1024);

        let server_handle = tokio::spawn(async move {
            Server::builder()
                .add_service(svc)
                .serve_with_incoming(UnixListenerStream::new(uds))
                .await
                .expect("server failed");
        });

        let client_sock = sock_path.clone();
        let channel = Endpoint::try_from("http://[::]:50051")
            .expect("endpoint")
            .connect_with_connector(service_fn(move |_: Uri| {
                let path = client_sock.clone();
                async move {
                    let stream = UnixStream::connect(path).await?;
                    Ok::<_, std::io::Error>(TokioIo::new(stream))
                }
            }))
            .await
            .expect("connect UDS");

        let mut client = ExternalProcessorClient::new(channel);

        // 1. Test valid headers (with x-request-id and :path) + valid KServe v2 body -> CONTINUE
        let valid_body = vec![
            0x00, 0x00, 0x00, 0x00, 0x0A, // 5-byte gRPC frame (len = 10)
            0x0A, 0x08, b'r', b'e', b's', b'n', b'e', b't', b'5', b'0',
        ];
        let headers = HttpHeaders {
            headers: Some(HeaderMap {
                headers: vec![
                    HeaderValue {
                        key: "x-request-id".to_owned(),
                        value: String::new(),
                        raw_value: b"req-cluster-001".to_vec(),
                    },
                    HeaderValue {
                        key: ":path".to_owned(),
                        value: "/inference.GRPCInferenceService/ModelInfer".to_owned(),
                        raw_value: Vec::new(),
                    },
                ],
            }),
            end_of_stream: false,
            ..Default::default()
        };
        let requests = vec![
            ProcessingRequest {
                request: Some(EnvoyRequest::RequestHeaders(headers)),
                ..Default::default()
            },
            ProcessingRequest {
                request: Some(EnvoyRequest::RequestBody(HttpBody {
                    body: valid_body,
                    end_of_stream: true,
                    ..Default::default()
                })),
                ..Default::default()
            },
        ];

        let mut resp_stream = client
            .process(tokio_stream::iter(requests))
            .await
            .expect("process RPC")
            .into_inner();

        let first = resp_stream
            .message()
            .await
            .expect("msg 1")
            .expect("some 1");
        assert!(matches!(
            first.response,
            Some(EnvoyResponse::RequestHeaders(ref h))
                if h.response.as_ref().map(|r| r.status) == Some(ResponseStatus::Continue as i32)
        ));

        let second = resp_stream
            .message()
            .await
            .expect("msg 2")
            .expect("some 2");
        assert!(matches!(
            second.response,
            Some(EnvoyResponse::RequestBody(ref b))
                if b.response.as_ref().map(|r| r.status) == Some(ResponseStatus::Continue as i32)
        ));

        // 2. Test malformed varint body -> ImmediateResponse (HTTP 400 + gRPC status 3)
        let mut bad_body = vec![0x00, 0x00, 0x00, 0x00, 0x0C, 0x08];
        bad_body.extend_from_slice(&[0xFF; 11]);
        let bad_requests = vec![ProcessingRequest {
            request: Some(EnvoyRequest::RequestBody(HttpBody {
                body: bad_body,
                end_of_stream: true,
                ..Default::default()
            })),
            ..Default::default()
        }];

        let mut bad_resp_stream = client
            .process(tokio_stream::iter(bad_requests))
            .await
            .expect("process bad RPC")
            .into_inner();

        let immediate = bad_resp_stream
            .message()
            .await
            .expect("msg")
            .expect("some");
        match immediate.response {
            Some(EnvoyResponse::ImmediateResponse(imm)) => {
                assert_eq!(imm.status.map(|s| s.code), Some(StatusCode::BadRequest as i32));
                assert_eq!(imm.grpc_status.map(|g| g.status), Some(3));
                assert_eq!(imm.details, "custos_denied:malformed_varint");
            }
            other => panic!("expected ImmediateResponse, got {other:?}"),
        }

        // Verify telemetry counters updated accurately
        assert_eq!(metrics.headers_evaluated.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.bodies_evaluated.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.bodies_admitted.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.bodies_rejected.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.rejected_malformed_varint.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.bytes_skipped_o1_total.load(Ordering::Relaxed), 8);

        server_handle.abort();
    }
}
