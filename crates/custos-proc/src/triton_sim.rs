use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::{Buf, BufMut, Bytes};
use custos_core::{ScannerConfig, inspect_raw_protobuf_with_report};
use tokio::sync::Semaphore;
use tonic::body::Body;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::server::NamedService;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use tower::Service;

#[derive(Debug, Default, Clone, Copy)]
struct RawCodec;

#[derive(Debug, Default, Clone, Copy)]
struct RawEncoder;

#[derive(Debug, Default, Clone, Copy)]
struct RawDecoder;

impl Codec for RawCodec {
    type Encode = Bytes;
    type Decode = Bytes;
    type Encoder = RawEncoder;
    type Decoder = RawDecoder;

    fn encoder(&mut self) -> Self::Encoder {
        RawEncoder
    }

    fn decoder(&mut self) -> Self::Decoder {
        RawDecoder
    }
}

impl Encoder for RawEncoder {
    type Item = Bytes;
    type Error = Status;

    fn encode(&mut self, item: Self::Item, dst: &mut EncodeBuf<'_>) -> Result<(), Self::Error> {
        dst.reserve(item.len());
        dst.put_slice(&item);
        Ok(())
    }
}

impl Decoder for RawDecoder {
    type Item = Bytes;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Self::Item>, Self::Error> {
        let len = src.remaining();
        Ok(Some(src.copy_to_bytes(len)))
    }
}

#[derive(Clone)]
struct KServeGrpcService {
    response_payload: Bytes,
    worker_pool: Arc<Semaphore>,
}

impl NamedService for KServeGrpcService {
    const NAME: &'static str = "inference.GRPCInferenceService";
}

impl Service<http::Request<Body>> for KServeGrpcService {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, mut req: http::Request<Body>) -> Self::Future {
        req.headers_mut().remove("grpc-encoding");
        let resp_bytes = self.response_payload.clone();
        let worker_pool = Arc::clone(&self.worker_pool);
        Box::pin(async move {
            match req.uri().path() {
                "/inference.GRPCInferenceService/ModelInfer" => {
                    let mut grpc = tonic::server::Grpc::new(RawCodec)
                        .max_decoding_message_size(32 * 1024 * 1024)
                        .max_encoding_message_size(32 * 1024 * 1024);
                    let svc = ModelInferMethod { resp_bytes, worker_pool };
                    let res = grpc.unary(svc, req).await;
                    Ok(res)
                }
                _ => {
                    let status = Status::unimplemented("Method not found");
                    Ok(status.into_http())
                }
            }
        })
    }
}

struct ModelInferMethod {
    resp_bytes: Bytes,
    worker_pool: Arc<Semaphore>,
}

impl tonic::server::UnaryService<Bytes> for ModelInferMethod {
    type Response = Bytes;
    type Future = Pin<Box<dyn Future<Output = Result<Response<Self::Response>, Status>> + Send>>;

    fn call(&mut self, request: Request<Bytes>) -> Self::Future {
        let payload = request.into_inner();
        let resp_bytes = self.resp_bytes.clone();
        let worker_pool = Arc::clone(&self.worker_pool);
        Box::pin(async move {
            let _permit = worker_pool.acquire().await.unwrap();
            let config = ScannerConfig::default();
            match inspect_raw_protobuf_with_report(&payload, &config) {
                Ok(_report) => {
                    Ok(Response::new(resp_bytes))
                }
                Err(failure) => {
                    // Unprotected backend worker slot stalled by malformed Protobuf / tensor exploit
                    tokio::time::sleep(std::time::Duration::from_millis(35)).await;
                    Err(Status::internal(format!(
                        "tritonserver runtime exception at byte {}: {:?}",
                        failure.byte_offset, failure.error
                    )))
                }
            }
        })
    }
}

fn build_kserve_response() -> Bytes {
    let mut out = Vec::with_capacity(4064);
    out.extend_from_slice(b"\x0a\x08resnet50\x12\x011");
    out.extend_from_slice(&[0x32, 0xA0, 0x1F]);
    out.resize(out.len() + 4000, 0u8);
    Bytes::from(out)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut port: u16 = 9001;
    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--grpc-port=") {
            if let Ok(p) = val.parse::<u16>() {
                port = p;
            }
        }
    }
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    eprintln!("tritonserver KServe v2 gRPC listening on {}", addr);
    let svc = KServeGrpcService {
        response_payload: build_kserve_response(),
        worker_pool: Arc::new(Semaphore::new(32)),
    };
    Server::builder()
        .max_concurrent_streams(1024)
        .initial_stream_window_size(16 * 1024 * 1024)
        .initial_connection_window_size(32 * 1024 * 1024)
        .add_service(svc)
        .serve(addr)
        .await?;
    Ok(())
}
