# syntax=docker/dockerfile:1

FROM rust:1.88-slim-bookworm AS builder
WORKDIR /usr/src/custos

COPY Cargo.toml ./
COPY proto ./proto
COPY crates ./crates

RUN cargo build --release --workspace && \
    mkdir -p /var/run/custos && \
    chown 10001:10001 /var/run/custos

FROM debian:bookworm-slim AS runtime
RUN groupadd -g 10001 custos && \
    useradd -u 10001 -g 10001 -s /usr/sbin/nologin -M custos && \
    mkdir -p /var/run/custos && \
    chown 10001:10001 /var/run/custos

COPY --from=builder /usr/src/custos/target/release/custos-proc /usr/local/bin/custos-proc

USER 10001:10001
ENTRYPOINT ["/usr/local/bin/custos-proc"]
CMD ["--socket-path=/var/run/custos/ext_proc.sock", "--max-nesting-depth=16", "--max-tensor-elements=16777216", "--max-message-size-mb=32"]
