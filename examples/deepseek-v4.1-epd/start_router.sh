#!/usr/bin/env bash
# Router: OpenAI-compatible front end that splits each request into a prefill
# call and a decode call across the disaggregated PD pair.
#
# Required environment:
#   PREFILL_IP  prefill host IP
#   DECODE_IP   decode host IP
# Optional:
#   PREFILL_BOOTSTRAP_PORT  prefill disaggregation bootstrap port (default: 8200)
#   ROUTER_PORT             router HTTP port (default: 8101)
set -euo pipefail

: "${PREFILL_IP:?set PREFILL_IP to the prefill host IP}"
: "${DECODE_IP:?set DECODE_IP to the decode host IP}"

exec python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://${PREFILL_IP}:8000" "${PREFILL_BOOTSTRAP_PORT:-8200}" \
  --decode "http://${DECODE_IP}:8000" \
  --policy round_robin \
  --max-concurrent-requests 1024 \
  --request-timeout-secs 6000 \
  --queue-timeout-secs 6000 \
  --queue-size 10000 \
  --disable-circuit-breaker \
  --log-level info \
  --host 0.0.0.0 \
  --port "${ROUTER_PORT:-8101}" \
  --disable-health-check
