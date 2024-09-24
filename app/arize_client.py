import logging
import os

# Import the automatic instrumentor from OpenInference
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = logging.getLogger("dev_logger")

ARIZE_SPACE_ID = os.environ["ARIZE_SPACE_ID"]
ARIZE_API_KEY = os.environ["ARIZE_API_KEY"]
TRACING_PROJECT_NAME = "serefine-chatbot-tracing"
MODEL_VERSION = os.environ["ARIZE_TRACING_ENV"]
ARIZE_ENDPOINT = "https://otlp.arize.com/v1"


def setup_arize_client():
    """
    This function sets up the Arize client to send OpenTelemetry traces to Arize.
    It sets the OTEL_EXPORTER_OTLP_TRACES_HEADERS environment variable with the Arize API key and space key,
    creates an OTLPSpanExporter, a SimpleSpanProcessor, and a TracerProvider with the resource,
    adds the span processor to the tracer provider, sets the tracer provider as the global OpenTelemetry tracer provider,
    and instruments the FastAPI app with the LangChainInstrumentor.
    """
    # Set the Space and API keys as headers
    os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = (
        f"space_id={ARIZE_SPACE_ID},api_key={ARIZE_API_KEY}"
    )
    # Set the model id and version as resource attributes
    resource = Resource(
        attributes={
            "model_id": TRACING_PROJECT_NAME,
            "model_version": MODEL_VERSION,
        }
    )
    span_exporter = OTLPSpanExporter(endpoint=ARIZE_ENDPOINT)
    span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    tracer_provider.add_span_processor(span_processor=span_processor)
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument()

    logger.info("✅ Import and Setup Arize Client Done! Now we can start using Arize!")
