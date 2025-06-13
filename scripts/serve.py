#!/usr/bin/env python3
"""
WronAI API Server
Production-ready FastAPI server for Polish language model inference.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import torch

from wronai import load_model, generate_text
from wronai.inference import InferenceEngine, InferenceConfig
from wronai.utils.logging import setup_logging, get_logger
from wronai.utils.memory import get_memory_usage, memory_monitor

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Global variables
inference_engine: Optional[InferenceEngine] = None
security = HTTPBearer(auto_error=False)

# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation", min_length=1, max_length=2000)
    max_length: Optional[int] = Field(256, description="Maximum generation length", ge=1, le=1000)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter", ge=0.1, le=1.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter", ge=1, le=100)
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty", ge=1.0, le=2.0)
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str = Field(..., description="Generated text")
    generation_time: float = Field(..., description="Generation time in seconds")
    prompt_tokens: int = Field(..., description="Number of input tokens")
    generated_tokens: int = Field(..., description="Number of generated tokens")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class ChatRequest(BaseModel):
    """Request model for chat."""
    message: str = Field(..., description="User message", min_length=1, max_length=1000)
    conversation_history: Optional[List[Dict[str, str]]] = Field([], description="Previous conversation")
    max_length: Optional[int] = Field(256, description="Maximum response length", ge=1, le=1000)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)

class ChatResponse(BaseModel):
    """Response model for chat."""
    response: str = Field(..., description="Model response")
    generation_time: float = Field(..., description="Generation time in seconds")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage information")
    uptime: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="WronAI version")

class StatsResponse(BaseModel):
    """Response model for statistics."""
    total_requests: int = Field(..., description="Total number of requests")
    total_generations: int = Field(..., description="Total number of generations")
    total_tokens: int = Field(..., description="Total tokens generated")
    average_generation_time: float = Field(..., description="Average generation time")
    memory_usage: Dict[str, float] = Field(..., description="Current memory usage")
    model_info: Dict[str, Any] = Field(..., description="Model information")

# Application state
app_state = {
    "start_time": time.time(),
    "total_requests": 0,
    "total_generations": 0,
    "total_tokens": 0,
    "total_generation_time": 0.0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting WronAI API server...")
    
    # Load model
    model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
    quantize = os.getenv("QUANTIZE", "true").lower() == "true"
    
    try:
        logger.info(f"Loading model: {model_name}")
        model = load_model(model_name, quantize=quantize)
        
        config = InferenceConfig(
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            use_polish_formatting=True
        )
        
        global inference_engine
        inference_engine = InferenceEngine(model, config)
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down WronAI API server...")

# Create FastAPI app
app = FastAPI(
    title="WronAI API",
    description="Production API for Polish language model inference",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting
request_counts = {}

async def rate_limit_check(request: Request):
    """Simple rate limiting."""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    for ip in list(request_counts.keys()):
        request_counts[ip] = [
            timestamp for timestamp in request_counts[ip]
            if current_time - timestamp < 3600  # 1 hour window
        ]
        if not request_counts[ip]:
            del request_counts[ip]
    
    # Check rate limit
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    request_counts[client_ip].append(current_time)
    
    # Rate limit: 100 requests per hour
    rate_limit = int(os.getenv("RATE_LIMIT", "100"))
    if len(request_counts[client_ip]) > rate_limit:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

# Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    if not credentials:
        return None
    
    expected_token = os.getenv("API_TOKEN")
    if expected_token and credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    
    return credentials.credentials

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    memory_info = get_memory_usage()
    uptime = time.time() - app_state["start_time"]
    
    return HealthResponse(
        status="healthy" if inference_engine else "unhealthy",
        model_loaded=inference_engine is not None,
        memory_usage=memory_info,
        uptime=uptime,
        version="0.1.0"
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats(token: str = Depends(verify_token)):
    """Get API statistics."""
    memory_info = get_memory_usage()
    
    avg_time = (app_state["total_generation_time"] / 
               max(app_state["total_generations"], 1))
    
    model_info = {}
    if inference_engine:
        stats = inference_engine.get_stats()
        model_info = stats.get("model_info", {})
    
    return StatsResponse(
        total_requests=app_state["total_requests"],
        total_generations=app_state["total_generations"],
        total_tokens=app_state["total_tokens"],
        average_generation_time=avg_time,
        memory_usage=memory_info,
        model_info=model_info
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text_endpoint(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(rate_limit_check),
    token: str = Depends(verify_token)
):
    """Generate text from prompt."""
    if not inference_engine:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )
    
    start_time = time.time()
    
    try:
        # Update inference config
        config = InferenceConfig(
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample
        )
        
        # Set random seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
        
        # Generate text
        generated_text = inference_engine.generate(request.prompt, config)
        
        generation_time = time.time() - start_time
        
        # Count tokens (rough estimate)
        prompt_tokens = len(request.prompt.split())
        generated_tokens = len(generated_text.split())
        
        # Update statistics
        background_tasks.add_task(update_stats, generation_time, generated_tokens)
        
        # Get model info
        model_info = inference_engine.get_stats().get("model_info", {})
        
        return GenerationResponse(
            generated_text=generated_text,
            generation_time=generation_time,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(rate_limit_check),
    token: str = Depends(verify_token)
):
    """Chat with the model."""
    if not inference_engine:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )
    
    start_time = time.time()
    
    try:
        # Generate response
        response = inference_engine.chat(
            request.message,
            conversation_history=request.conversation_history,
            config=InferenceConfig(
                max_length=request.max_length,
                temperature=request.temperature
            )
        )
        
        generation_time = time.time() - start_time
        
        # Update statistics
        generated_tokens = len(response.split())
        background_tasks.add_task(update_stats, generation_time, generated_tokens)
        
        return ChatResponse(
            response=response,
            generation_time=generation_time,
            conversation_id=None  # TODO: Implement conversation tracking
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

@app.post("/generate/stream")
async def generate_stream_endpoint(
    request: GenerationRequest,
    _: None = Depends(rate_limit_check),
    token: str = Depends(verify_token)
):
    """Stream text generation (TODO: Implement streaming)."""
    # TODO: Implement streaming generation
    raise HTTPException(
        status_code=501,
        detail="Streaming generation not yet implemented"
    )

@app.get("/models")
async def list_models(token: str = Depends(verify_token)):
    """List available models."""
    if not inference_engine:
        return {"models": [], "loaded": None}
    
    stats = inference_engine.get_stats()
    model_info = stats.get("model_info", {})
    
    return {
        "models": [model_info.get("model_name", "unknown")],
        "loaded": model_info.get("model_name", "unknown"),
        "info": model_info
    }

# Background tasks
async def update_stats(generation_time: float, tokens_generated: int):
    """Update application statistics."""
    app_state["total_requests"] += 1
    app_state["total_generations"] += 1
    app_state["total_tokens"] += tokens_generated
    app_state["total_generation_time"] += generation_time

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

def main():
    """Main server function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WronAI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model to load")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_NAME"] = args.model
    os.environ["QUANTIZE"] = str(args.quantize).lower()
    
    # Run server
    uvicorn.run(
        "scripts.serve:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()