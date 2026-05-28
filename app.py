from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.startup import initialize_application, shutdown_application
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    await initialize_application()
    yield
    # Shutdown
    await shutdown_application()

# Create FastAPI app with lifespan
app = FastAPI(
    title="RAG Agent API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # For development. Later you can restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers AFTER lifespan definition
from api.v1 import api_router
app.include_router(api_router)
