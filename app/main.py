from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from app.routers import main, training, analysis
from app.config.settings import STATIC_DIR, TEMPLATES_DIR
from app.utils.logger import logger

# Initialize FastAPI application
app = FastAPI(
    title="PCOS Data Analysis Platform",
    description="Advanced data analytics and visualization platform for PCOS research",
    version="1.0.0"
)

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="your-secret-key-here")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include routers
app.include_router(main.router)
app.include_router(training.router)
app.include_router(analysis.router)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Application shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 