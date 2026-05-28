from fastapi import APIRouter
from api.v1.rag_routes import rag_router
from api.v1.epistemic_routes import epistemic_router



api_router = APIRouter(prefix="/rag", tags=["rag"])

# Include all routers
api_router.include_router(rag_router)
api_router.include_router(epistemic_router)
