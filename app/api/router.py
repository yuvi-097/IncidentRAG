from fastapi import APIRouter

from app.api.routes import agent, health, incidents, insights

API_PREFIX = "/api"

api_router = APIRouter(prefix=API_PREFIX)
api_router.include_router(health.router)
api_router.include_router(agent.router)
api_router.include_router(incidents.router)
api_router.include_router(insights.router)
