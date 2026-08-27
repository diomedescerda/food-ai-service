"""Arranque del AI Service en Windows.

uvicorn 0.36+ fuerza ProactorEventLoop, incompatible con psycopg async.
Este runner fuerza SelectorEventLoop para evitar problemas en Windows.
Uso: python run_dev.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("FOOD_AI_SERVICE_PORT", "8010")),
        reload=True,
    )