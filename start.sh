#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
uvicorn app:app --host 0.0.0.0 --port $PORT