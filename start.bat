@echo off
ECHO Starting Face Recognition System...
ECHO.
ECHO This will build and start all required services.
ECHO.

REM Check if Docker is running
docker info > NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    ECHO Docker is not running. Please start Docker Desktop first.
    EXIT /B 1
)

REM Build and start the services
ECHO Building and starting services...
docker-compose build
docker-compose up -d

ECHO.
ECHO Services are starting up...
ECHO.
ECHO Wait a moment and then access the API at: http://localhost:8000/docs
ECHO.
ECHO To check logs: docker-compose logs -f
ECHO To stop services: docker-compose down
ECHO.

REM Open the API documentation in browser
START http://localhost:8000/docs

EXIT /B 0
