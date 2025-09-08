import os
from uvicorn import run


def main() -> None:
    workers = int(os.getenv("WORKERS", "1"))
    run(
        "src.backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        factory=False,
        workers=workers,
    )


if __name__ == "__main__":
    main()
