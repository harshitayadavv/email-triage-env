from env.server import app
import os


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("env.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()