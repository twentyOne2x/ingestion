try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional for lightweight local scripts
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
