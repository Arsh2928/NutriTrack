"""Entry point:  python app.py   ->  http://127.0.0.1:5000

Thin wrapper kept so the CA1 command still works. The Flask application
itself lives in nutritrack/web/app.py.
"""
from nutritrack.web.app import main

if __name__ == "__main__":
    main()
