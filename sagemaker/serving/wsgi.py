import predictor as myapp

# Gunicorn entrypoint: `gunicorn wsgi:app` finds this `app`.
app = myapp.app
