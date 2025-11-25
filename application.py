"""
WSGI entry point for AWS Elastic Beanstalk
"""
from src.api import app

# Elastic Beanstalk looks for 'application' variable
application = app

if __name__ == "__main__":
    application.run()


