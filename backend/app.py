from flask import Flask
from flask_cors import CORS
from routes.chatbot import chatbot_bp
from routes.authentication import auth_routes
from routes.proctoring_routes import proctoring_bp

app = Flask(__name__)
CORS(app)

# Register authentication routes
app.register_blueprint(auth_routes)
# Register chatbot API route
app.register_blueprint(chatbot_bp)
# Register proctoring routes
app.register_blueprint(proctoring_bp, url_prefix='/proctoring')

if __name__ == "__main__":
    app.run(debug=True)