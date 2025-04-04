from flask import Flask
from flask_cors import CORS
from routes.chatbot import chatbot_bp
from routes.proctoring_routes import proctoring_bp
from routes.assignment_routes import assignment_bp

app = Flask(__name__)
CORS(app)

# Register chatbot API route
app.register_blueprint(chatbot_bp)
# Register proctoring routes
app.register_blueprint(proctoring_bp, url_prefix='/proctoring')
# Register assignment routes
app.register_blueprint(assignment_bp, url_prefix='/assignment')

if __name__ == "__main__":
    app.run(debug=True)