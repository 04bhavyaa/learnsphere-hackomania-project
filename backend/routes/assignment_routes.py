# routes/assignment_routes.py

from flask import Blueprint, request, jsonify
import random
import string
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import os
import uuid
import re

assignment_bp = Blueprint('assignment', __name__)

# Storage for generated assignments
assignments = {}

# Text cleaning utility
def clean_text(text):
    """Basic text cleaning for comparison"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Function to generate assignment based on input
def generate_assignment(course, title, topics, summary):
    """
    Generate a 10-question assignment with answer key based on course details
    """
    import openai
    
    # Use environment variable for the API key in production
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    # Create prompt for OpenAI
    prompt = f"""
    Create a 10-question assignment for a {course} course titled "{title}".
    The assignment should cover these topics: {topics}.
    Class summary: {summary}
    
    Each question should be worth 3 marks and should test understanding of key concepts.
    
    Format the output as a JSON object with this structure:
    {{
        "title": "Assignment title",
        "questions": [
            {{
                "id": 1,
                "question": "Question text",
                "answer": "Detailed answer that would earn full marks"
            }},
            ...and so on for all 10 questions
        ]
    }}
    
    Make sure answers are comprehensive enough to allow for evaluation of student responses.
    """
    
    try:
        # Generate the assignment using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use appropriate model
            messages=[
                {"role": "system", "content": "You are an educational assistant that creates assignments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        # Extract and parse the JSON response
        assignment_json = response.choices[0].message.content
        # Find JSON content within the response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', assignment_json)
        if json_match:
            json_content = json_match.group(1) or json_match.group(2)
            try:
                assignment_data = json.loads(json_content)
            except:
                # If parsing fails, try to clean up the JSON
                json_content = re.sub(r'[\n\r\t]', '', json_content)
                assignment_data = json.loads(json_content)
        else:
            # If no JSON format found, attempt to parse the whole response
            assignment_data = json.loads(assignment_json)
        
        return assignment_data
    
    except Exception as e:
        # Fallback to a simple generated assignment if API fails
        print(f"Error generating assignment with OpenAI: {e}")
        return generate_fallback_assignment(course, title, topics)

def generate_fallback_assignment(course, title, topics):
    """Generate a simple fallback assignment if the API call fails"""
    topic_list = topics.split(',')
    
    questions = []
    for i in range(1, 11):
        topic = random.choice(topic_list).strip()
        questions.append({
            "id": i,
            "question": f"Explain the concept of {topic} in the context of {course}.",
            "answer": f"A comprehensive explanation of {topic} would include its definition, importance in {course}, and practical applications."
        })
    
    return {
        "title": f"Assignment: {title}",
        "questions": questions
    }

def evaluate_answer(student_answer, correct_answer, threshold=0.6):
    """
    Evaluate student answer against the correct answer using cosine similarity
    Returns a score between 0-3 and a feedback comment
    """
    # Clean both texts
    student_text = clean_text(student_answer)
    correct_text = clean_text(correct_answer)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([student_text, correct_text])
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        # Fallback if TF-IDF fails (e.g., empty input)
        similarity = 0.0
    
    # Calculate score (0-3 marks)
    if similarity >= 0.8:
        score = 3
        feedback = "Excellent answer! Comprehensive and accurate."
    elif similarity >= threshold:
        score = 2 + (similarity - threshold) / (0.8 - threshold)
        feedback = "Good answer with some key points covered."
    elif similarity >= 0.3:
        score = similarity / 0.3
        feedback = "Partial answer. Some important elements missing."
    else:
        score = 0
        feedback = "Answer does not address the question adequately."
    
    return {
        "score": round(score, 1),
        "similarity": round(similarity, 2),
        "feedback": feedback
    }

def speech_to_text(audio_file):
    """Convert speech audio file to text"""
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Audio not understood"
        except sr.RequestError:
            return "Could not request results from speech recognition service"

@assignment_bp.route('/create', methods=['POST'])
def create_assignment():
    """Create a new assignment based on course details"""
    data = request.json
    
    # Check required fields
    required_fields = ['course', 'title', 'topics', 'summary']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    try:
        # Generate the assignment
        assignment_data = generate_assignment(
            data['course'], 
            data['title'], 
            data['topics'], 
            data['summary']
        )
        
        # Generate unique ID for this assignment
        assignment_id = str(uuid.uuid4())
        
        # Store assignment
        assignments[assignment_id] = {
            "details": assignment_data,
            "submissions": {}
        }
        
        return jsonify({
            "assignment_id": assignment_id,
            "title": assignment_data["title"],
            "questions": [{"id": q["id"], "question": q["question"]} for q in assignment_data["questions"]]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@assignment_bp.route('/<assignment_id>', methods=['GET'])
def get_assignment(assignment_id):
    """Get assignment questions without answers"""
    if assignment_id not in assignments:
        return jsonify({"error": "Assignment not found"}), 404
    
    assignment = assignments[assignment_id]
    
    return jsonify({
        "assignment_id": assignment_id,
        "title": assignment["details"]["title"],
        "questions": [{"id": q["id"], "question": q["question"]} for q in assignment["details"]["questions"]]
    })

@assignment_bp.route('/<assignment_id>/answer/<int:question_id>', methods=['POST'])
def submit_answer(assignment_id, question_id):
    """Submit answer for a specific question"""
    if assignment_id not in assignments:
        return jsonify({"error": "Assignment not found"}), 404
    
    assignment = assignments[assignment_id]
    
    # Find the question
    question = None
    for q in assignment["details"]["questions"]:
        if q["id"] == question_id:
            question = q
            break
    
    if not question:
        return jsonify({"error": "Question not found"}), 404
    
    # Check request type: text or speech
    if request.json and 'answer' in request.json:
        # Handle text submission
        student_answer = request.json['answer']
    elif request.files and 'audio' in request.files:
        # Handle speech submission
        audio_file = request.files['audio']
        temp_filename = f"temp_{uuid.uuid4()}.wav"
        audio_file.save(temp_filename)
        
        try:
            student_answer = speech_to_text(temp_filename)
            # Remove temp file
            os.remove(temp_filename)
        except Exception as e:
            return jsonify({"error": f"Speech processing error: {str(e)}"}), 500
    else:
        return jsonify({"error": "No answer provided"}), 400
    
    # Evaluate the answer
    correct_answer = question["answer"]
    evaluation = evaluate_answer(student_answer, correct_answer)
    
    # Store the submission
    if "submissions" not in assignment:
        assignment["submissions"] = {}
    
    user_id = request.json.get('user_id', 'anonymous')
    
    if user_id not in assignment["submissions"]:
        assignment["submissions"][user_id] = {}
    
    assignment["submissions"][user_id][question_id] = {
        "answer": student_answer,
        "evaluation": evaluation
    }
    
    return jsonify({
        "question_id": question_id,
        "score": evaluation["score"],
        "feedback": evaluation["feedback"],
        "max_score": 3
    })

@assignment_bp.route('/<assignment_id>/score', methods=['GET'])
def get_score(assignment_id):
    """Get total score for the assignment"""
    if assignment_id not in assignments:
        return jsonify({"error": "Assignment not found"}), 404
    
    user_id = request.args.get('user_id', 'anonymous')
    assignment = assignments[assignment_id]
    
    if user_id not in assignment.get("submissions", {}):
        return jsonify({"error": "No submissions found for this user"}), 404
    
    submissions = assignment["submissions"][user_id]
    
    # Calculate total score
    total_score = 0
    max_score = len(assignment["details"]["questions"]) * 3
    
    question_scores = []
    for q in assignment["details"]["questions"]:
        q_id = q["id"]
        if q_id in submissions:
            score = submissions[q_id]["evaluation"]["score"]
            total_score += score
            question_scores.append({
                "question_id": q_id,
                "score": score,
                "feedback": submissions[q_id]["evaluation"]["feedback"]
            })
        else:
            question_scores.append({
                "question_id": q_id,
                "score": 0,
                "feedback": "No submission"
            })
    
    return jsonify({
        "user_id": user_id,
        "assignment_id": assignment_id,
        "total_score": round(total_score, 1),
        "max_score": max_score,
        "percentage": round((total_score / max_score) * 100, 1),
        "question_scores": question_scores
    })

@assignment_bp.route('/<assignment_id>/answer-key', methods=['GET'])
def get_answer_key(assignment_id):
    """Get the answer key for an assignment (teacher access)"""
    if assignment_id not in assignments:
        return jsonify({"error": "Assignment not found"}), 404
    
    # This should have authentication to ensure only teachers can access
    # For now, we'll just return it
    
    assignment = assignments[assignment_id]
    
    return jsonify({
        "assignment_id": assignment_id,
        "title": assignment["details"]["title"],
        "answer_key": [
            {"id": q["id"], "question": q["question"], "answer": q["answer"]} 
            for q in assignment["details"]["questions"]
        ]
    })