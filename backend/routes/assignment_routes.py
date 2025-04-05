# routes/assignment_routes.py

from flask import Blueprint, request, jsonify
import random
import string
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
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
    using Ollama LLM
    """
    # Initialize the Ollama LLM
    llm = Ollama(model="llama3.2")
    
    # Create prompt for Ollama
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an educational assistant that creates assignments."),
        ("user", f"""
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
        """)
    ])
    
    try:
        # Generate the assignment using Ollama
        chain = prompt_template | llm
        response = chain.invoke({})
        
        # Extract and parse the JSON response
        assignment_json = response
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
        print(f"Error generating assignment with Ollama: {e}")
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
            "submissions": {},
            "allocated_to": []  # Track students allocated to this assignment
        }
        
        return jsonify({
            "assignment_id": assignment_id,
            "title": assignment_data["title"],
            "questions": [{"id": q["id"], "question": q["question"]} for q in assignment_data["questions"]]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@assignment_bp.route('/allocate', methods=['POST'])
def allocate_assignment():
    """Allocate an assignment to a student"""
    data = request.json
    
    # Check required fields
    if 'assignment_id' not in data or 'user_id' not in data:
        return jsonify({"error": "Missing assignment_id or user_id"}), 400
    
    assignment_id = data['assignment_id']
    user_id = data['user_id']
    
    if assignment_id not in assignments:
        return jsonify({"error": "Assignment not found"}), 404
    
    # Add user to allocated list if not already there
    if user_id not in assignments[assignment_id]["allocated_to"]:
        assignments[assignment_id]["allocated_to"].append(user_id)
        
        # Initialize empty submission record for this user
        if user_id not in assignments[assignment_id]["submissions"]:
            assignments[assignment_id]["submissions"][user_id] = {}
    
    return jsonify({
        "success": True,
        "message": f"Assignment allocated to user {user_id}",
        "assignment_id": assignment_id
    })

@assignment_bp.route('/<assignment_id>', methods=['GET'])
def get_assignment(assignment_id):
    """Get assignment questions without answers"""
    if assignment_id not in assignments:
        return jsonify({"error": "Assignment not found"}), 404
    
    # Check if user is allocated to this assignment
    user_id = request.args.get('user_id')
    if user_id and user_id not in assignments[assignment_id]["allocated_to"]:
        return jsonify({"error": "User not allocated to this assignment"}), 403
    
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
    
    # Get user ID and answer from request
    data = request.json
    if not data or 'user_id' not in data or 'answer' not in data:
        return jsonify({"error": "Missing user_id or answer"}), 400
    
    user_id = data['user_id']
    student_answer = data['answer']
    
    # Check if user is allocated to this assignment
    if user_id not in assignments[assignment_id]["allocated_to"]:
        return jsonify({"error": "User not allocated to this assignment"}), 403
    
    assignment = assignments[assignment_id]
    
    # Find the question
    question = None
    for q in assignment["details"]["questions"]:
        if q["id"] == question_id:
            question = q
            break
    
    if not question:
        return jsonify({"error": "Question not found"}), 404
    
    # Evaluate the answer
    correct_answer = question["answer"]
    evaluation = evaluate_answer(student_answer, correct_answer)
    
    # Store the submission
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

@assignment_bp.route('/<assignment_id>/result', methods=['GET'])
def get_question_results(assignment_id):
    """Get results for individual questions in the assignment for a user"""
    if assignment_id not in assignments:
        return jsonify({"error": "Assignment not found"}), 404
    
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400
    
    # Check if user is allocated to this assignment
    if user_id not in assignments[assignment_id]["allocated_to"]:
        return jsonify({"error": "User not allocated to this assignment"}), 403
    
    assignment = assignments[assignment_id]
    
    if user_id not in assignment.get("submissions", {}):
        return jsonify({
            "user_id": user_id,
            "assignment_id": assignment_id,
            "title": assignment["details"]["title"],
            "question_results": [],
            "message": "No submissions found for this user"
        }), 200
    
    submissions = assignment["submissions"][user_id]
    question_results = []
    
    # Get results for each question
    for q in assignment["details"]["questions"]:
        q_id = q["id"]
        if q_id in submissions:
            question_results.append({
                "question_id": q_id,
                "question": q["question"],
                "student_answer": submissions[q_id]["answer"],
                "score": submissions[q_id]["evaluation"]["score"],
                "feedback": submissions[q_id]["evaluation"]["feedback"],
                "correct_answer": q["answer"]
            })
        else:
            question_results.append({
                "question_id": q_id,
                "question": q["question"],
                "student_answer": "",
                "score": 0,
                "feedback": "Not attempted",
                "correct_answer": None  # Don't show correct answer for unattempted questions
            })
    
    return jsonify({
        "user_id": user_id,
        "assignment_id": assignment_id,
        "title": assignment["details"]["title"],
        "question_results": question_results
    })

@assignment_bp.route('/<assignment_id>/score', methods=['GET'])
def get_score(assignment_id):
    """Get total score and overall feedback for the assignment"""
    if assignment_id not in assignments:
        return jsonify({"error": "Assignment not found"}), 404
    
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400
    
    # Check if user is allocated to this assignment
    if user_id not in assignments[assignment_id]["allocated_to"]:
        return jsonify({"error": "User not allocated to this assignment"}), 403
    
    assignment = assignments[assignment_id]
    
    if user_id not in assignment.get("submissions", {}):
        return jsonify({
            "user_id": user_id,
            "assignment_id": assignment_id,
            "total_score": 0,
            "max_score": len(assignment["details"]["questions"]) * 3,
            "percentage": 0,
            "overall_feedback": "No questions attempted",
            "attempted_questions": 0,
            "total_questions": len(assignment["details"]["questions"])
        }), 200
    
    submissions = assignment["submissions"][user_id]
    
    # Calculate total score and gather metrics
    total_score = 0
    max_score = len(assignment["details"]["questions"]) * 3
    attempted_questions = 0
    question_scores = []
    
    for q in assignment["details"]["questions"]:
        q_id = q["id"]
        if q_id in submissions:
            attempted_questions += 1
            score = submissions[q_id]["evaluation"]["score"]
            total_score += score
            question_scores.append({
                "question_id": q_id,
                "score": score
            })
        else:
            question_scores.append({
                "question_id": q_id,
                "score": 0
            })
    
    # Calculate percentage
    percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0
    
    # Generate overall feedback based on performance
    if percentage >= 90:
        overall_feedback = "Outstanding! You have demonstrated excellent understanding of the subject matter."
    elif percentage >= 80:
        overall_feedback = "Great job! You show a strong grasp of most concepts."
    elif percentage >= 70:
        overall_feedback = "Good work! You understand the major concepts but could improve in some areas."
    elif percentage >= 60:
        overall_feedback = "Satisfactory. You've understood the basics but need to work on deeper understanding."
    elif percentage >= 50:
        overall_feedback = "You've passed, but consider reviewing the material to strengthen your understanding."
    else:
        overall_feedback = "More study is needed. Please review the material and consider seeking additional help."
    
    return jsonify({
        "user_id": user_id,
        "assignment_id": assignment_id,
        "title": assignment["details"]["title"],
        "total_score": round(total_score, 1),
        "max_score": max_score,
        "percentage": percentage,
        "overall_feedback": overall_feedback,
        "attempted_questions": attempted_questions,
        "total_questions": len(assignment["details"]["questions"]),
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