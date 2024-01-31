from flask import Flask, jsonify, make_response, request
from os import environ as env

# Initialization of application
app = Flask(__name__)

def _get_fields_f_json(request):
    entry_json = request.json
    return  entry_json["id"], entry_json["entity"]

@app.route('/question', methods=["POST", "GET"])
def question_processing():
    user_id, question_entity = _get_fields_f_json(request)

    #result = your_app_session_f_questions_function(question_entity)
    result = "Question processing"

    return make_response(jsonify({'response': result, "id": user_id}))

@app.route('/smthelse', methods=['POST'])
def smthelse_processing():
    user_id, question_entity = _get_fields_f_json(request)

    #result = your_app_session_f_smthelse(question_entity)
    result = "Other version"

    return make_response(jsonify({'response': result, "id": user_id}))


@app.route('/', methods=['GET'])
def welcome():
    return make_response(jsonify({'welcomeMessage': "Welcome to Flask!"}))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=env.get("PORT", 3000))