from flask import Flask, request, jsonify

from llm_service import feed_llm

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/update_knowledge")
def update_knowledge():
    return "Knowledge updated!"


@app.route("/get_prediction", methods=['POST'])
def get_prediction():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        prompt = f"""Analyze historical demand patterns and predict tomorrow's demand considering:
        - Initial prediction: {data.get('demand')}
        - Date: {data.get('date')}
        - Day: {data.get('day_of_week')}
        - Special day: {data.get('is_special_day')}
        - Weather: {data.get('weather')}

        Consider seasonality, weather impact, and special day effects. Return only the final predicted demand value."""

        response = feed_llm(prompt)
        return jsonify({"prediction": response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
    # res = feed_llm("""According to demand of previous days,
    #     initial demand predictor predicted that the tomorrow's demand will be {req.demand}.
    #        Predicted values for tomorrow as follows:
    #                "date":{req.date},
    #                "day_of_week":{req.day_of_week},
    #                "is_special_day":{req.is_special_day},
    #                "whether":{req.whether}
    #
    #      Use given historical data to predict the 'demand' of tomorrow.""")
    #
    # print(res.content)