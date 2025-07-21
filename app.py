from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
from getWindData import download_and_convert
from getWrfTmpData import convert_tmp_nc_to_json
from getWrfPollData import convert_poll_nc_to_json
from projectionTest import get_projection_test_data

app = Flask(__name__)

CORS(app)

@app.route('/api/wind/test', methods=['POST'])
def get_wind_test():
    try:
        body = request.get_json()
        date = body.get('date')
        time = body.get('time')
        print(f"📡 Requesting wind data for {date} {time}Z")

        result = download_and_convert(date, time)
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/wind', methods=['POST'])
def get_wind():
    try:
        body = request.get_json()
        option = body.get('option')
        wind_gap = body.get('windGap')
        tstep = body.get('tstep')
        print(f"📡 Requesting wind data for {option}, {wind_gap}, {tstep}")
        
        # if option == 'tmp':
        #     result = convert_tmp_nc_to_json()
        # else:
        result = convert_poll_nc_to_json(option, int(wind_gap), int(tstep))
            
        # print(result)
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/proj/test', methods=['GET'])
def get_proj_test():
    try:
        result = get_projection_test_data()
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)
    