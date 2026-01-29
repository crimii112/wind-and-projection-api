from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
from getWindData import download_and_convert
from getWrfTmpData import convert_tmp_nc_to_json
from getWrfPollData import convert_poll_nc_to_json
from projectionTest import get_projection_test_data
from projectionTestLcc import get_projection_test_lcc_data
from projectionTestUtm import get_projection_test_utm_data
from projectionTestUtm import get_projection_test_utm_ol_wind
from markerTestLcc import get_marker_test_lcc_data
from markerTestLcc import get_sido_shp
from markerTestLccLayer import get_marker_test_lcc_layer_data
from markerTestLccEarth import get_earth_data

app = Flask(__name__, static_folder="static", static_url_path="/static")

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

@app.route('/api/lcc', methods=['POST'])
def get_lcc_test():
    try:
        body = request.get_json()
        bg_poll = body.get('bgPoll')
        arrow_gap = body.get('arrowGap')
        
        result = get_projection_test_lcc_data(bg_poll, int(arrow_gap))
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/utm', methods=['GET'])
def get_utm_test():
    try:
        result = get_projection_test_utm_data()
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/utm/olwind', methods=['GET'])
def get_utm_ol_wind_test():
    try:
        result = get_projection_test_utm_ol_wind()
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/marker/lcc', methods=['POST'])
def get_marker_lcc_test():
    try:
        body = request.get_json()
        grid_km = body.get('gridKm')
        layer = body.get('layer')
        tstep = body.get('tstep')
        bg_poll = body.get('bgPoll')
        arrow_gap = body.get('arrowGap')
        
        result = get_marker_test_lcc_data(grid_km, layer, tstep, bg_poll, int(arrow_gap))
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/marker/lcc/layer', methods=['POST'])
def get_marker_lcc_layer_test():
    try:
        body = request.get_json()
        grid_km = body.get('gridKm')
        layer = body.get('layer')
        tstep = body.get('tstep')
        bg_poll = body.get('bgPoll')
        arrow_gap = body.get('arrowGap')
        
        result = get_marker_test_lcc_layer_data(grid_km, layer, tstep, bg_poll, int(arrow_gap))
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/marker/sidoshp', methods=['POST'])
def get_sido_shp_test():
    try:
        result = get_sido_shp()
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/api/marker/earth', methods=['POST'])
def get_earth_test():
    try:
        body = request.get_json()
        grid_km = body.get('gridKm')
        layer = body.get('layer')
        tstep = body.get('tstep')
        bg_poll = body.get('bgPoll')
        result = get_earth_data(grid_km, tstep, layer, bg_poll)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
    serve(app, host="0.0.0.0", port=5000)
    