from flask import Flask, send_from_directory, send_file
import os
app = Flask(__name__, 
    static_url_path='/static',
    static_folder='static'
)

@app.route('/')
def serve_index():
    return send_file('index.html')
@app.route('/static/<path:path>')
def serve_static(path):
    try:
        return app.send_static_file(path)
    except Exception as e:
        print(f"Error serving {path}: {str(e)}")
        return f"Error: {str(e)}", 404

if __name__ == '__main__':
    app.run(debug=True)