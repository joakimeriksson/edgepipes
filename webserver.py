from flask import Flask, render_template
import base64, json
import cv2
import calculators.image
import threading
from flask_socketio import SocketIO, emit
import requests, io
from contextlib import redirect_stdout
from pyvis.network import Network

class ServerThread(threading.Thread):

    def __init__(self, socketio, app):
        threading.Thread.__init__(self)
        self.app = app
        self.socketio = socketio

    def run(self):
        print("starting server at http://127.0.0.1:5000")
        self.socketio.run(self.app)

    def shutdown(self):
        print("stopping server")
        r = requests.get('http://127.0.0.1:5000/stop')
        print("Result: " + r.text)

app = Flask(__name__,)
pipeline = None
server = None
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
cli = None

@app.route('/data/<name>')
def data(name):
    nodes = pipeline.get_node_by_output(name)
    if len(nodes) == 0:
        return "can not find stream data: " + name
    else:
        d = nodes[0]
        index = d.get_output_index(name)
        out = d.get_output(index)
        if isinstance(out, calculators.image.ImageData):
            img =  b'data:image/jpeg;base64,' + base64.encodebytes(cv2.imencode('.jpeg',  out.image)[1].tostring())
            val = "Image:<img src=\"" + img.decode('ascii') + "\"/>";
        else:
            val = str(out)
        return val;

@app.route('/')
def hello():
    data = "<ul>"
    for n in pipeline.pipeline:
        data += "<li>Node:" +  n.name + "<ul>"
        data += '<li>input :' + str(n.input)
        data += '<li>output:'
        for out in n.output:
            data += '<a href="/data/' + str(out) + '">' + str(out) + '</a>'
        data += '</ul>'
    return "EdgePipes - pipeline." + data + "</ul>"

@app.route('/graph')
def graph():
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", heading="Pipeline", directed=True)
    labels = dict()
    # First add all nodes
    for n in pipeline.pipeline:
        print("Adding node:" + n.name)
        net.add_node(n.name, n.name, title=n.name, shape="box")

    for n in pipeline.pipeline:
        # Add input edges
        for ni in n.input:
            nodes = pipeline.get_node_by_output(ni)
            if len(nodes) > 0:
                net.add_edge(nodes[0].name, n.name)
                labels[(n.name, nodes[0].name)] = ni
    net.show("graph.html")
    return open("graph.html", "r").read()

@app.route("/stop")
def stop():
    socketio.stop()
    return "Stopping server."

@app.route("/tcli")
def index():
    return render_template("index.html")

@socketio.on('chat message')
def test_message(message):
    print("Received message:", message)
    f = io.StringIO()
    with redirect_stdout(f):
        cli.onecmd(message)
    emit('chat message', f.getvalue().replace("\n", "<br>"))
    f.close()

# This will run the web-server
def start(pipes, cmd=None):
    global pipeline, server, socketio, cli
    pipeline = pipes
    server = ServerThread(socketio, app)
    server.start()
    cli = cmd

def stop():
    global server
    server.shutdown()
