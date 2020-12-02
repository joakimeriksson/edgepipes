#!/usr/bin/env python3
#
import argparse
import os
import sys
import threading
import edgepipes
import cv2
from calculators.audio import get_pyaudio
from calculators.core import SwitchNode
from cmd import Cmd
import webserver, sys

wsthread = None



class PipeCli(Cmd):
    prompt = 'pipecli> '

    def __init__(self, completekey='tab', stdin=None, stdout=None):
        Cmd.__init__(self, completekey=completekey, stdin=stdin, stdout=stdout)
        self.pipeline = edgepipes.Pipeline()
        self.ctr = 1
        self.options = {}

    def do_exit(self, inp):
        """exit the application."""
        print("Bye")
        self.pipeline.exit()
        webserver.stop()
        return True

    def do_setvideo(self, inp):
        """set the current (default) video stream input. (setvideo <source>)"""
        print("Set video input '{}'".format(inp))
        self.options['input_video'] = {'video': inp}

    def do_setaudio(self, inp):
        """set the current (default) audio stream input. (setaudio <source>)"""
        print("Set audio input '{}'".format(inp))
        self.options['input_audio'] = {'audio': inp}

    def do_list(self, inp):
        """list the available audio or video input devices"""
        if inp == 'audio':
            paud = get_pyaudio()
            info = paud.get_host_api_info_by_index(0)
            device_count = info.get('deviceCount')
            print("Available audio output devices:")
            for i in range(0, device_count):
                device_info = paud.get_device_info_by_host_api_device_index(0, i)
                if (device_info.get('maxOutputChannels')) > 0:
                    print(f"  Audio Output Device index {i} - {device_info.get('name')}")
            print()
            print("Available audio input devices:")
            for i in range(0, device_count):
                device_info = paud.get_device_info_by_host_api_device_index(0, i)
                if (device_info.get('maxInputChannels')) > 0:
                    print(f"  Audio Input Device index {i} - {device_info.get('name')}")
        elif inp == 'video':
            ports = []
            dev_port = 0
            while True:
                camera = cv2.VideoCapture(dev_port)
                if not camera.isOpened():
                    break
                is_frame_read, img = camera.read()
                w, h = int(camera.get(3)), int(camera.get(4))
                ports.append((dev_port, is_frame_read, w, h))
                camera.release()
                dev_port += 1
            if ports:
                print("Available video input devices:")
                for port in ports:
                    if port[1]:
                        print(f"  Port {port[0]} ({port[2]} x {port[3]})")
                    else:
                        print(f"  Port {port[0]} ({port[2]} x {port[3]}) - failed to read")
        else:
            print("Unknown option to list")

    def do_togglestate(self, inp):
        nodes = self.pipeline.get_nodes_by_type(SwitchNode)
        if nodes:
            for n in nodes:
                n.toggle_state()
                print(f"Toggled state in {n.name} => {n.get_switch_state()}")
        else:
            print("Found no switch nodes to toggle")

    def do_print(self, inp):
        for n in self.pipeline.pipeline:
            print("N:", n.name)
            print("  Time consumed:", self.pipeline.elapsed[n.name], self.pipeline.elapsed[n.name] / self.pipeline.count[n.name] )
            print("  input :", n.input)
            print("  output:", n.output)
        print("Done...")

    def do_webserver(self, inp):
        print("Starting webserver")
        webserver.start(self.pipeline, cmd=self)

    def emptyline(self):
        return

    def do_load(self, inp):
        if len(inp) == 0:
            files = [f for f in os.listdir("graphs")]
            print("Available pipelines (in graphs):")
            for file in files:
                print(file)
        else:
            print("Loading pipeline from ", inp)
            try:
                f = open(inp, "r")
            except:
                try:
                    f = open("graphs/" + inp, "r")
                except:
                    print("File not found:", inp)
                    return
            txt = f.read()
            f.close()
            print("Load graphs: '{}'".format(txt))
            self.pipeline.setup_pipeline(txt, prefix=str(self.ctr) + ".", options=self.options)
            self.ctr += 1

    def do_start(self, inp):
        if not self.pipeline.run_pipeline:
            self.pipeline.start()

    def do_stop(self, inp):
        self.pipeline.stop()

    def do_step(self, inp):
        self.pipeline.step()

if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        p = argparse.ArgumentParser()
        p.add_argument('--input', dest='input_video', default=None, help='video stream input')
        p.add_argument('--input_audio', dest='input_audio', default=None, help='audio stream input')
        p.add_argument('pipeline', nargs='?')
        conopts = p.parse_args(args)
    except Exception as e:
        sys.exit(f"Illegal arguments: {e}")

    pipeline_graph = None
    if conopts.pipeline:
        print(f"Loading pipeline from {conopts.pipeline}")
        try:
            with open(conopts.pipeline, "r") as f:
                pipeline_graph = f.read()
        except FileNotFoundError:
            sys.exit(f"Could not find the pipeline config file {conopts.pipeline}")

    # Setup the CLI and start a separate thread for that - as main is needed for the CV processing.
    p = PipeCli()

    opts = {}
    if conopts.input_video:
        video = int(conopts.input_video) if conopts.input_video.isnumeric() else conopts.input_video
        opts['input_video'] = {'video': video}
    if conopts.input_audio:
        audio = int(conopts.input_audio) if conopts.input_audio.isnumeric() else conopts.input_audio
        opts['input_audio'] = {'audio': audio}
    p.options = opts

    if pipeline_graph:
        p.pipeline.setup_pipeline(pipeline_graph, options=opts)
        p.pipeline.start()
    thread = threading.Thread(target=p.cmdloop)
    thread.start()
    p.pipeline.run()
    thread.join()
