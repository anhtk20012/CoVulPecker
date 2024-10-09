import subprocess
from pathlib import Path

class Joern2Dot():
    def __init__(self, path_file):
        # Prepare the path for Docker
        path_folder = "/" + str(Path(path_file).parent).replace(':','') + ":/home/ubuntu:rw"
        path_folder = path_folder.replace('\\', '/')

        # Start the Joern container
        self.start_docker(path_folder)
        self.code2bin()
        self.bin2dots()
        self.stopdocker()
        
    def start_docker(self, path_folder):
        try:
            result = subprocess.run(['docker', 'inspect', '--format={{.State.Running}}', 'joern_teamq'],
                check=True,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            if result.stdout.strip() != "true":
                print("Container joern_teamq is not running. Restarting...")
                subprocess.run(['docker', 'start', 'joern_teamq'], check=True, stdout=subprocess.PIPE)
                print("Container joern_teamq has been restarted.")
                
        except subprocess.CalledProcessError:
            print("Container joern_teamq does not exist")
            # If the container is not running, start a new container
            subprocess.run([
                'docker', 'run', '-d',
                '--name', 'joern_teamq',
                '-v', '/tmp:/tmp',
                '-v', f'{path_folder}',
                '-w', '/home/ubuntu',
                '-t', 'anhtk20012/joern-teamq:latest'
            ], check=True, stdout=subprocess.PIPE)
            print("Started a new container joern_teamq.")
                
    def code2bin(self):
        # Run the command to convert code to binary file
        try:
            subprocess.run([
                'docker', 'exec',
                'joern_teamq', 
                '/bin/bash', '-c', 'joern-parse -o ./tmp/code.bin ./tmp/code.c'
            ], check=True, stdout=subprocess.PIPE)
            print("Successfully converted code to binary file.")
        except subprocess.CalledProcessError as e:
            print(f"Error while running code2bin: {e}")
            
    def bin2dots(self):
        # Run the command to convert binary file to DOT file
        try:
            subprocess.run([
                'docker', 'exec',
                'joern_teamq', 
                '/bin/bash', '-c', 'joern-export --repr cpg14 --out ./tmp/code.dot ./tmp/code.bin'
            ], check=True, stdout=subprocess.PIPE)
            print("Successfully converted binary file to DOT file.")
        except subprocess.CalledProcessError as e:
            print(f"Error while running bin2dots: {e}")
            
    def stopdocker(self):
        # Stop the Joern container
        try:
            subprocess.run(['docker', 'stop', 'joern_teamq'], check=True, stdout=subprocess.PIPE)
            subprocess.run(['docker', 'rm', 'joern_teamq'], check=True, stdout=subprocess.PIPE)
            print("Container joern_teamq has been stopped.")
        except subprocess.CalledProcessError as e:
            print(f"Error while stopping the container: {e}")

