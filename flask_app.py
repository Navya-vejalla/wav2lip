from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
import os
import tempfile
import subprocess
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='front_end')
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Allowed file extensions
ALLOWED_AUDIO_VIDEO_EXTENSIONS = {'mp3', 'mp4', 'wav'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        # Check if the post request has the files
        if 'file1' not in request.files or 'file2' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file1 = request.files['file1']  # Video file
        file2 = request.files['file2']  # Audio file

        # Validate the files
        if not (file1 and allowed_file(file1.filename, ALLOWED_AUDIO_VIDEO_EXTENSIONS)):
            flash('File 1 must be an audio/video file (mp3, mp4, wav).')
            return redirect(request.url)
        if not (file2 and allowed_file(file2.filename, ALLOWED_AUDIO_VIDEO_EXTENSIONS)):
            flash('File 2 must be an audio/video file (mp3, mp4, wav).')
            return redirect(request.url)

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            # Secure filenames and save them
            file1_path = os.path.join(tempdir, secure_filename(file1.filename))
            file2_path = os.path.join(tempdir, secure_filename(file2.filename))
            file1.save(file1_path)
            file2.save(file2_path)

            # Path for the checkpoint file
            checkpoint_path = "checkpoints/wav2lip_gan.pth"
            # Run the inference script with the file paths as arguments
            try:
                logging.error("Starting ")

                result = subprocess.run(
                    ['python', 'inference.py', '--checkpoint_path', checkpoint_path, '--face', file1_path, '--audio', file2_path],
                    capture_output=True, text=True
                )
                #logging.error(f"Output saved to {output_file}")
                #output_url = url_for('serve_results', filename="result_voice.mp4")
                output = result.stdout or result.stderr
                return f"Output: {output}"
            except Exception as e:
                logging.error(f"Error running inference script: {e}")
                return "An error occurred while processing the files."

    return render_template('file_upload.html')


# @app.route('/results/<path:filename>')
# def serve_results(filename):
#     return send_from_directory(RESULTS_DIR, filename)

if __name__ == '__main__':
    #app.run(host="0.0.0.0", port=8080, debug=True)  # Set to False in production
    app.run(debug=True)









# from flask import Flask, request, jsonify, render_template, send_from_directory
# import os
# import tempfile
# import subprocess
# import logging
# import threading
# import uuid
# from werkzeug.utils import secure_filename
# from flask_cors import CORS

# app = Flask(__name__, template_folder='front_end')
# CORS(app)

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Folder to store output videos
# OUTPUT_FOLDER = '\\Wav2Lip-master\\results'
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # Allowed file extensions
# ALLOWED_AUDIO_VIDEO_EXTENSIONS = {'mp3', 'mp4', 'wav'}

# def allowed_file(filename, allowed_extensions):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# @app.route('/', methods=['GET', 'POST'])
# def upload_files():
#     if request.method == 'POST':
#         if 'file1' not in request.files or 'file2' not in request.files:
#             return jsonify({'error': 'No file part'}), 400

#         file1 = request.files['file1']
#         file2 = request.files['file2']

#         if not (file1 and allowed_file(file1.filename, ALLOWED_AUDIO_VIDEO_EXTENSIONS)):
#             return jsonify({'error': 'File 1 must be an audio/video file (mp3, mp4, wav).'}), 400

#         if not (file2 and allowed_file(file2.filename, ALLOWED_AUDIO_VIDEO_EXTENSIONS)):
#             return jsonify({'error': 'File 2 must be an audio/video file (mp3, mp4, wav).'}), 400

#         # Create a temporary directory for file storage
#         with tempfile.TemporaryDirectory() as tempdir:
#             # Secure filenames and save them
#             file1_path = os.path.join(tempdir, secure_filename(file1.filename))
#             file2_path = os.path.join(tempdir, secure_filename(file2.filename))
#             file1.save(file1_path)
#             file2.save(file2_path)
#             print("Files saved")

#             # Path for the output video
#             unique_id = str(uuid.uuid4())
#             output_video_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_output_video.mp4")

#             # Path for the checkpoint file
#             checkpoint_path = "checkpoints/wav2lip_gan.pth"

#             # Run the inference script with the file paths as arguments
#             try:
#                 print("Subprocess start")
#                 result = subprocess.run(
#                     ['python', 'inference.py', '--checkpoint_path', checkpoint_path, '--face', file1_path, '--audio', file2_path, '--outfile', output_video_path],
#                     capture_output=True, text=True
#                 )
#                 print(output_video_path)
#                 # print(result)
#                 if result.returncode == 0:
#                     logging.info(f"Video created successfully: {output_video_path}")
#                     #return jsonify({'message': 'Processing completed.', 'output_video': "/Wav2Lip-master/results/result_voice.mp4"}), 200
#                 else:
#                     logging.error(f"Error during processing: {result.stderr}")
#                     print(result.stderr)
#                     return jsonify({'error': result.stderr}), 500
#             except Exception as e:
#                 print(e)
#                 logging.error(f"Error running inference script: {e}")
#                 return jsonify({'error': 'An error occurred during processing.'}), 500
#         return jsonify({'message': 'Processing completed.', 'output_video': f"/output/{os.path.basename(output_video_path)}"}), 202

#     return render_template('file_upload.html')

# # Route to serve the generated video
# @app.route('/output/<path:filename>')
# def serve_video(filename):
#     return send_from_directory('results', filename)

# # @app.route('/output/<path:filename>')
# # def serve_video(filename):
# #     logging.info("Serve video")
# #     print("In serve video")
# #     return send_from_directory(OUTPUT_FOLDER, filename)

# if __name__ == '__main__':
#     app.run(debug=True)  # Set to False in production