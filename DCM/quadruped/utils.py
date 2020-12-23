import numpy as np
import imageio
import sys
import os

def lower_frame(inputpath, skip_rate=8, duration=None):
    outputpath = os.path.splitext(inputpath)[0] + "_lowerFrame.mp4"
    print("converting\r\n\t{0}\r\nto\r\n\t{1}".format(inputpath, outputpath))

    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, fps=fps)

    if duration != None and duration > 0:
        video_len = reader.get_meta_data()['duration']
        skip_rate = max(int(video_len/duration), 1)

    for i,im in enumerate(reader):
        if i%skip_rate != 0:
            continue
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)
    print("\r\nFinalizing...")
    writer.close()
    print("Done.")

def x_rot(t):
  rot = [[1.0,       0.0,        0.0],
          [0.0, np.cos(t), -np.sin(t)],
          [0.0, np.sin(t), np.cos(t)]]
  return np.array(rot)

def y_rot(t):
  rot = [[np.cos(t), 0.0, np.sin(t)],
          [      0.0, 1.0,       0.0],
        [-np.sin(t), 0.0, np.cos(t)]]
  return np.array(rot)

def z_rot(t):
  rot = [[np.cos(t), -np.sin(t), 0.0],
          [np.sin(t),  np.cos(t), 0.0],
          [      0.0,        0.0, 1.0]]
  return np.array(rot)

def rpy_rot(rpy):
    return np.matmul(z_rot(rpy[2]), np.matmul(y_rot(rpy[1]), x_rot(rpy[0])))

def diag_mat(diag):
  mat = np.eye(len(diag))
  for i in range(len(diag)):
    mat[i,i] = diag[i]
  return mat

def x_rot_dot(t):
  rot = [[0.0,       0.0,        0.0],
          [0.0, -np.sin(t), -np.cos(t)],
          [0.0, np.cos(t), -np.sin(t)]]
  return np.array(rot)

def y_rot_dot(t):
  rot = [[-np.sin(t), 0.0, np.cos(t)],
          [      0.0, 0.0,       0.0],
        [-np.cos(t), 0.0, -np.sin(t)]]
  return np.array(rot)

def z_rot_dot(t):
  rot = [[-np.sin(t), -np.cos(t), 0.0],
          [np.cos(t),  -np.sin(t), 0.0],
          [      0.0,        0.0, 0.0]]
  return np.array(rot)

if __name__ == "__main__":
    lower_frame("log.mp4", skip_rate=2, duration=20)