import imageio
import sys
import cv2

class Renderer:
    def __init__(self, env, fps):
        self.env = env
        self.fps = fps
        self.RENDER_WIDTH, self.RENDER_HEIGHT = 800, 600        
        
    
    def reset(self):
        self.global_t = 0.0
        self.img_list = []
    
    def render(self, time_step):
        self.global_t += time_step

        view_matrix = self.env.pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.env.camTargetPos,
            distance=self.env.camDist,
            yaw=self.env.camYaw,
            pitch=self.env.camPitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self.env.pybullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.RENDER_WIDTH)/self.RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)

        while self.global_t*self.fps > len(self.img_list):
            width, height, rgb_img, depth_img, seg_img = self.env.pybullet_client.getCameraImage(
                width=self.RENDER_WIDTH, 
                height=self.RENDER_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
            rgb_img = rgb_img[:,:,:3]
            self.img_list.append(rgb_img)
                
    def save_video(self):
        output = "./result.mp4"
        writer = imageio.get_writer(output, fps=self.fps)
        for i, img in enumerate(self.img_list):
            sys.stdout.write("\rframe {0}".format(i))
            sys.stdout.flush()
            writer.append_data(img)
        print("\r\nFinalizing...")
        writer.close()
