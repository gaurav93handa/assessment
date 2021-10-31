import cv2
import os
import shutil
import glob
import numpy as np


# To create a new dir (delete existing):
def cretae_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)

# To convert video to frames:
def vid_to_frames(video_file, frames_folder):

    # Create fresh output dir:
    cretae_dir(frames_folder)

    #Save each frame from video to that folder:
    vidcap = cv2.VideoCapture(video_file)
    success,frame = vidcap.read()
    count = 0
    while success:
        print("writing frame {}".format(count+1))
        cv2.imwrite("{}/frame_{}.png".format(frames_folder,count),frame)
        success, frame = vidcap.read()
        count += 1

# Split left-right camera images from combined frame
def img_split(input_folder,output_folder):

    # Create output subfolders left & right
    left_cam_dir = "{}/Left_cam".format(output_folder)
    right_cam_dir = "{}/Right_cam".format(output_folder)
    cretae_dir(left_cam_dir)
    cretae_dir(right_cam_dir)

    assert os.path.exists(input_folder), "Input folder:{} doesn't exists!".format(input_folder)

    # Get list of all images from input folder:
    img_path_list = glob.glob("{}/*.png".format(input_folder))+glob.glob("{}/*.jpg".format(input_folder))
    assert len(img_path_list)>0, "Input folder:{} doesn't have any image!"

    # Loop through each image and save it in respective folder:
    for image_path in img_path_list:
        image_path = image_path.replace("\\","/")
        image_name = image_path.rsplit("/",1)[1].rsplit('.',1)[0]
        image = cv2.imread(image_path)
        width = image.shape[1]
        height = image.shape[0]
        width_mid = width//2
        img_left = image[:,:width_mid]
        img_right = image[:,width_mid:]
        print("processing: {}".format(image_name))
        cv2.imwrite("{}/{}_left.png".format(left_cam_dir,image_name),img_left)
        cv2.imwrite("{}/{}_right.png".format(right_cam_dir, image_name), img_right)



# Get the rectified images:
def get_rectifiedImg(input_folder,output_folder):
    assert os.path.exists(input_folder), "dir not  found : {}".format(input_folder)
    cretae_dir(output_folder)

    #TODO: Hardcoded camera parameters
    cameraMatrix_Left = np.array([[1048.62, 0, 1081.38], [0, 1048.62, 607.91], [0, 0, 1]])
    cameraMatrix_Right = np.array([[1048.62, 0, 1081.38], [0, 1048.62, 607.91], [0, 0, 1]])
    distCoeffs_Left = np.array([[-0.175995, 0.0265533,0, 0, 0]])
    distCoeffs_Right = np.array([[-0.175152, 0.0267266, 0, 0, 0]])

    image_list = glob.glob(input_folder+"/*.png")+glob.glob(input_folder+"/*.jpg")

    for imag_file in image_list:
        image_name = imag_file.rsplit("\\",1)[1]
        img = cv2.imread(imag_file)
        undistorted = cv2.undistort(img, cameraMatrix_Left,distCoeffs_Left)
        print("undistorting: {}".format(image_name))
        cv2.imwrite("{}/undist_{}".format(output_folder,image_name),undistorted)

# Trying Stereo match algo
# def stereo_match():
#     imgL = cv2.imread(r"C:\Users\gaura\Desktop\ladybug_assessment\cameras\Left_cam\frame_37_left.png")
#     imgR = cv2.imread(r"C:\Users\gaura\Desktop\ladybug_assessment\cameras\Right_cam\frame_37_right.png")
#
#     # disparity range is tuned for 'aloe' image pair
#     window_size = 3
#     min_disp = 16
#     num_disp = 112-min_disp
#     stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
#         numDisparities = num_disp,
#         blockSize = 7,
#         P1 = 8*3*window_size**2,
#         P2 = 32*3*window_size**2,
#         disp12MaxDiff = 10,
#         uniquenessRatio = 10,
#         speckleWindowSize = 100,
#         speckleRange = 32
#     )
#
#     print('computing disparity...')
#     disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
#
#     print('generating 3d point cloud...',)
#     h, w = imgL.shape[:2]
#     f = 1048.62                        # guess for focal length
#     Q = np.float32([[1, 0, 0, -0.5*w],
#                     [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
#                     [0, 0, 0,     -f], # so that y-axis looks up
#                     [0, 0, 1,      0]])
#     points = cv2.reprojectImageTo3D(disp, Q)
#     colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
#     mask = disp > disp.min()
#     out_points = points[mask]
#     out_colors = colors[mask]
#     out_fn = r'C:\Users\gaura\Desktop\ladybug_assessment\out.ply'
#     write_ply(out_fn, out_points, out_colors)
#     print('%s saved' % out_fn)
#
#     cv2.imshow('left', imgL)
#     cv2.imshow('disparity', (disp-min_disp)/num_disp)
#     cv2.waitKey()
#
#     print('Done')


# Create and save depth maps:
def get_disparity(left_cam_dir,right_cam_dir,output_dir, block_size):
    assert os.path.exists(left_cam_dir),"dir not  found : {}".format(left_cam_dir)
    assert os.path.exists(left_cam_dir), "dir not  found : {}".format(right_cam_dir)
    cretae_dir(output_dir)
    left_cam_images = glob.glob(left_cam_dir+"/*.png") + glob.glob(left_cam_dir+"/*.jpg")
    right_cam_images = glob.glob(right_cam_dir + "/*.png") + glob.glob(right_cam_dir + "/*.jpg")
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=block_size)

    for left_img_path in left_cam_images:
        image_key_name = left_img_path.rsplit("\\", 1)[1].rsplit('.', 1)[0].rsplit('_left',1)[0]
        print(image_key_name)
        right_img_path = "{}/{}_right.png".format(right_cam_dir,image_key_name)
        if os.path.exists(right_img_path):
            imgL = cv2.imread(left_img_path, 0)
            imgR = cv2.imread(right_img_path, 0)
            disparity = stereo.compute(imgL, imgR)
            print(disparity.min(),disparity.max())
            cv2.imwrite("{}/{}_disparity.png".format(output_dir,image_key_name),disparity)
            print("Writing disparity mpa for {}".format(image_key_name))
        else:
           print("Right image does not exists for {}".format(image_key_name))

def get_masked_disparity(disp_dir,mask_dir,output_dir):

    mask_imgs = glob.glob(mask_dir+"/*.png")+glob.glob(mask_dir+"/*.jpg")
    cretae_dir(output_dir)
    for mask in mask_imgs:
        img_name = mask.rsplit("\\",1)[1]
        frame_index = ''.join(x for x in img_name if x.isdigit())
        print(frame_index)
        disp = "{}/frame_{}_disparity.png".format(disp_dir,frame_index)
        if os.path.exists(disp):
            print("yes")
            mask_img = cv2.imread(mask)
            disp_img = cv2.imread(disp)
            out_img = np.zeros_like(disp_img)
            out_img[mask_img==255]=disp_img[mask_img==255]
            cv2.imwrite("{}/frame_{}_mask_disp.png".format(output_dir,frame_index),out_img)

def combine_disparity(disp_dir_1,disp_dir_2, output_dir):
    disp_imgs_1 = glob.glob(disp_dir_1 + "/*.png") + glob.glob(disp_dir_1 + "/*.jpg")
    cretae_dir(output_dir)
    for disp_1 in disp_imgs_1:
        img_name = disp_1.rsplit("\\", 1)[1]
        frame_index = ''.join(x for x in img_name if x.isdigit())
        disp_2 = "{}/frame_{}_mask_disp.png".format(disp_dir_2,frame_index)
        if os.path.exists(disp_2):
            print(frame_index)
            disp_img_1 = cv2.imread(disp_1)
            disp_img_2 = cv2.imread(disp_2)
            out_img = np.copy(disp_img_2)
            out_img[disp_img_1>disp_img_2]=disp_img_1[disp_img_1>disp_img_2]
            cv2.imwrite("{}/frame_{}_comb_disp.png".format(output_dir,frame_index),out_img)



def get_mask(input_path,output_path):
    assert os.path.exists(input_path),"dir not  found : {}".format(input_path)
    cretae_dir(output_path)
    img_files = glob.glob("{}/*.png".format(input_path))
    for img in img_files:
        image_name = img.rsplit("\\",1)[1]
        # Read image
        img = cv2.imread(img)
        hh, ww = img.shape[:2]

        # threshold on white
        # Define lower and uppper limits
        lower = np.array([0, 0, 0])
        upper = np.array([50, 50, 50])

        # Create mask to only select black
        thresh = cv2.inRange(img, lower, upper)

        # apply mask to image
        result = cv2.bitwise_and(img, img, mask=thresh)
        print('{}/mask_{}'.format(output_path,image_name))
        cv2.imwrite('{}/mask_{}'.format(output_path,image_name), thresh)

def get_depthMap(disp_dir,output_dir,f,baseline):
    cretae_dir(output_dir)
    dis_files = glob.glob("{}/*.png".format(disp_dir))
    for dis_file in dis_files:
        img_name = dis_file.rsplit("\\", 1)[1]
        frame_index = ''.join(x for x in img_name if x.isdigit())
        disp_img = cv2.imread(dis_file)
        h = disp_img.shape[0]
        w = disp_img.shape[1]
        depth_map = np.zeros([h, w])
        for row in range(0,h):
            for col in range(0,w):
                if disp_img[row,col][0]!=0:
                    depth_map[row,col]= f*baseline/(16.0*disp_img[row,col][0])
                    print(f*baseline/(16.0*disp_img[row,col][0]))
        cv2.imwrite("{}/frame_{}_depth.png".format(output_dir,frame_index),depth_map)




if __name__ == "__main__":
    dir = "C:/Users/gaura/Desktop/ladybug_assessment"
    vid_to_frames(dir+"/video/stereo-video.mp4",dir+"/frames")
    img_split(dir+"/frames","C:/Users/gaura/Desktop/ladybug_assessment/cameras")
    get_rectifiedImg(dir+"/cameras/Left_cam",dir+"/cameras/Left_cam_undistorted")
    get_rectifiedImg(dir+"/cameras/Right_cam",dir+"/cameras/Right_cam_undistorted")
    get_disparity(dir+"/cameras/Left_cam",dir+"/cameras/Right_cam",dir+"/disparity_map_BS5",5)
    get_disparity(dir+"/cameras/Left_cam",dir+"/cameras/Right_cam",dir+"/disparity_map_BS7", 7)
    get_mask(dir+"/cameras/Left_cam",dir+"/mask/Left_cam")
    get_mask(dir+"/cameras/Right_cam",dir+"/mask/Right_cam")
    get_masked_disparity(dir+"/disparity_map_BS5",dir+"/mask/Left_cam",dir+"/masked_disparity_BS5")
    get_masked_disparity(dir + "/disparity_map_BS7", dir + "/mask/Left_cam", dir + "/masked_disparity_BS7")
    combine_disparity(dir+"/masked_disparity_BS5",dir+"/masked_disparity_BS7",dir+"/masked_disparity_combined")
    get_depthMap(dir+"/masked_disparity_combined",dir+"/depth_mask",1048.62,1.2)