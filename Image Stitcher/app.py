from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import io
import base64

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import random
from tqdm.notebook import tqdm

plt.rcParams['figure.figsize'] = [15, 15]

def resize_images(image_list):
    # Find the maximum dimensions among all images
    max_width = max(img.shape[1] for _, _, img in image_list)
    max_height = max(img.shape[0] for _, _, img in image_list)

    # Resize all images to the maximum dimensions
    resized_images = []
    for gray, _, rgb in image_list:
        resized_gray = cv2.resize(gray, (max_width, max_height))
        resized_rgb = cv2.resize(rgb, (max_width, max_height))
        resized_images.append((resized_gray, None, resized_rgb))

    return resized_images

# Read image and convert them to gray!!
def read_image(path, orientation):
    img = cv2.imread(path)
    if orientation == 'vertical':
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

def SIFT(img):
    siftDetector = cv2.SIFT_create()
    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des

def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches

def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2]  # standardize to let w*H[2,2] = 1
    return H

def random_point(matches, k=4):
    if k > len(matches):
        k = len(matches)
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2]
    errors = np.linalg.norm(all_p2 - estimate_p2, axis=1) ** 2
    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)

        if np.linalg.matrix_rank(H) < 3:
            continue

        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()

    # print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

def stitch_images(image_list):
    num_images = len(image_list)
    homographies = []
    for i in range(num_images - 1):
        left_gray, _, left_rgb = image_list[i]
        right_gray, _, right_rgb = image_list[i + 1]

        kp_left, des_left = SIFT(left_gray)
        kp_right, des_right = SIFT(right_gray)

        matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)

        inliers, H = ransac(matches, 0.5, 2000)
        homographies.append(H)

    stitched_image = image_list[0][2]
    for i in range(1, num_images):
        stitched_image = stitch_img(stitched_image, image_list[i][2], homographies[i - 1])
        plt.imshow(stitched_image)
        stitched_image_path = 'static/images/stitched_image.png'
        plt.savefig(stitched_image_path)
    
    return stitched_image

def stitch_img(left, right, H):
    left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    height_l, width_l, _ = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)

    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, _ = right.shape
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)

    black = np.zeros(3)  

    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]

            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass

    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image

app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stitch_images', methods=['POST'])
def stitch_images_route():
    num_images = int(request.form['num_images'])
    image_paths = request.form.getlist('image_paths[]')
    orientation = request.form['orientation']

    image_list = []
    # Stitch the images together
    # You may need to modify the stitch_images function to accept a list of file paths
    for i in range(num_images):
        path = 'images/'+image_paths[i]
        gray, _, rgb = read_image(path, orientation)
        image_list.append((gray, _, rgb))

    image_list_resized = resize_images(image_list)

    stitched_image = stitch_images(image_list_resized)
    

    # Convert the stitched image to base64 string
    _, img_encoded = cv2.imencode('.png', stitched_image)
    stitched_image_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Return the stitched image as a response
    return jsonify({'stitched_image': stitched_image_base64})

if __name__ == "__main__":
    app.run(debug=True)
