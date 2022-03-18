import numpy as np
import cv2

def find_ball(pred_image, image_ori, ratio_w, ratio_h):

    if np.amax(pred_image) <= 0: #no ball
        return image_ori

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_image, connectivity = 8)
    # print(type(stats))

    if len(stats): 
        stats = np.delete(stats, 0, axis = 0)
        centroids = np.delete(centroids, 0, axis = 0)

    x, y , w, h, area = stats[np.argmax(stats[:,-1])]
    x_cen, y_cen = centroids[np.argmax(stats[:,-1])]

    cv2.rectangle(image_ori, (int(x * ratio_w), int(y * ratio_h)), (int((x + w) * ratio_w), int((y + h) * ratio_h)), (255,0,0), 3)
    cv2.circle(image_ori, (int(x_cen * ratio_w), int(y_cen * ratio_h)),  3, (0,0,255), -1)


    #for i in range(len(stats)):
    #    x, y, w, h, area = stats[i]

    return image_ori

def find_ball_v2(pred_image, image_ori, ratio_w, ratio_h):

    if np.amax(pred_image) <= 0: #no ball
        return image_ori

    ball_cand_score = []

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_image, connectivity = 8)
    # print(type(stats))

    if len(stats): 
        stats = np.delete(stats, 0, axis = 0)
        centroids = np.delete(centroids, 0, axis = 0)

    for i in range(len(stats)):
        x, y, w, h, area = stats[i]

        score = np.mean(pred_image[y:y+h, x:x+w])

        ball_cand_score.append(score)

    ball_pos = stats[np.argmax(ball_cand_score)]
    x_cen, y_cen = centroids[np.argmax(ball_cand_score)]

    x, y, w, h, area = ball_pos

    radius = int((((x + w) * ratio_w) - (x * ratio_w)) / 2)

    x0, y0, x1, y1 = int(x * ratio_w), int(y * ratio_h), int((x + w) * ratio_w), int((y + h) * ratio_h)

    plot_one_box([x0, y0, x1, y1], image_ori, label='tennis ball', color=(25,50,255), line_thickness=3)

    #cv2.rectangle(image_ori, (int(x * ratio_w), int(y * ratio_h)), (int((x + w) * ratio_w), int((y + h) * ratio_h)), (255,0,0), 3)
    #cv2.circle(image_ori, (int(x_cen * ratio_w), int(y_cen * ratio_h)),  3, (0,0,255), -1)

    return image_ori

def tran_input_img(img_list):

    trans_img = []

    #for i in reversed(range(len(img_list))):
    for i in range(len(img_list)):

        img = img_list[i]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #img = cv2.resize(img,(WIDTH, HEIGHT))
        img = np.asarray(img).transpose(2, 0, 1)

        trans_img.append(img[0])
        trans_img.append(img[1])
        trans_img.append(img[2])

    trans_img = np.asarray(trans_img)

    return trans_img.reshape(1,trans_img.shape[0],trans_img.shape[1],trans_img.shape[2])


def ball_segmentation(image_ori, image_pred, width, height):

    """ret, y_pred = cv2.threshold(image_pred,50,255, cv2.THRESH_BINARY)
    y_pred_rgb = cv2.cvtColor(y_pred, cv2.COLOR_GRAY2RGB)

    y_pred_rgb[...,0] = 0
    y_pred_rgb[...,1] = 0

    y_pred_rgb = cv2.resize(y_pred_rgb,(width, height))
    y_pred_rgb = cv2.resize(y_jet,(width, height))
    img = cv2.addWeighted(image_ori, 1, y_pred_rgb, 0.8, 0)
    """

    y_jet = cv2.applyColorMap(image_pred, cv2.COLORMAP_JET)
    y_jet = cv2.resize(y_jet,(width, height))

    img = cv2.addWeighted(image_ori, 1, y_jet, 0.3, 0)

    return img

def WBCE(y_pred, y_true):
    eps = 1e-7
    loss = (-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) +
            torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
    return torch.mean(loss)


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
