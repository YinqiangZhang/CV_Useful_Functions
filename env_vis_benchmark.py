import numpy as np
import cv2
import os


class ResultsEvaluation(object):
    """
    The class of results evaluation
    """

    def __init__(self, txt_results_name, txt_results_dir, video_save_dir, raw_img_dir, enable_visual=True, enable_video=True):
        """
        The init parameters of results evaluation class:
        :param txt_results_name: the name of txt results to be evaluated
        :param txt_results_dir:  the directory of txt results to be evaluated
        :param video_save_dir:   the save directory of the created videos
        :param raw_img_dir:      the image files which is need by the video writing
        :param enable_video:     the flag whether to create the videos (default: True)
        :param enable_visual:    the flag whether to enable the visualisation (default: True)
        """
        self.env_name = txt_results_name
        self.env_data_dir = txt_results_dir
        self.video_results_dir = video_save_dir
        self.raw_img_dir = raw_img_dir

        self.COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
                        (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
                        (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
                        (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
                        (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
                        (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
                        (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
                        (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
                        (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
                        (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
                        (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

        self.enable_video = enable_video
        self.enable_visual = enable_visual
        self.fps = 30
        self.size = (1920, 1080)

        # check if the corresponding directories (results and video path) exist
        if not os.path.exists(self.env_data_dir):
            os.makedirs(self.env_data_dir)
        if (not os.path.exists(self.video_results_dir)) and self.enable_video:
            os.makedirs(self.video_results_dir)

    def __call__(self):

        dataset_path = os.path.join(self.env_data_dir, self.env_name)
        for ind, file in enumerate(os.listdir(dataset_path), 1):
            filename = file.split(sep='.')[0]
            print('the {}. dataset is: {}'.format(ind, file))

            txt_path = os.path.join(dataset_path, file)
            img_dir = os.path.join(self.raw_img_dir, filename, 'img1')
            # the list of dataset images
            img_list = (img for img in os.listdir(img_dir))

            # read the results file from MOT algorithm
            res_data = np.loadtxt(txt_path, delimiter=',')

            if self.enable_video:
                video_path = os.path.join(self.video_results_dir, '{}.mp4'.format(filename))
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, self.size)

            for i, img in enumerate(img_list, 1):
                img_path = os.path.join(img_dir, img)
                ori_im = cv2.imread(img_path)
                im = ori_im

                records = res_data[res_data[:, 0] == float(i)]
                # record the identifier and bounding box
                id_num = records[:, 1]
                bbox = records[:, 2:]
                for k in range(len(id_num)):
                    xl, yt, w, h = [int(i) for i in bbox[k, :]]
                    identifier = int(id_num[k])
                    xr = xl + w - 1
                    yb = yt + h - 1

                    color = self.COLORS_10[identifier % len(self.COLORS_10)]
                    label = '{}{:d}'.format("", identifier)

                    # draw the rectangle:
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                    cv2.rectangle(im, (xl, yt), (xr, yb), color, 3)
                    cv2.rectangle(im, (xl, yt), (xl + t_size[0] + 3, yt + t_size[1] + 4), color, -1)
                    cv2.putText(im, label, (xl, yt + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

                if self.enable_video:
                    writer.write(im)
                    print('Video {}: Frame {}'.format(filename, i))
                if self.enable_visual:
                    # show the frame
                    cv2.imshow("env", im)
                    cv2.waitKey(int(1000/self.fps))


if __name__ == '__main__':

    env_name = 'CenterNet_ECO'
    env_data_dir = '.\\results'
    vis_results_dir = '.\\video'
    raw_img_dir = '..\\DMAN_MOT-master\\data\\MOT_ZJ\\train'

    enable_video = True
    enable_visual = False

    res_env = ResultsEvaluation(env_name, env_data_dir, vis_results_dir,
                                raw_img_dir, enable_visual, enable_video)

    res_env()
